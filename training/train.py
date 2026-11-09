"""
training/train.py  –  FAZ 4: Model Eğitimi
===========================================
Graph2Seq mimarisini (HourGlassGNNEncoder + AutoregressiveMeshDecoder)
teacher-forcing ile eğiten ve en iyi checkpoint'i kaydeden ana script.

Bölümler:
    1. Config      – tüm hiperparametreler ve dosya yolları
    2. Dataset     – .pt dosyalarını tarayan MeshDataset + collate_fn
    3. Graph2Seq   – encoder + decoder'ı birleştiren ana model
    4. Loss        – CrossEntropyLoss + calculate_stitch_loss (placeholder)
    5. Trainer     – train_epoch / eval_epoch / checkpoint yönetimi
    6. __main__    – veri yükleme, model oluşturma, eğitimi başlatma

Kullanım:
    python -m training.train
    # veya proje kökünden doğrudan:
    python training/train.py
"""

from __future__ import annotations
from pathlib import Path as _P
_REPO_ROOT = _P(__file__).resolve().parents[1]
_WORKSPACE = _REPO_ROOT.parent  # expected layout: <workspace>/DATA and <workspace>/<this repo>

import json
import math
import sys
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader, random_split
from torch_geometric.data import Batch, Data
from tqdm import tqdm

# Proje kökünü sys.path'e ekle (modül veya doğrudan çalıştırma için)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models import HourGlassGNNEncoder, AutoregressiveMeshDecoder


# =========================================================================== #
#  1. Yapılandırma (Configuration)                                             #
# =========================================================================== #

@dataclass
class Config:
    """
    Tüm hiperparametreleri ve dosya yollarını tek bir yerde tutar.

    Eğitim başlatılmadan önce bu sınıfı örnekleyin ve gerekirse
    alanları geçersiz kılın (override edin).
    """

    # ── Dosya Yolları ──────────────────────────────────────────────────────────
    PT_FILES_PATH: str = (
        str(_WORKSPACE / "DATA" / "PT_FILES")
    )
    VOCAB_PATH: str = (
        str(_REPO_ROOT / "data_vocabulary" / "ai_vocabulary.json")
    )
    CHECKPOINT_PATH: str = (
        str(_REPO_ROOT / "checkpoints")
    )

    # ── Encoder Hiperparametreleri ─────────────────────────────────────────────
    IN_CHANNELS: int      = 4       # data.x: [valence, AR, skewness, jacobian]
    HIDDEN_DIM: int       = 256
    ENC_NUM_LAYERS: int   = 4
    ENC_NUM_HEADS: int    = 4
    ENC_DROPOUT: float    = 0.1
    ENC_POOLING: str      = "mean"  # "mean" | "max" | "mean+max"

    # ── Decoder Hiperparametreleri ─────────────────────────────────────────────
    VOCAB_SIZE: int          = 64    # GPU hizalaması; gerçek vocab boyutu: 33
    DEC_NUM_LAYERS: int      = 4
    DEC_NHEAD: int           = 8
    DEC_DIM_FEEDFORWARD: int = 1024
    DEC_DROPOUT: float       = 0.1
    DEC_MAX_SEQ_LEN: int     = 2048

    # ── Eğitim Hiperparametreleri ──────────────────────────────────────────────
    BATCH_SIZE: int    = 4
    LEARNING_RATE: float = 3e-4
    WEIGHT_DECAY: float  = 1e-4
    EPOCHS: int        = 50
    GRAD_CLIP: float   = 1.0    # Gradient clipping maksimum L2 normu
    TRAIN_RATIO: float = 0.85   # Eğitim / validation bölünme oranı
    NUM_WORKERS: int   = 0      # Windows'ta fork güvenilmez; 0 önerilir

    # ── Kayıp Fonksiyonu ──────────────────────────────────────────────────────
    PAD_TOKEN_ID: int = 0    # CrossEntropyLoss ignore_index
    ALPHA: float      = 0.1  # L_total = L_CE + alpha * L_Stitch

    # ── Cihaz ─────────────────────────────────────────────────────────────────
    DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"


# =========================================================================== #
#  2. Veri Yükleme                                                             #
# =========================================================================== #

class MeshDataset(Dataset):
    """
    PT_FILES_PATH klasöründeki .pt dosyalarını tarayan veri kümesi.

    FAZ 2 pipeline'ı tarafından üretilen her .pt dosyası bir
    torch_geometric.data.Data nesnesidir ve şu alanları içermelidir:

        data.x          Tensor [num_nodes, 4]   – düğüm özellikleri
                            (valence, aspect_ratio, skewness, jacobian)
        data.edge_index Tensor [2, E]            – çift yönlü kenar listesi
        data.y          Tensor [seq_len]  long   – hedef token dizisi
                            (<START>=1 ile başlar, <END>=2 ile biter)
    """

    def __init__(self, pt_files_path: str) -> None:
        self.file_paths = sorted(Path(pt_files_path).glob("*.pt"))
        if not self.file_paths:
            raise FileNotFoundError(
                f"'{pt_files_path}' klasöründe hiç .pt dosyası bulunamadı."
            )
        

    def __len__(self) -> int:
        return len(self.file_paths)

    def __getitem__(self, idx: int) -> Data:
        return torch.load(self.file_paths[idx], weights_only=False)


def collate_fn(batch: list[Data], max_seq_len: int = 2048) -> Batch:
    """
    Farklı boyutlardaki grafikleri ve farklı uzunluktaki hedef dizilerini
    tek bir batch'de birleştirir.

    Grafik verileri (x, edge_index):
        Standart PyG Batch.from_data_list mekanizmasıyla birleştirilir.
        batch.batch tensörü her düğümün hangi grafiğe ait olduğunu gösterir.

    Hedef diziler (y):
        Önce max_seq_len'e truncate edilir, sonra pad_sequence ile
        PAD_TOKEN_ID=0 kullanılarak doldurulur.
        Sonuç: batch.y  Tensor [batch_size, min(T_max, max_seq_len)]

    Parameters
    ----------
    batch : list[Data]
        DataLoader'ın örneklediği Data nesneleri listesi.
    max_seq_len : int
        Dizi uzunluğu üst sınırı. Daha uzun diziler kesilir.

    Returns
    -------
    Batch
        Birleştirilmiş ve padlenmiş PyG Batch nesnesi.
    """
    # Dizileri max_seq_len'e truncate et (causal mask O(T²) bellek tüketir)
    ys = [data.y[:max_seq_len] for data in batch]

    # PyG grafik batch'i
    pyg_batch = Batch.from_data_list(batch)

    # Hedef dizileri pad: [max_seq_len, B] → transpose → [B, max_seq_len]
    padded_y = pad_sequence(ys, batch_first=True, padding_value=0)
    pyg_batch.y = padded_y  # [B, T_max]

    return pyg_batch


# =========================================================================== #
#  3. Ana Model: Graph2Seq                                                     #
# =========================================================================== #

class Graph2Seq(nn.Module):
    """
    Encoder-Decoder Graf-to-Sequence modeli.

    HourGlassGNNEncoder ile mesh grafiğini latent vektöre sıkıştırır;
    AutoregressiveMeshDecoder ile teacher-forcing eğitiminde logits üretir.

    Parameters
    ----------
    config : Config
        Tüm hiperparametreleri içeren yapılandırma nesnesi.
    """

    def __init__(self, config: Config) -> None:
        super().__init__()

        self.encoder = HourGlassGNNEncoder(
            in_channels = config.IN_CHANNELS,
            hidden_dim  = config.HIDDEN_DIM,
            num_layers  = config.ENC_NUM_LAYERS,
            num_heads   = config.ENC_NUM_HEADS,
            dropout     = config.ENC_DROPOUT,
            pooling     = config.ENC_POOLING,
        )

        self.decoder = AutoregressiveMeshDecoder(
            vocab_size      = config.VOCAB_SIZE,
            hidden_dim      = config.HIDDEN_DIM,
            num_layers      = config.DEC_NUM_LAYERS,
            nhead           = config.DEC_NHEAD,
            dim_feedforward = config.DEC_DIM_FEEDFORWARD,
            dropout         = config.DEC_DROPOUT,
            max_seq_len     = config.DEC_MAX_SEQ_LEN,
        )

    def forward(self, batch: Batch) -> torch.Tensor:
        """
        Teacher-Forcing ileri geçişi.

        Parameters
        ----------
        batch : Batch
            PyG Batch nesnesi. Beklenen alanlar:
                batch.x          [total_nodes, 4]
                batch.edge_index [2, E]
                batch.batch      [total_nodes]       – düğüm→graf indeksi
                batch.y          [B, T_max]  long    – padlenmiş hedef diziler

        Returns
        -------
        logits : Tensor  [B, T-1, vocab_size]
            batch.y[:, :-1] giriş; tahmin hedefi batch.y[:, 1:] olacaktır.
        """
        # ── 1. Encode ──────────────────────────────────────────────────────────
        # Grafik topolojisi → sabit boyutlu latent bağlam vektörü
        context = self.encoder(
            batch.x, batch.edge_index, batch.batch
        )  # [B, hidden_dim]

        # ── 2. Teacher-Forcing Girişi ──────────────────────────────────────────
        # Decoder'a verilen giriş: tüm hedef (son token hariç)
        #   batch.y[:, :-1]  →  giriş   (<START> ... son_öncesi_token)
        #   batch.y[:, 1:]   →  hedef   (ikinci_token ... <END>)
        tgt_input = batch.y[:, :-1]  # [B, T-1]

        # PAD pozisyonlarını maskele (True = bu pozisyonu görmezden gel)
        tgt_key_padding_mask = (tgt_input == 0)  # [B, T-1]

        # ── 3. Decode ──────────────────────────────────────────────────────────
        logits = self.decoder(
            tgt_tokens           = tgt_input,
            memory               = context,
            tgt_key_padding_mask = tgt_key_padding_mask,
        )  # [B, T-1, vocab_size]

        return logits


# =========================================================================== #
#  4. Kayıp Fonksiyonları                                                      #
# =========================================================================== #

def calculate_stitch_loss(
    logits: torch.Tensor,
    vocabulary: dict,
) -> float:
    """
    Geometrik Süreklilik Kaybı (L_Stitch) – PLACEHOLDER.

    Bu fonksiyon, modelin ürettiği token dizisini geometriye dönüştürür ve
    canonical (BFS) sırasında yan yana gelen yamalar arasındaki paylaşılan
    düğümlerin koordinat tutarsızlığını ölçer.

    Planlanan Implementasyon Adımları (ileride doldurulacak):
    ----------------------------------------------------------

    ADIM 1 – Token → Geometri Dönüşümü:
        predicted_tokens = logits.argmax(dim=-1)   # [B, T-1]
        vocabulary[str(token_id)]["local_nodes"] → {node_id: [x, y, z]}
        SPECIAL token'lar (id < 4) için geometri yoktur; bu pozisyonlar atlanır.

    ADIM 2 – Paylaşılan Düğümlerin Belirlenmesi:
        Canonical (BFS) sıralamasında ardışık iki yama (patch_i, patch_j)
        belirli düğümleri fiziksel olarak paylaşır.
        Örnek: patch_i'nin "3. düğümü" ile patch_j'nin "0. düğümü"
               aynı 3D koordinata karşılık gelmelidir.
        Bu eşleşme haritası (stitch map) FAZ 2 BFS sıralamasından türetilir
        veya vocabulary metadata'sındaki topoloji bilgisinden hesaplanır.

    ADIM 3 – MSE Hesabı:
        stitch_losses = []
        for (coord_a, coord_b) in shared_node_pairs:
            stitch_losses.append(F.mse_loss(coord_a, coord_b))
        loss_stitch = torch.stack(stitch_losses).mean()

    ADIM 4 – Gradyan Uyumluluğu:
        argmax differentiable değildir. Gerçek implementasyonda:
          - Gumbel-Softmax (straight-through estimator) veya
          - Soft token embedding (logits → softmax → embedding matmul)
        kullanılarak gradyanın logits'e geri akması sağlanmalıdır.

    Parameters
    ----------
    logits : Tensor  [B, T-1, vocab_size]
        Modelin ürettiği ham logit değerleri.
    vocabulary : dict
        ai_vocabulary.json içeriği (token_id → patch bilgisi eşleşmesi).

    Returns
    -------
    float
        Hesaplanan geometrik süreklilik kaybı.
        Şu an sıfır döndürülmektedir (placeholder).
    """
    # TODO: Yukarıdaki 4 adımı uygula.
    #       Adım 4 için differentiable token sampling (Gumbel-Softmax) gerekli.
    return 0.0


# =========================================================================== #
#  5. Eğitim Döngüsü: Trainer                                                  #
# =========================================================================== #

class Trainer:
    """
    Eğitim ve değerlendirme döngüsünü yöneten sınıf.

    Her epoch:
        1. train_epoch  – modeli günceller, train loss döndürür
        2. eval_epoch   – validation loss hesaplar, checkpoint kaydeder

    Parameters
    ----------
    model : Graph2Seq
    train_loader : DataLoader
    val_loader : DataLoader
    optimizer : torch.optim.Optimizer
    config : Config
    vocabulary : dict
        ai_vocabulary.json içeriği (L_Stitch hesabı için).
    """

    def __init__(
        self,
        model: Graph2Seq,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        config: Config,
        vocabulary: dict,
    ) -> None:
        self.model        = model
        self.train_loader = train_loader
        self.val_loader   = val_loader
        self.optimizer    = optimizer
        self.config       = config
        self.vocabulary   = vocabulary
        self.device       = torch.device(config.DEVICE)

        # Öğrenme Oranı Zamanlayıcısını (LR Scheduler) oluştur
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.config.EPOCHS,  # Döngünün yarısını tamamlayacağı epoch sayısı
            eta_min=1e-6               # Ulaşacağı minimum öğrenme oranı
        )

        # Ana Kayıp: CrossEntropyLoss
        #   ignore_index=PAD_TOKEN_ID → model PAD tahmin etmeye çalışmaz.
        self.criterion = nn.CrossEntropyLoss(
            ignore_index=config.PAD_TOKEN_ID,
        )

        Path(config.CHECKPOINT_PATH).mkdir(parents=True, exist_ok=True)
        self.best_val_loss: float = math.inf

    # ----------------------------------------------------------------------- #

    def _compute_loss(self, batch: Batch) -> torch.Tensor:
        """
        Tek bir batch için toplam kaybı hesaplar ve döndürür.

            L_total = L_CE + alpha * L_Stitch

        CrossEntropyLoss boyut beklentisi:
            input  [N, C]  →  logits.reshape(B * (T-1), vocab_size)
            target [N]     →  targets.reshape(B * (T-1))
        """
        batch = batch.to(self.device)

        logits  = self.model(batch)           # [B, T-1, vocab_size]
        targets = batch.y[:, 1:].long()       # [B, T-1]  –  <START> atlanır

        B, T, V = logits.shape
        loss_ce = self.criterion(
            logits.reshape(B * T, V),   # [B*(T-1), vocab_size]
            targets.reshape(B * T),     # [B*(T-1)]
        )

        # Geometrik süreklilik kaybı (şimdilik 0.0)
        loss_stitch = calculate_stitch_loss(logits.detach(), self.vocabulary)

        return loss_ce + self.config.ALPHA * loss_stitch

    # ----------------------------------------------------------------------- #

    def train_epoch(self, epoch: int) -> float:
        """
        Tek bir eğitim epoch'u.

        Modeli train() moduna alır, tüm batch'leri işler,
        gradyanları hesaplar ve ağırlıkları günceller.

        Returns
        -------
        float
            Epoch boyunca hesaplanan ortalama eğitim kaybı.
        """
        self.model.train()
        total_loss  = 0.0
        num_batches = len(self.train_loader)

        progress = tqdm(
            self.train_loader,
            desc=f"Epoch {epoch:03d} [Train]",
            leave=False,
            dynamic_ncols=True,
        )

        for batch in progress:
            self.optimizer.zero_grad()

            loss = self._compute_loss(batch)
            loss.backward()

            # Patlayan gradyanları klipler
            nn.utils.clip_grad_norm_(
                self.model.parameters(), self.config.GRAD_CLIP
            )

            self.optimizer.step()

            total_loss += loss.item()
            progress.set_postfix(loss=f"{loss.item():.4f}")

        return total_loss / num_batches

    # ----------------------------------------------------------------------- #

    def eval_epoch(self, epoch: int) -> float:
        """
        Validation seti üzerinde kayıp hesaplar.

        Modeli eval() moduna alır ve torch.no_grad() bloğunda
        validation seti üzerinde kayıpları toplar.

        Returns
        -------
        float
            Epoch boyunca hesaplanan ortalama validation kaybı.
        """
        self.model.eval()
        total_loss  = 0.0
        num_batches = len(self.val_loader)

        with torch.no_grad():
            progress = tqdm(
                self.val_loader,
                desc=f"Epoch {epoch:03d} [Val]  ",
                leave=False,
                dynamic_ncols=True,
            )
            for batch in progress:
                loss = self._compute_loss(batch)
                total_loss += loss.item()
                progress.set_postfix(loss=f"{loss.item():.4f}")

        return total_loss / num_batches

    # ----------------------------------------------------------------------- #

    def _save_checkpoint(self, epoch: int, val_loss: float) -> None:
        """En iyi modeli .pth dosyasına kaydeder."""
        checkpoint = {
            "epoch":           epoch,
            "val_loss":        val_loss,
            "model_state":     self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "config":          self.config,
        }
        ckpt_path = Path(self.config.CHECKPOINT_PATH) / "best_model.pth"
        torch.save(checkpoint, ckpt_path)
        print(
            f"  [Checkpoint] Epoch {epoch:03d} – val_loss={val_loss:.6f}"
            f" → {ckpt_path}"
        )

    # ----------------------------------------------------------------------- #

    def run(self) -> None:
        """
        Ana eğitim döngüsü.

        Config.EPOCHS kadar epoch boyunca train_epoch ve eval_epoch'u
        sırayla çalıştırır. Her epoch sonunda validation loss iyileştiyse
        checkpoint kaydeder.
        """
        print(f"\n{'=' * 62}")
        print(f"  Eğitim başlıyor  –  Cihaz : {self.device.type.upper()}")
        print(f"  Epochs : {self.config.EPOCHS}"
              f"  |  Batch : {self.config.BATCH_SIZE}"
              f"  |  LR : {self.config.LEARNING_RATE}"
              f"  |  alpha : {self.config.ALPHA}")
        print(f"{'=' * 62}\n")

        for epoch in range(1, self.config.EPOCHS + 1):
            train_loss = self.train_epoch(epoch)
            val_loss   = self.eval_epoch(epoch)

            print(
                f"Epoch {epoch:03d}/{self.config.EPOCHS:03d}"
                f"  │  train={train_loss:.6f}"
                f"  │  val={val_loss:.6f}"
            )

            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self._save_checkpoint(epoch, val_loss)

            # Öğrenme oranını scheduler'a göre güncelle
            self.scheduler.step()
            print(f"  [LR] Yeni öğrenme oranı: {self.scheduler.get_last_lr()[0]:.7f}")

        print(
            f"\n  Eğitim tamamlandı."
            f" En iyi val_loss: {self.best_val_loss:.6f}"
        )


# =========================================================================== #
#  6. Giriş Noktası                                                            #
# =========================================================================== #

if __name__ == "__main__":

    # ── 1. Yapılandırma ───────────────────────────────────────────────────────
    config = Config()
    device = torch.device(config.DEVICE)

    print(f"[Config] Cihaz      : {device}")
    print(f"[Config] PT klasörü : {config.PT_FILES_PATH}")
    print(f"[Config] Vocab      : {config.VOCAB_PATH}")

    # ── 2. Vocabulary Yükle ───────────────────────────────────────────────────
    with open(config.VOCAB_PATH, encoding="utf-8") as f:
        vocabulary: dict = json.load(f)
    print(f"[Vocab]  {len(vocabulary) - 1} giriş yüklendi (metadata dahil).")

    # ── 3. Dataset ────────────────────────────────────────────────────────────
    dataset = MeshDataset(config.PT_FILES_PATH)
    print(f"[Dataset] {len(dataset)} .pt dosyası bulundu.")

    # Eğitim / Validation bölünmesi (tekrar üretilebilir bölünme için seed)
    n_train = int(len(dataset) * config.TRAIN_RATIO)
    n_val   = len(dataset) - n_train
    train_dataset, val_dataset = random_split(
        dataset,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(42),
    )
    print(f"[Dataset] Train: {n_train}  |  Val: {n_val}")

    # ── 4. DataLoader ─────────────────────────────────────────────────────────
    # collate_fn: grafikleri PyG Batch, y dizilerini truncate+pad ile birleştirir
    _collate = partial(collate_fn, max_seq_len=config.DEC_MAX_SEQ_LEN)
    train_loader = DataLoader(
        train_dataset,
        batch_size  = config.BATCH_SIZE,
        shuffle     = True,
        collate_fn  = _collate,
        num_workers = config.NUM_WORKERS,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size  = config.BATCH_SIZE,
        shuffle     = False,
        collate_fn  = _collate,
        num_workers = config.NUM_WORKERS,
    )

    # ── 5. Model ──────────────────────────────────────────────────────────────
    model = Graph2Seq(config).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Model]  Eğitilebilir parametre: {n_params:,}")

    # ── 6. Optimizer ──────────────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr           = config.LEARNING_RATE,
        weight_decay = config.WEIGHT_DECAY,
    )

    # ── 7. Trainer ────────────────────────────────────────────────────────────
    trainer = Trainer(
        model        = model,
        train_loader = train_loader,
        val_loader   = val_loader,
        optimizer    = optimizer,
        config       = config,
        vocabulary   = vocabulary,
    )
    trainer.run()
