"""
models/decoder.py  –  Autoregressive Transformer Decoder
=========================================================
GNN Encoder'dan gelen Latent Context vektörünü ve o ana kadar üretilen
token dizisini alarak bir sonraki token'ın olasılık dağılımını (logits)
tahmin eder.

Mimari:
    ┌──────────────────────────────────────────────────────┐
    │  nn.Embedding (vocab_size=64, hidden_dim)            │
    ├──────────────────────────────────────────────────────┤
    │  PositionalEncoding  (sinusoidal, seq × batch × dim) │
    ├──────────────────────────────────────────────────────┤
    │  nn.TransformerDecoder × num_layers                  │
    │   ├─ Masked Self-Attention  (Causal Mask ✓)          │
    │   └─ Cross-Attention  ←  memory [1, B, hidden_dim]  │
    ├──────────────────────────────────────────────────────┤
    │  nn.Linear(hidden_dim → vocab_size)                  │
    └──────────────────────────────────────────────────────┘
    Çıktı: logits  [batch_size, seq_len, vocab_size]

Autoregressive Çalışma Prensibi:
    - Eğitim: teacher-forcing – tgt_tokens'ın tümü tek seferde verilir,
      causal mask sayesinde her pozisyon yalnızca önceki token'ları görür.
    - Çıkarım (inference): token-by-token üretim, her adımda en yüksek
      logit'e sahip token bir sonraki adıma eklenir.

FAZ 2 / Vocabulary ile uyum:
    PAD   = 0   (padding_idx → embedding sıfıra sabitlenmez, logit maskelenir)
    START = 1
    END   = 2
    UNK   = 3
    vocab_size = 64  (GPU bellek hizalaması için 33 token → 64 yukarı yuvarlandı)
"""

import math
import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional


# =========================================================================== #
#  Yardımcı: Sinüzoidal Pozisyonel Kodlama                                     #
# =========================================================================== #

class PositionalEncoding(nn.Module):
    """
    "Attention Is All You Need" makalesinden sinüzoidal konumsal kodlama.

    Gömme (embedding) tensörüne pozisyon bilgisini ekler; böylece Transformer
    token sırasından haberdar olur.

    Parameters
    ----------
    d_model : int
        Gömme boyutu (hidden_dim ile eşleşmeli).
    dropout : float
        Kodlama eklendikten sonra uygulanan dropout oranı.
    max_len : int
        Desteklenen maksimum dizi uzunluğu.
    """

    def __init__(
        self,
        d_model: int,
        dropout: float = 0.1,
        max_len: int = 2048,
    ) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # pe: [max_len, 1, d_model]  (broadcast için 1 batch boyutu)
        position  = torch.arange(max_len).unsqueeze(1)                  # [L, 1]
        div_term  = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10_000.0) / d_model)
        )                                                                 # [d_model/2]

        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)                                   # eğitilmez ama state_dict'e girer

    def forward(self, x: Tensor) -> Tensor:
        """
        Parameters
        ----------
        x : Tensor  [seq_len, batch_size, d_model]

        Returns
        -------
        Tensor  [seq_len, batch_size, d_model]
        """
        seq_len = x.size(0)
        if seq_len > self.pe.size(0):
            # PE buffer'ı dinamik olarak uzat (gerçek dizi max_len'i aşarsa)
            d_model  = self.pe.size(2)
            position = torch.arange(seq_len).unsqueeze(1)
            div_term = torch.exp(
                torch.arange(0, d_model, 2) * (-math.log(10_000.0) / d_model)
            )
            pe = torch.zeros(seq_len, 1, d_model)
            pe[:, 0, 0::2] = torch.sin(position * div_term)
            pe[:, 0, 1::2] = torch.cos(position * div_term)
            self.register_buffer("pe", pe.to(x.device))
        x = x + self.pe[:seq_len]
        return self.dropout(x)


# =========================================================================== #
#  AutoregressiveMeshDecoder                                                    #
# =========================================================================== #

class AutoregressiveMeshDecoder(nn.Module):
    """
    Autoregressive Transformer Decoder: 3D mesh token dizisi üretir.

    GNN Encoder'ın ürettiği latent bağlam vektörüne çapraz-dikkat
    (cross-attention) uygulayarak her pozisyon için bir sonraki token'ın
    log-olasılık dağılımını tahmin eder.

    Parameters
    ----------
    vocab_size : int
        Token sözlüğü boyutu. GPU hizalaması için 64 (gerçek vocab: 33).
    hidden_dim : int
        Gömme, dikkat ve ileri besleme katmanlarının boyutu.
    num_layers : int
        TransformerDecoderLayer sayısı.
    nhead : int
        Multi-head attention başı sayısı. hidden_dim % nhead == 0 olmalı.
    dim_feedforward : int
        Her TransformerDecoderLayer içindeki FFN boyutu.
    dropout : float
        Attention, FFN ve embedding dropout oranı.
    max_seq_len : int
        Desteklenen maksimum dizi uzunluğu (PositionalEncoding için).

    Note
    ----
    memory (encoder çıktısı) [batch_size, hidden_dim] formatında beklenir.
    Decoder içinde [1, batch_size, hidden_dim]'e dönüştürülür; böylece
    cross-attention her token'ı tüm latent bağlamla ilişkilendirebilir.
    """

    def __init__(
        self,
        vocab_size: int     = 64,
        hidden_dim: int     = 256,
        num_layers: int     = 4,
        nhead: int          = 8,
        dim_feedforward: int = 1024,
        dropout: float      = 0.1,
        max_seq_len: int    = 2048,
    ) -> None:
        super().__init__()

        assert hidden_dim % nhead == 0, (
            f"hidden_dim ({hidden_dim}) nhead ({nhead}) ile tam bölünebilmeli."
        )

        self.hidden_dim  = hidden_dim
        self.vocab_size  = vocab_size

        # ── 1. Token Gömme Katmanı ──────────────────────────────────────
        # padding_idx=0 → PAD token'ının gradyanı sıfırlanır
        self.embedding = nn.Embedding(
            num_embeddings = vocab_size,
            embedding_dim  = hidden_dim,
            padding_idx    = 0,
        )

        # ── 2. Pozisyonel Kodlama ───────────────────────────────────────
        self.pos_encoding = PositionalEncoding(
            d_model = hidden_dim,
            dropout = dropout,
            max_len = max_seq_len,
        )

        # ── 3. Transformer Decoder ──────────────────────────────────────
        decoder_layer = nn.TransformerDecoderLayer(
            d_model         = hidden_dim,
            nhead           = nhead,
            dim_feedforward = dim_feedforward,
            dropout         = dropout,
            batch_first     = False,    # [seq, batch, feat] convention
            norm_first      = True,     # Pre-LN: daha stabil eğitim
        )
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer = decoder_layer,
            num_layers    = num_layers,
            norm          = nn.LayerNorm(hidden_dim),
        )

        # ── 4. Memory Projeksiyon (encoder → decoder boyut hizalama) ────
        # Encoder çıktısı zaten hidden_dim; bu katman gerekirse farklı
        # encoder boyutlarını desteklemek için bir ara kapı sağlar.
        self.memory_proj = nn.Linear(hidden_dim, hidden_dim)

        # ── 5. Çıktı Projeksiyonu ───────────────────────────────────────
        self.output_proj = nn.Linear(hidden_dim, vocab_size)

        # Ağırlık bağlama (weight tying): çıktı projeksiyonu embedding
        # ağırlıklarını paylaşır → daha az parametre, genellikle daha iyi genelleştirme.
        self.output_proj.weight = self.embedding.weight

        # Kaynon başlatma
        self._init_weights()

    # ----------------------------------------------------------------------- #

    def _init_weights(self) -> None:
        """Xavier uniform başlatma (Transformer için standart)."""
        for name, param in self.named_parameters():
            if param.dim() > 1:
                nn.init.xavier_uniform_(param)

    # ----------------------------------------------------------------------- #

    @staticmethod
    def _make_causal_mask(seq_len: int, device: torch.device) -> Tensor:
        """
        Kausal (subsequent) maske üretir.

        Üst üçgen '-inf', diyagonal ve alt üçgen '0.0' olan [seq_len, seq_len]
        tensörü döndürür. TransformerDecoder bunu additive mask olarak kullanır:
        maskeli pozisyonlar softmax'tan önce -inf'e gönderilir → efektif sıfır dikkat.

        Parameters
        ----------
        seq_len : int
        device  : torch.device

        Returns
        -------
        Tensor  [seq_len, seq_len]  dtype=float
        """
        return torch.triu(
            torch.full((seq_len, seq_len), float("-inf"), device=device),
            diagonal=1,
        )

    # ----------------------------------------------------------------------- #

    def forward(
        self,
        tgt_tokens: Tensor,
        memory: Tensor,
        tgt_key_padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Decoder ileri geçişi.

        Parameters
        ----------
        tgt_tokens : Tensor  [batch_size, seq_len]
            Şu ana kadar üretilen (veya teacher-forced) token ID dizisi.
            <START>=1 ile başlamalı.
        memory : Tensor  [batch_size, hidden_dim]
            GNN Encoder'dan gelen latent bağlam vektörü.
        tgt_key_padding_mask : Tensor  [batch_size, seq_len] | None
            PAD token'larını maskelemek için boolean tensör (True = gz ardı et).
            Eğitimde batch padding varsa sağlanmalıdır.

        Returns
        -------
        Tensor  [batch_size, seq_len, vocab_size]
            Her pozisyon için ham logit değerleri.
            Kayıp hesabı için CrossEntropyLoss'a doğrudan verilebilir.
        """
        batch_size, seq_len = tgt_tokens.shape
        device = tgt_tokens.device

        # 1. Token Gömme + Pozisyonel Kodlama
        #    [batch, seq] → [batch, seq, hidden] → [seq, batch, hidden]
        tgt: Tensor = self.embedding(tgt_tokens)          # [B, T, H]
        tgt = tgt.transpose(0, 1)                          # [T, B, H]  (PyTorch convention)
        tgt = self.pos_encoding(tgt)                       # [T, B, H]

        # 2. Memory dönüşümü: [B, H] → [1, B, H]
        #    Tek bir "bağlam token'ı" olarak çapraz-dikkate sunulur.
        mem: Tensor = self.memory_proj(memory)             # [B, H]
        mem = mem.unsqueeze(0)                             # [1, B, H]

        # 3. Kausal maske oluştur [T, T]
        causal_mask: Tensor = self._make_causal_mask(seq_len, device)

        # 4. Transformer Decoder
        out: Tensor = self.transformer_decoder(
            tgt                  = tgt,             # [T, B, H]
            memory               = mem,             # [1, B, H]
            tgt_mask             = causal_mask,     # [T, T]  kausal maske
            tgt_key_padding_mask = tgt_key_padding_mask,   # [B, T] | None
        )                                           # → [T, B, H]

        # 5. Çıktı projeksiyonu
        out = out.transpose(0, 1)                   # [B, T, H]
        logits: Tensor = self.output_proj(out)      # [B, T, vocab_size]

        return logits

    # ----------------------------------------------------------------------- #

    @torch.no_grad()
    def greedy_decode(
        self,
        memory: Tensor,
        start_token: int = 1,
        end_token: int   = 2,
        max_len: int     = 512,
    ) -> Tensor:
        """
        Açgözlü (greedy) autoregressive çıkarım.

        Her adımda en yüksek logit'e sahip token seçilir ve dizi END token'ına
        ulaştığında veya max_len'e geldiğinde durur.

        Parameters
        ----------
        memory : Tensor  [batch_size, hidden_dim]
            Encoder latent bağlam vektörü.
        start_token : int
            Diziye başlangıç token'ı (START=1).
        end_token : int
            Durdurma koşulu token'ı (END=2).
        max_len : int
            Üretilecek maksimum token sayısı.

        Returns
        -------
        Tensor  [batch_size, seq_len]
            Üretilen token ID dizisi (START dahil, END dahil ya da max_len'de kesilmiş).
        """
        batch_size = memory.size(0)
        device     = memory.device

        # [B, 1] – her dizi START ile başlar
        generated = torch.full(
            (batch_size, 1), start_token, dtype=torch.long, device=device
        )
        ended = torch.zeros(batch_size, dtype=torch.bool, device=device)

        for _ in range(max_len - 1):
            logits = self.forward(generated, memory)   # [B, cur_len, V]
            next_token = logits[:, -1, :].argmax(dim=-1)  # [B]

            generated = torch.cat(
                [generated, next_token.unsqueeze(1)], dim=1
            )                                            # [B, cur_len+1]

            ended |= (next_token == end_token)
            if ended.all():
                break

        return generated


# =========================================================================== #
#  Standalone Test                                                              #
# =========================================================================== #

if __name__ == "__main__":
    """
    Rastgele tensörlerle forward pass ve greedy decode'u doğrular.

    Çalıştırma:
        python -m models.decoder
        # veya proje kökünden:
        python models/decoder.py
    """
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

    print("=" * 60)
    print("  AutoregressiveMeshDecoder – Forward Pass Testi")
    print("=" * 60)

    torch.manual_seed(42)

    BATCH_SIZE   = 4
    SEQ_LEN      = 20     # teacher-forced dizi uzunluğu
    HIDDEN_DIM   = 256
    VOCAB_SIZE   = 64

    # ── Dummy girdiler ──────────────────────────────────────────────────
    # Gerçek kullanımda memory encoder.forward() çıktısıdır
    memory     = torch.randn(BATCH_SIZE, HIDDEN_DIM)
    tgt_tokens = torch.randint(1, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN))

    # Bazı pozisyonları PAD yap (padding maskesi testi için)
    tgt_tokens[:, -3:] = 0
    pad_mask = (tgt_tokens == 0)  # [B, T]  True = padding

    print(f"\n[Girdi]")
    print(f"  memory.shape    : {list(memory.shape)}")
    print(f"  tgt_tokens.shape: {list(tgt_tokens.shape)}")
    print(f"  pad_mask.shape  : {list(pad_mask.shape)}")

    # ── Model ───────────────────────────────────────────────────────────
    decoder = AutoregressiveMeshDecoder(
        vocab_size      = VOCAB_SIZE,
        hidden_dim      = HIDDEN_DIM,
        num_layers      = 4,
        nhead           = 8,
        dim_feedforward = 1024,
        dropout         = 0.1,
    )
    decoder.eval()

    # ── Teacher-forced Forward Pass ─────────────────────────────────────
    with torch.no_grad():
        logits = decoder(tgt_tokens, memory, tgt_key_padding_mask=pad_mask)

    print(f"\n[Çıktı – Teacher-Forced Forward]")
    print(f"  logits.shape    : {list(logits.shape)}")   # [4, 20, 64]
    assert logits.shape == (BATCH_SIZE, SEQ_LEN, VOCAB_SIZE), (
        f"Beklenmeyen boyut: {logits.shape}"
    )
    print("  [OK] Boyut kontrolü geçti.")

    # ── Causal Mask Doğrulama ────────────────────────────────────────────
    mask = AutoregressiveMeshDecoder._make_causal_mask(5, device=torch.device("cpu"))
    print(f"\n[Causal Mask 5×5]")
    print(mask)
    assert mask[0, 1] == float("-inf"), "Kausal maske yanlış!"
    assert mask[1, 0] == 0.0,          "Kausal maske yanlış (alt üçgen)!"
    print("  [OK] Kausal maske doğru.")

    # ── Greedy Decode ────────────────────────────────────────────────────
    with torch.no_grad():
        generated = decoder.greedy_decode(memory, max_len=30)

    print(f"\n[Greedy Decode]")
    print(f"  generated.shape : {list(generated.shape)}")
    print(f"  İlk örnek tokens: {generated[0].tolist()}")
    assert generated.shape[0] == BATCH_SIZE
    assert generated.shape[1] <= 30
    print("  [OK] Greedy decode boyut kontrolü geçti.")

    # ── Weight Tying Kontrolü ────────────────────────────────────────────
    assert decoder.output_proj.weight.data_ptr() == decoder.embedding.weight.data_ptr(), \
        "Weight tying çalışmıyor!"
    print("\n  [OK] Weight tying doğrulandı.")

    # ── Parametre Sayısı ─────────────────────────────────────────────────
    n_params = sum(p.numel() for p in decoder.parameters() if p.requires_grad)
    print(f"\n  Eğitilebilir parametre: {n_params:,}")
    print("=" * 60)
