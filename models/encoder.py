"""
models/encoder.py  –  HourGlass GNN Encoder
============================================
RAW mesh grafiğini (düğüm özellikleri + topoloji) alır ve
Transformer Decoder'ın çapraz-dikkat (cross-attention) uygulayabileceği
sabit boyutlu bir Latent Context vektörüne sıkıştırır.

Mimari (Darboğaz / HourGlass):
    ┌──────────────────────────────────────────┐
    │  Linear(in_channels -> hidden_dim)       │  Giriş projeksiyonu
    ├──────────────────────────────────────────┤
    │  GATConv  × num_layers                   │  Mesaj iletim katmanları
    │  (her katman: BatchNorm + ReLU + Skip)   │  (daralan bağlam)
    ├──────────────────────────────────────────┤
    │  global_mean_pool                        │  Graf -> tek vektör
    ├──────────────────────────────────────────┤
    │  MLP (Linear → ReLU → Dropout → Linear)  │  Latent projeksiyon
    └──────────────────────────────────────────┘
    Çıktı: [batch_size, hidden_dim]

FAZ 2 veri formatıyla uyum:
    data.x          [num_nodes, 4]    – valence, AR, skewness, jacobian
    data.edge_index [2, E]            – çift yönlü komşuluk
    batch.batch     [num_nodes]       – hangi düğüm hangi grafikte
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional

from torch_geometric.nn import GATConv, global_mean_pool, global_max_pool


# =========================================================================== #
#  HourGlassGNNEncoder                                                         #
# =========================================================================== #

class HourGlassGNNEncoder(nn.Module):
    """
    Graf dikkat ağı tabanlı HourGlass encoder.

    Düğüm düzeyindeki özellikleri birden fazla GATConv katmanı ile işler,
    ardından global havuzlama (pooling) ile tüm grafı tek bir latent vektöre
    sıkıştırır. Bu vektör Transformer Decoder'a "memory" olarak aktarılır.

    Parameters
    ----------
    in_channels : int
        data.x boyutu (FAZ2'de sabit = 4: valence, AR, skewness, jacobian).
    hidden_dim : int
        Tüm iç katmanlarda ve çıktıda kullanılan gizli boyut.
    num_layers : int
        GATConv mesaj iletim katmanı sayısı.
    num_heads : int
        Her GATConv katmanındaki dikkat başı (head) sayısı.
        (concat=False ile başlar ortalaması alınır → boyut sabit kalır.)
    dropout : float
        Dropout oranı (hem bağlantı hem dikkat için).
    pooling : str
        Global havuzlama stratejisi: "mean" | "max" | "mean+max".
        "mean+max" seçilirse iki vektör birleştirilir ve hidden_dim'e
        yeniden projekte edilir.
    """

    def __init__(
        self,
        in_channels: int = 4,
        hidden_dim: int = 256,
        num_layers: int = 4,
        num_heads: int = 4,
        dropout: float = 0.1,
        pooling: str = "mean",
    ) -> None:
        super().__init__()

        assert pooling in ("mean", "max", "mean+max"), (
            f"pooling '{pooling}' geçersiz. 'mean', 'max' veya 'mean+max' olmalıdır."
        )
        assert hidden_dim % num_heads == 0, (
            f"hidden_dim ({hidden_dim}) num_heads ({num_heads}) ile tam bölünebilmeli."
        )

        self.hidden_dim = hidden_dim
        self.dropout_p  = dropout
        self.pooling    = pooling

        # ── 1. Giriş projeksiyonu ──────────────────────────────────────────
        self.input_proj = nn.Linear(in_channels, hidden_dim)

        # ── 2. GATConv katmanları + BatchNorm ─────────────────────────────
        # concat=False: her başın çıktısı ortalanır → boyut hidden_dim kalır.
        self.convs = nn.ModuleList([
            GATConv(
                in_channels  = hidden_dim,
                out_channels = hidden_dim,
                heads        = num_heads,
                concat       = False,
                dropout      = dropout,
                add_self_loops = True,
            )
            for _ in range(num_layers)
        ])
        self.norms = nn.ModuleList([
            nn.BatchNorm1d(hidden_dim) for _ in range(num_layers)
        ])

        # ── 3. Çıktı MLP ──────────────────────────────────────────────────
        # "mean+max" ise havuzlama sonrası 2*hidden_dim → hidden_dim
        mlp_in = hidden_dim * 2 if pooling == "mean+max" else hidden_dim
        self.output_mlp = nn.Sequential(
            nn.Linear(mlp_in, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )

    # ----------------------------------------------------------------------- #

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        batch: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Encoder ileri geçişi.

        Parameters
        ----------
        x : Tensor  [total_nodes, in_channels]
            Düğüm özellik matrisi (tüm batch'teki düğümler birleşik).
        edge_index : Tensor  [2, E]
            Kenar listesi (PyG convention).
        batch : Tensor  [total_nodes] | None
            Her düğümün ait olduğu graf indeksi.
            None verilirse tek bir graf (batch_size=1) kabul edilir.

        Returns
        -------
        Tensor  [batch_size, hidden_dim]
            Her graftaki bilgiyi özetleyen latent bağlam vektörü.
        """
        if batch is None:
            batch = x.new_zeros(x.size(0), dtype=torch.long)

        # 1. Giriş projeksiyonu
        h: Tensor = F.relu(self.input_proj(x))  # [N, hidden_dim]

        # 2. GATConv mesaj iletimi (artık bağlantılı)
        for conv, norm in zip(self.convs, self.norms):
            h_new: Tensor = conv(h, edge_index)   # [N, hidden_dim]
            h_new = norm(h_new)
            h_new = F.relu(h_new)
            h_new = F.dropout(h_new, p=self.dropout_p, training=self.training)
            h = h + h_new                          # skip bağlantısı

        # 3. Global havuzlama → [batch_size, hidden_dim]
        if self.pooling == "mean":
            h_global = global_mean_pool(h, batch)
        elif self.pooling == "max":
            h_global = global_max_pool(h, batch)
        else:  # "mean+max"
            h_mean = global_mean_pool(h, batch)
            h_max  = global_max_pool(h, batch)
            h_global = torch.cat([h_mean, h_max], dim=-1)  # [B, 2*hidden_dim]

        # 4. Latent projeksiyon → [batch_size, hidden_dim]
        latent: Tensor = self.output_mlp(h_global)
        return latent


# =========================================================================== #
#  Standalone Test                                                              #
# =========================================================================== #

if __name__ == "__main__":
    """
    Rastgele PyG Data nesneleriyle forward pass'i doğrular.
    Boyut uyuşmazlığı varsa anında hata verir.

    Çalıştırma:
        python -m models.encoder
        # veya proje kökünden:
        python models/encoder.py
    """
    import sys
    from pathlib import Path

    # Proje kökünü path'e ekle (doğrudan çalıştırma için)
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

    from torch_geometric.data import Data, Batch

    print("=" * 60)
    print("  HourGlassGNNEncoder – Forward Pass Testi")
    print("=" * 60)

    torch.manual_seed(42)

    # ── Dummy batch oluştur: farklı boyutlu 3 graf ──────────────────────
    def _ring_edges(n: int) -> Tensor:
        """n düğümlü çift yönlü halka grafı kenar listesi."""
        src = torch.arange(n)
        dst = (src + 1) % n
        return torch.stack([
            torch.cat([src, dst]),
            torch.cat([dst, src]),
        ], dim=0)

    graphs = [
        Data(x=torch.randn(10, 4), edge_index=_ring_edges(10)),
        Data(x=torch.randn(18, 4), edge_index=_ring_edges(18)),
        Data(x=torch.randn(14, 4), edge_index=_ring_edges(14)),
    ]
    batch = Batch.from_data_list(graphs)

    print(f"\n[Girdi]")
    print(f"  batch.x.shape         : {list(batch.x.shape)}")       # [42, 4]
    print(f"  batch.edge_index.shape: {list(batch.edge_index.shape)}")
    print(f"  batch.batch.shape     : {list(batch.batch.shape)}")

    # ── Varsayılan hiperparametreler ─────────────────────────────────────
    encoder = HourGlassGNNEncoder(
        in_channels = 4,
        hidden_dim  = 256,
        num_layers  = 4,
        num_heads   = 4,
        dropout     = 0.1,
        pooling     = "mean",
    )
    encoder.eval()

    with torch.no_grad():
        latent = encoder(batch.x, batch.edge_index, batch.batch)

    print(f"\n[Çıktı]")
    print(f"  latent.shape          : {list(latent.shape)}")   # [3, 256]
    assert latent.shape == (3, 256), f"Beklenmeyen boyut: {latent.shape}"
    print("\n  [OK] Boyut kontrolü geçti.")

    # ── mean+max pooling testi ────────────────────────────────────────────
    encoder_mm = HourGlassGNNEncoder(
        in_channels = 4,
        hidden_dim  = 256,
        num_layers  = 2,
        num_heads   = 4,
        pooling     = "mean+max",
    )
    encoder_mm.eval()
    with torch.no_grad():
        latent_mm = encoder_mm(batch.x, batch.edge_index, batch.batch)
    print(f"  latent (mean+max).shape: {list(latent_mm.shape)}")
    assert latent_mm.shape == (3, 256)
    print("  [OK] mean+max pooling boyut kontrolü geçti.")

    # ── Parametre sayısı ─────────────────────────────────────────────────
    n_params = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
    print(f"\n  Eğitilebilir parametre: {n_params:,}")
    print("=" * 60)
