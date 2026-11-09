"""
codebook_builder.py — WL-Hash Frekans Analizi ve Codebook (Kelime Dağarcığı) Oluşturma.

3D mesh yamalarının Weisfeiler-Lehman hash dağılımını analiz eder,
Pareto grafiği çizer ve "Elit" yamaları seçerek nihai Codebook'u üretir.

Kullanım:
    from utils.codebook_builder import CodebookBuilder

    builder = CodebookBuilder()
    builder.load_topology_groups(groups)          # Dict[str, List[int]]
    builder.load_from_json("patches.json")        # veya JSON dosyasından
    df = builder.build_frequency_table()
    builder.plot_pareto(df)
    codebook = builder.build_codebook(df, coverage_threshold=0.95)
"""

from __future__ import annotations

import json
import math


class _NumpyEncoder(json.JSONEncoder):
    """NumPy tiplerini native Python tiplerine dönüştüren JSON encoder."""

    def default(self, o):
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return super().default(o)
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("TkAgg")  # GUI penceresi açabilen interactive renderer
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd


# ─────────────────────────────────────────────────────────────────────
# Veri Yapıları
# ─────────────────────────────────────────────────────────────────────

@dataclass
class CodebookEntry:
    """Codebook'taki tek bir yama türünü temsil eder."""
    rank: int
    wl_hash: str
    frequency: int
    percentage: float
    cumulative_percentage: float
    center_node_ids: List[int] = field(default_factory=list)


@dataclass
class CodebookResult:
    """build_codebook() metodunun döndürdüğü yapı."""
    codebook: pd.DataFrame
    outliers: pd.DataFrame
    cut_method: str          # "top_k" veya "coverage"
    cut_value: float         # k değeri veya threshold
    total_patches: int
    covered_patches: int
    coverage_ratio: float
    unique_in_codebook: int
    unique_outliers: int


# ─────────────────────────────────────────────────────────────────────
# Ana Sınıf
# ─────────────────────────────────────────────────────────────────────

class CodebookBuilder:
    """WL-Hash frekans dağılımı analizi ve Codebook oluşturucu."""

    def __init__(self) -> None:
        self._hash_counter: Counter[str] = Counter()
        self._hash_node_ids: Dict[str, List[int]] = {}
        self._source_files: List[str] = []

    # ─────────────────────────────────────────────────────────────────
    # ADIM 0 — Veri Yükleme
    # ─────────────────────────────────────────────────────────────────

    def load_topology_groups(self, groups: Dict[str, List[int]]) -> None:
        """Dict[str, List[int]] formatındaki topoloji gruplarını yükler.

        Args:
            groups: Anahtar → WL-Hash, Değer → Bu hash'e sahip node ID listesi.
        """
        for wl_hash, node_ids in groups.items():
            self._hash_counter[wl_hash] += len(node_ids)
            self._hash_node_ids.setdefault(wl_hash, []).extend(node_ids)

    def load_from_json(self, json_path: str | Path) -> None:
        """main.py'nin ürettiği *_patches.json dosyasından topology_groups yükler.

        Args:
            json_path: JSON dosyasının yolu.
        """
        json_path = Path(json_path)
        with json_path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)

        self._source_files.append(data.get("file_name", json_path.name))

        topo_groups: Dict[str, Any] = data.get("topology_groups", {})
        for wl_hash, info in topo_groups.items():
            count = info["count"]
            node_ids = info.get("center_node_ids", [])
            self._hash_counter[wl_hash] += count
            self._hash_node_ids.setdefault(wl_hash, []).extend(node_ids)

    def load_from_directory(self, directory: str | Path, pattern: str = "*_patches.json") -> int:
        """Bir dizindeki tüm JSON dosyalarını toplu yükler.

        Args:
            directory: JSON dosyalarının bulunduğu dizin.
            pattern: Glob deseni.

        Returns:
            Yüklenen dosya sayısı.
        """
        directory = Path(directory)
        json_files = sorted(directory.glob(pattern))
        for jf in json_files:
            self.load_from_json(jf)
        return len(json_files)

    @property
    def total_patches(self) -> int:
        """Yüklenmiş toplam yama sayısı."""
        return sum(self._hash_counter.values())

    @property
    def unique_hashes(self) -> int:
        """Benzersiz WL-Hash sayısı."""
        return len(self._hash_counter)

    # ─────────────────────────────────────────────────────────────────
    # ADIM 1 & 2 — Frekans Tablosu ve Kümülatif Dağılım
    # ─────────────────────────────────────────────────────────────────

    def build_frequency_table(self) -> pd.DataFrame:
        """Frekansları büyükten küçüğe sıralar ve kümülatif dağılım hesaplar.

        Returns:
            Sıralı frekans tablosu DataFrame'i:
                rank, wl_hash, frequency, percentage, cumulative_frequency,
                cumulative_percentage
        """
        if not self._hash_counter:
            raise ValueError("Veri yüklenmemiş. Önce load_* metodlarından birini çağırın.")

        total = self.total_patches

        # Büyükten küçüğe sıralı (hash, count) çiftleri
        sorted_items = self._hash_counter.most_common()

        rows: List[Dict[str, Any]] = []
        cum_freq = 0

        for rank, (wl_hash, freq) in enumerate(sorted_items, start=1):
            cum_freq += freq
            pct = freq / total * 100
            cum_pct = cum_freq / total * 100
            rows.append({
                "rank": rank,
                "wl_hash": wl_hash,
                "frequency": freq,
                "percentage": round(pct, 4),
                "cumulative_frequency": cum_freq,
                "cumulative_percentage": round(cum_pct, 4),
            })

        df = pd.DataFrame(rows)
        return df

    # ─────────────────────────────────────────────────────────────────
    # ADIM 3 — Pareto Grafiği / Görselleştirme
    # ─────────────────────────────────────────────────────────────────

    def plot_pareto(
        self,
        df: pd.DataFrame,
        *,
        max_bars: int = 80,
        title: str = "WL-Hash Frekans Dağılımı — Pareto Grafiği",
        figsize: Tuple[float, float] = (16, 7),
        save_path: Optional[str | Path] = None,
        show: bool = True,
        coverage_line: Optional[float] = None,
    ) -> plt.Figure:
        """Pareto grafiği çizer (bar + kümülatif çizgi).

        Args:
            df: build_frequency_table() çıktısı.
            max_bars: X ekseninde gösterilecek maksimum çubuk sayısı.
            title: Grafik başlığı.
            figsize: Figür boyutu.
            save_path: Kaydedilecek dosya yolu (None ise kaydetmez).
            show: plt.show() çağrılsın mı.
            coverage_line: Yatay kesikli çizgi çizilecek kümülatif yüzde
                           (Örn: 95.0 → %95 çizgisi).

        Returns:
            Oluşturulan matplotlib Figure nesnesi.
        """
        plot_df = df.head(max_bars).copy()
        n = len(plot_df)
        x = np.arange(n)

        fig, ax1 = plt.subplots(figsize=figsize)

        # ── Sol Y ekseni: Frekans çubukları ──
        bars = ax1.bar(
            x,
            plot_df["frequency"].values,
            color="#4C72B0",
            alpha=0.85,
            edgecolor="white",
            linewidth=0.5,
            label="Frekans",
        )
        ax1.set_xlabel("Yama Sırası (Rank)", fontsize=12, fontweight="bold")
        ax1.set_ylabel("Frekans (Adet)", fontsize=12, fontweight="bold", color="#4C72B0")
        ax1.tick_params(axis="y", labelcolor="#4C72B0")

        # X ekseni etiketleri: çok fazlaysa sadece belli aralıklarla göster
        step = max(1, n // 20)
        ax1.set_xticks(x[::step])
        ax1.set_xticklabels(plot_df["rank"].values[::step], fontsize=9)

        # ── Sağ Y ekseni: Kümülatif yüzde çizgisi ──
        ax2 = ax1.twinx()
        ax2.plot(
            x,
            plot_df["cumulative_percentage"].values,
            color="#DD8452",
            linewidth=2.5,
            marker="o",
            markersize=3,
            label="Kümülatif %",
        )
        ax2.set_ylabel(
            "Kümülatif Kapsam (%)", fontsize=12, fontweight="bold", color="#DD8452"
        )
        ax2.tick_params(axis="y", labelcolor="#DD8452")
        ax2.set_ylim(0, 105)
        ax2.yaxis.set_major_formatter(mticker.PercentFormatter())

        # ── Opsiyonel kapsam çizgisi (ör. %95) ──
        if coverage_line is not None:
            ax2.axhline(
                y=coverage_line,
                color="#C44E52",
                linestyle="--",
                linewidth=1.5,
                alpha=0.8,
            )
            # Dirsek noktası: kümülatif yüzdenin coverage_line'ı ilk aştığı rank
            elbow_mask = plot_df["cumulative_percentage"] >= coverage_line
            if elbow_mask.any():
                elbow_idx = elbow_mask.idxmax()
                elbow_rank = plot_df.loc[elbow_idx, "rank"]
                elbow_cum = plot_df.loc[elbow_idx, "cumulative_percentage"]
                ax2.annotate(
                    f"  Dirsek: Rank {elbow_rank}\n  Kapsam: %{elbow_cum:.1f}",
                    xy=(elbow_idx, elbow_cum),
                    xytext=(elbow_idx + n * 0.05, elbow_cum - 8),
                    fontsize=10,
                    fontweight="bold",
                    color="#C44E52",
                    arrowprops=dict(arrowstyle="->", color="#C44E52", lw=1.5),
                )
                ax1.axvline(
                    x=elbow_idx,
                    color="#C44E52",
                    linestyle=":",
                    linewidth=1,
                    alpha=0.6,
                )

        # ── Başlık ve düzen ──
        ax1.set_title(title, fontsize=14, fontweight="bold", pad=15)

        # Özet bilgi kutusu
        total = df["frequency"].sum()
        unique = len(df)
        info_text = (
            f"Toplam Yama: {total:,}\n"
            f"Benzersiz Hash: {unique:,}\n"
            f"İlk {n} hash gösteriliyor"
        )
        ax1.text(
            0.98, 0.55, info_text,
            transform=ax1.transAxes,
            fontsize=10,
            verticalalignment="top",
            horizontalalignment="right",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="lightyellow", alpha=0.9),
        )

        # Lejant
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="center right", fontsize=10)

        fig.tight_layout()

        if save_path:
            fig.savefig(str(save_path), dpi=150, bbox_inches="tight")

        if show:
            plt.show()

        return fig

    # ─────────────────────────────────────────────────────────────────
    # ADIM 4 — Codebook Kesimi (Truncation)
    # ─────────────────────────────────────────────────────────────────

    def build_codebook(
        self,
        df: pd.DataFrame,
        *,
        top_k: Optional[int] = None,
        coverage_threshold: Optional[float] = None,
    ) -> CodebookResult:
        """Frekans tablosundan nihai Codebook'u oluşturur.

        İki durdurma kriterinden **tam olarak biri** verilmelidir.

        Args:
            df: build_frequency_table() çıktısı.
            top_k: En popüler N adet yamayı seç (ör: 64, 128).
            coverage_threshold: Toplam kapsamın bu oranına ulaşana kadar yama al
                                (ör: 0.95 → %95).

        Returns:
            CodebookResult dataclass'ı:
                - codebook: Seçilen elit yamalar DataFrame'i
                - outliers: Dışarıda kalan nadir yamalar DataFrame'i
                - cut_method, cut_value, total_patches, covered_patches,
                  coverage_ratio, unique_in_codebook, unique_outliers

        Raises:
            ValueError: top_k ve coverage_threshold aynı anda verilir veya ikisi de
                        verilmezse.
        """
        if (top_k is None) == (coverage_threshold is None):
            raise ValueError(
                "top_k veya coverage_threshold parametrelerinden tam olarak birini verin."
            )

        total = df["frequency"].sum()

        if top_k is not None:
            # ── top_k modu ──
            cut_idx = min(top_k, len(df))
            method = "top_k"
            value = float(top_k)
        else:
            # ── coverage modu ──
            target_pct = coverage_threshold * 100  # type: ignore[operator]
            mask = df["cumulative_percentage"] >= target_pct
            if mask.any():
                cut_idx = int(mask.idxmax()) + 1  # dahil et
            else:
                cut_idx = len(df)  # hepsi yetmez, tamamını al
            method = "coverage"
            value = coverage_threshold  # type: ignore[assignment]

        codebook_df = df.iloc[:cut_idx].copy()
        outlier_df = df.iloc[cut_idx:].copy()

        covered = int(codebook_df["frequency"].sum())

        # Node ID'lerini codebook'a ekle
        codebook_df["center_node_ids"] = codebook_df["wl_hash"].map(
            lambda h: self._hash_node_ids.get(h, [])
        )
        outlier_df["center_node_ids"] = outlier_df["wl_hash"].map(
            lambda h: self._hash_node_ids.get(h, [])
        )

        return CodebookResult(
            codebook=codebook_df,
            outliers=outlier_df,
            cut_method=method,
            cut_value=value,
            total_patches=total,
            covered_patches=covered,
            coverage_ratio=covered / total if total > 0 else 0.0,
            unique_in_codebook=len(codebook_df),
            unique_outliers=len(outlier_df),
        )

    # ─────────────────────────────────────────────────────────────────
    # Yardımcı — Özet Yazdırma
    # ─────────────────────────────────────────────────────────────────

    def print_summary(self, result: CodebookResult) -> None:
        """Codebook sonucunun özetini konsola yazdırır."""
        sep = "=" * 60
        print(f"\n{sep}")
        print("  CODEBOOK ÖZET RAPORU")
        print(sep)
        print(f"  Kesim Yöntemi      : {result.cut_method}")
        print(f"  Kesim Değeri       : {result.cut_value}")
        print(f"  Toplam Yama        : {result.total_patches:,}")
        print(f"  Kapsanan Yama      : {result.covered_patches:,}")
        print(f"  Kapsam Oranı       : %{result.coverage_ratio * 100:.2f}")
        print(f"  Codebook Boyutu    : {result.unique_in_codebook} benzersiz topoloji")
        print(f"  Outlier Sayısı     : {result.unique_outliers} nadir topoloji")
        print(sep)

        # İlk 10 girişi göster
        top = result.codebook.head(10)
        print("\n  İlk 10 Codebook Girişi:")
        print("  " + "-" * 56)
        print(f"  {'Rank':>4s}  {'Hash':20s}  {'Frekans':>8s}  {'Küm. %':>8s}")
        print("  " + "-" * 56)
        for _, row in top.iterrows():
            print(
                f"  {row['rank']:4d}  "
                f"{row['wl_hash'][:20]:20s}  "
                f"{row['frequency']:8,d}  "
                f"{row['cumulative_percentage']:7.2f}%"
            )
        print()

    # ─────────────────────────────────────────────────────────────────
    # Export — Codebook'u JSON olarak kaydetme
    # ─────────────────────────────────────────────────────────────────

    def save_codebook_json(
        self, result: CodebookResult, output_path: str | Path
    ) -> Path:
        """Codebook sonucunu JSON dosyasına kaydeder.

        Args:
            result: build_codebook() çıktısı.
            output_path: Kaydedilecek dosya yolu.

        Returns:
            Kaydedilen dosyanın Path'i.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        payload: Dict[str, Any] = {
            "meta": {
                "cut_method": result.cut_method,
                "cut_value": result.cut_value,
                "total_patches": result.total_patches,
                "covered_patches": result.covered_patches,
                "coverage_ratio": round(result.coverage_ratio, 6),
                "unique_in_codebook": result.unique_in_codebook,
                "unique_outliers": result.unique_outliers,
                "source_files": self._source_files,
            },
            "codebook": {
                row["wl_hash"]: {
                    "rank": int(row["rank"]),
                    "frequency": int(row["frequency"]),
                    "percentage": float(row["percentage"]),
                    "cumulative_percentage": float(row["cumulative_percentage"]),
                    "center_node_ids": [int(x) for x in row["center_node_ids"]],
                }
                for _, row in result.codebook.iterrows()
            },
            "outlier_hashes": result.outliers["wl_hash"].tolist(),
        }

        output_path.write_text(
            json.dumps(payload, cls=_NumpyEncoder, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return output_path


# ─────────────────────────────────────────────────────────────────────
# Örnek Kullanım (Dummy Veri ile Demo)
# ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import hashlib
    import random

    random.seed(42)
    np.random.seed(42)

    # ── Dummy Veri Üretimi (Zipf Dağılımı) ──
    # Gerçek 3D mesh'lerdeki gibi long-tail dağılım simüle ediyoruz
    NUM_UNIQUE_HASHES = 200
    TOTAL_PATCHES = 50_000

    # Zipf-benzeri frekanslar üret
    zipf_weights = np.array([1.0 / (i ** 1.2) for i in range(1, NUM_UNIQUE_HASHES + 1)])
    zipf_probs = zipf_weights / zipf_weights.sum()
    frequencies = np.random.multinomial(TOTAL_PATCHES, zipf_probs)

    # Rastgele WL-Hash'ler oluştur
    dummy_groups: Dict[str, List[int]] = {}
    node_id_counter = 1

    for i, freq in enumerate(frequencies):
        fake_hash = hashlib.md5(f"topology_{i}".encode()).hexdigest()[:16]
        node_ids = list(range(node_id_counter, node_id_counter + freq))
        node_id_counter += freq
        dummy_groups[fake_hash] = node_ids

    # ── CodebookBuilder Kullanımı ──
    print("=" * 60)
    print("  CODEBOOK BUILDER — DEMO (Dummy Zipf Dağılımı)")
    print("=" * 60)

    builder = CodebookBuilder()
    builder.load_topology_groups(dummy_groups)

    print(f"\n  Toplam Yama    : {builder.total_patches:,}")
    print(f"  Benzersiz Hash : {builder.unique_hashes}")

    # Frekans tablosu
    freq_df = builder.build_frequency_table()
    print(f"\n  Frekans Tablosu (ilk 10 satır):")
    print(freq_df.head(10).to_string(index=False))

    # Pareto grafiği
    print("\n  Pareto grafiği çiziliyor...")
    builder.plot_pareto(
        freq_df,
        coverage_line=95.0,
        title="Demo: WL-Hash Zipf Dağılımı — Pareto Grafiği",
        save_path=None,
        show=True,
    )

    # Codebook — coverage modunda (%95)
    result_cov = builder.build_codebook(freq_df, coverage_threshold=0.95)
    builder.print_summary(result_cov)

    # Codebook — top_k modunda (k=64)
    result_topk = builder.build_codebook(freq_df, top_k=64)
    builder.print_summary(result_topk)

    print("\n  Demo tamamlandı.\n")
