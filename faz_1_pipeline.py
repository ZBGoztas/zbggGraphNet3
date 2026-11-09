"""
FAZ 1 – Veri Hazırlığı ve AI Sözlük (Vocabulary) İnşası Pipeline'ı
====================================================================
Bu script, ham .nas mesh dosyalarından başlayarak PyTorch Transformer
modelinin okuyabileceği bir `ai_vocabulary.json` sözlük dosyası üretir.

Adımlar:
    1. Veri Yükleme & Yama Çıkarma   (PatchExtractor)
    2. Topoloji Analizi – WL-Hash     (PatchAnalyzer)
    3. Frekans Analizi & Şablon Seçimi
    4. AI Vocabulary JSON Üretimi

Kullanım:
    python faz_1_pipeline.py                          # varsayılan config.yaml
    python faz_1_pipeline.py --source refined         # refined meshler
    python faz_1_pipeline.py --mesh-dir ./my_meshes   # özel klasör
"""

import json
import sys
import argparse
from pathlib import Path
from collections import Counter
from datetime import datetime, timezone
from typing import Dict, List, Tuple, Any

# ── Proje kökünü sys.path'e ekle ──────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.parser_utils import ParserNas
from utils.patch_extractor import PatchExtractor, PatchAnalyzer, Patch
from utils.config_utils import load_config
from utils.logger_utils import get_context_logger


# ====================================================================== #
#  Sabitler
# ====================================================================== #

SPECIAL_TOKENS: List[Dict[str, str]] = [
    {"name": "<PAD>",   "description": "Padding (Doldurma)"},
    {"name": "<START>", "description": "Diziyi Başlat"},
    {"name": "<END>",   "description": "Diziyi Bitir (Mesh Tamamlandı)"},
    {"name": "<UNK>",   "description": "Bilinmeyen Yama (Out-of-Vocabulary)"},
]

DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data_vocabulary"
DEFAULT_VOCAB_NAME = "ai_vocabulary.json"


# ====================================================================== #
#  Yardımcı Fonksiyonlar
# ====================================================================== #

def collect_nas_files(mesh_dir: Path) -> List[Path]:
    """Verilen klasördeki .nas dosyalarını deterministik sırada döndürür."""
    files = sorted(mesh_dir.glob("*.nas"), key=lambda p: p.name)
    return files


def _serialize_local_nodes(
    local_nodes: Dict[int, Tuple[float, float, float]],
) -> Dict[str, List[float]]:
    """Lokal node dict'ini JSON-uyumlu formata çevirir (int key -> str key)."""
    return {
        str(lid): [round(c, 10) for c in coords]
        for lid, coords in sorted(local_nodes.items())
    }


def _serialize_local_elements(
    local_elements: Dict[int, Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    """Lokal element dict'ini JSON-uyumlu formata çevirir."""
    return {
        str(eid): edata
        for eid, edata in sorted(local_elements.items())
    }


# ====================================================================== #
#  ADIM 1 – Veri Yükleme ve Yama Çıkarma
# ====================================================================== #

def step1_extract_all(
    nas_files: List[Path],
    logger,
) -> Tuple[Dict[str, Patch], int, int]:
    """
    Tüm .nas dosyalarını sırayla okur, her birinden 1-Ring yamalarını
    koparır ve tek bir havuzda toplar.

    Returns:
        (merged_patches, total_nodes_processed, total_skipped)
        merged_patches key formatı: "<dosya_adı>::<global_node_id>"
    """
    parser = ParserNas()
    merged_patches: Dict[str, Patch] = {}
    total_nodes = 0
    total_skipped = 0

    for idx, nas_path in enumerate(nas_files, start=1):
        file_label = nas_path.stem
        logger.info(f"[{idx}/{len(nas_files)}] İşleniyor: {nas_path.name}")

        mesh_data = parser.parse_file(nas_path)
        nodes = mesh_data["nodes"]
        elements = mesh_data["elements"]

        extractor = PatchExtractor(file_name=file_label)
        patches_dict, skipped_count = extractor.extract_all_patches(nodes, elements)

        total_nodes += len(nodes)
        total_skipped += skipped_count

        # Havuza ekle – benzersiz anahtar: "dosya::node_id"
        for nid, patch in patches_dict.items():
            unique_key = f"{file_label}::{nid}"
            merged_patches[unique_key] = patch

        logger.info(
            f"  -> {len(patches_dict)} yama çıkarıldı, "
            f"{skipped_count} düğüm atlandı"
        )

    logger.info(
        f"ADIM 1 Tamamlandı | "
        f"Dosya={len(nas_files)} | "
        f"Toplam Düğüm={total_nodes} | "
        f"Toplam Yama={len(merged_patches)} | "
        f"Atlanan={total_skipped}"
    )
    return merged_patches, total_nodes, total_skipped


# ====================================================================== #
#  ADIM 2 – Topoloji Analizi (WL-Hash)
# ====================================================================== #

def step2_group_by_topology(
    merged_patches: Dict[str, Patch],
    logger,
) -> Dict[str, List[str]]:
    """
    Tüm yamaların WL-Hash'ini hesaplar ve aynı hash'e sahip olanları gruplar.

    Returns:
        {wl_hash: [unique_key_1, unique_key_2, ...]}
    """
    analyzer = PatchAnalyzer(file_name="pipeline")

    # Hash hesapla
    for patch in merged_patches.values():
        analyzer.compute_hash(patch)

    # Grupla
    from collections import defaultdict
    groups: Dict[str, List[str]] = defaultdict(list)
    for key, patch in merged_patches.items():
        groups[patch.topology_hash].append(key)

    unique_count = len(groups)
    logger.info(
        f"ADIM 2 Tamamlandı | "
        f"{len(merged_patches)} yama -> {unique_count} benzersiz topoloji"
    )
    return dict(groups)


# ====================================================================== #
#  ADIM 3 – Frekans Analizi ve Şablon (Template) Seçimi
# ====================================================================== #

def step3_frequency_and_templates(
    groups: Dict[str, List[str]],
    merged_patches: Dict[str, Patch],
    logger,
) -> List[Tuple[str, int, Patch]]:
    """
    Her WL-Hash'in frekansını sayar, en sıktan en seyreğe sıralar ve
    her benzersiz topoloji için İLK yamayı şablon (template) olarak seçer.

    Returns:
        Sıralı liste: [(wl_hash, frequency, template_patch), ...]
    """
    freq_counter = Counter({wl_hash: len(keys) for wl_hash, keys in groups.items()})

    # Deterministik sıralama: frekans azalan, eşitlikte hash string artan
    sorted_hashes = sorted(
        freq_counter.items(),
        key=lambda item: (-item[1], item[0]),
    )

    templates: List[Tuple[str, int, Patch]] = []
    for wl_hash, freq in sorted_hashes:
        # Grubun İLK anahtarını al (deterministik: sorted key)
        first_key = sorted(groups[wl_hash])[0]
        template_patch = merged_patches[first_key]
        templates.append((wl_hash, freq, template_patch))

    logger.info(
        f"ADIM 3 Tamamlandı | "
        f"{len(templates)} benzersiz şablon, "
        f"en yüksek frekans={sorted_hashes[0][1] if sorted_hashes else 0}"
    )
    return templates


# ====================================================================== #
#  ADIM 4 – AI Vocabulary JSON Üretimi
# ====================================================================== #

def step4_build_vocabulary(
    templates: List[Tuple[str, int, Patch]],
    metadata: Dict[str, Any],
    output_path: Path,
    logger,
) -> None:
    """
    Özel tokenlar + frekans sıralı yama şablonlarından ai_vocabulary.json
    dosyasını üretir.
    """
    vocabulary: Dict[str, Any] = {}

    # ── Metadata ────────────────────────────────────────────────────────
    vocabulary["metadata"] = metadata

    # ── Özel Tokenlar (0-3) ─────────────────────────────────────────────
    for token_id, spec in enumerate(SPECIAL_TOKENS):
        vocabulary[str(token_id)] = {
            "type": "SPECIAL",
            "name": spec["name"],
            "description": spec["description"],
            "wl_hash": None,
            "frequency": None,
            "local_nodes": None,
            "local_elements": None,
        }

    # ── Yama Tokenları (4, 5, 6, ...) ──────────────────────────────────
    for offset, (wl_hash, freq, patch) in enumerate(templates):
        token_id = offset + len(SPECIAL_TOKENS)
        vocabulary[str(token_id)] = {
            "type": "PATCH",
            "wl_hash": wl_hash,
            "frequency": freq,
            "local_nodes": _serialize_local_nodes(patch.local_nodes),
            "local_elements": _serialize_local_elements(patch.local_elements),
        }

    # ── Dosyaya yaz ────────────────────────────────────────────────────
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(vocabulary, f, indent=4, ensure_ascii=False)

    logger.info(f"ADIM 4 Tamamlandı | Sözlük kaydedildi: {output_path}")
    logger.info(
        f"  Token Sayısı: {len(vocabulary) - 1} "  # metadata hariç
        f"({len(SPECIAL_TOKENS)} Özel + {len(templates)} Yama)"
    )


# ====================================================================== #
#  Konsol Raporu
# ====================================================================== #

def print_summary(
    total_files: int,
    total_nodes: int,
    total_skipped: int,
    total_patches: int,
    unique_topologies: int,
    output_path: Path,
) -> None:
    """İşlem özetini konsola yazdırır."""
    sep = "=" * 64
    print(f"\n{sep}")
    print("  FAZ 1 – Pipeline Raporu")
    print(sep)
    print(f"  İşlenen .nas dosyası     : {total_files}")
    print(f"  Toplam düğüm (node)      : {total_nodes:,}")
    print(f"  Toplam çıkarılan yama    : {total_patches:,}")
    print(f"  Atlanan düğüm            : {total_skipped:,}")
    print(f"  Benzersiz topoloji sayısı: {unique_topologies}")
    print(f"  Özel token sayısı        : {len(SPECIAL_TOKENS)}")
    print(f"  Toplam sözlük boyutu     : {len(SPECIAL_TOKENS) + unique_topologies}")
    print(f"  Çıktı dosyası            : {output_path}")
    print(f"{sep}\n")


# ====================================================================== #
#  CLI & Main
# ====================================================================== #

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="FAZ 1: Veri Hazırlığı ve AI Sözlük İnşası",
    )
    parser.add_argument(
        "--source",
        choices=["raw", "refined"],
        default="refined",
        help="Hangi mesh kaynağı kullanılsın (varsayılan: refined)",
    )
    parser.add_argument(
        "--mesh-dir",
        type=Path,
        default=None,
        help="Özel mesh klasör yolu. Verilirse --source yok sayılır.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Çıktı klasörü (varsayılan: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--output-name",
        type=str,
        default=DEFAULT_VOCAB_NAME,
        help=f"Çıktı dosya adı (varsayılan: {DEFAULT_VOCAB_NAME})",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logger = get_context_logger(name="faz1_pipeline", file_name="pipeline")

    # ── Mesh klasörünü belirle ──────────────────────────────────────────
    if args.mesh_dir is not None:
        mesh_dir = args.mesh_dir.resolve()
    else:
        config = load_config()
        key = "raw_output_dir" if args.source == "raw" else "refined_output_dir"
        mesh_dir = Path(config[key])

    if not mesh_dir.is_dir():
        logger.error(f"Mesh klasörü bulunamadı: {mesh_dir}")
        sys.exit(1)

    nas_files = collect_nas_files(mesh_dir)
    if not nas_files:
        logger.error(f"Klasörde .nas dosyası bulunamadı: {mesh_dir}")
        sys.exit(1)

    logger.info(f"Kaynak klasör: {mesh_dir} ({len(nas_files)} dosya)")

    output_path = args.output_dir.resolve() / args.output_name

    # ── ADIM 1 ──────────────────────────────────────────────────────────
    merged_patches, total_nodes, total_skipped = step1_extract_all(
        nas_files, logger,
    )

    # ── ADIM 2 ──────────────────────────────────────────────────────────
    groups = step2_group_by_topology(merged_patches, logger)

    # ── ADIM 3 ──────────────────────────────────────────────────────────
    templates = step3_frequency_and_templates(groups, merged_patches, logger)

    # ── Metadata ────────────────────────────────────────────────────────
    metadata = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_dir": str(mesh_dir),
        "total_files": len(nas_files),
        "total_nodes_processed": total_nodes,
        "total_patches_extracted": len(merged_patches),
        "total_nodes_skipped": total_skipped,
        "unique_topologies": len(templates),
        "special_tokens": len(SPECIAL_TOKENS),
        "vocab_size": len(SPECIAL_TOKENS) + len(templates),
    }

    # ── ADIM 4 ──────────────────────────────────────────────────────────
    step4_build_vocabulary(templates, metadata, output_path, logger)

    # ── Konsol Raporu ───────────────────────────────────────────────────
    print_summary(
        total_files=len(nas_files),
        total_nodes=total_nodes,
        total_skipped=total_skipped,
        total_patches=len(merged_patches),
        unique_topologies=len(templates),
        output_path=output_path,
    )


if __name__ == "__main__":
    main()
