"""
RAW ve REFINED mesh dosyalarını karşılaştırır.
Node koordinatları, element bağlantıları ve genel istatistikleri raporlar.

Kullanım:
    python scripts/compare_meshes.py                  # tüm dosyaları karşılaştır
    python scripts/compare_meshes.py -f TUNNEL        # sadece TUNNEL içerenler
    python scripts/compare_meshes.py --tol 1e-3       # koordinat toleransı değiştir
    python scripts/compare_meshes.py --normalized      # normalize çıktıları karşılaştır
"""

import sys
from scipy.spatial import cKDTree
import argparse
import csv
from pathlib import Path

import numpy as np

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from utils.config_utils import load_config
from utils.parser_utils import ParserNas


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="RAW vs REFINED mesh karşılaştırma aracı."
    )
    ap.add_argument("-f", "--filter", type=str, default=None,
                    help="Dosya adı filtresi (büyük/küçük harf duyarsız).")
    ap.add_argument("--tol", type=float, default=1e-6,
                    help="Koordinat eşleşme toleransı (varsayılan: 1e-6).")
    ap.add_argument("--normalized", action="store_true",
                    help="Normalize edilmiş çıktıları karşılaştır (varsayılan: ham dosyalar).")
    ap.add_argument("-o", "--output", type=str, default=None,
                    help="Sonuçları CSV dosyasına yaz.")
    return ap.parse_args()


def compare_nodes(raw_nodes: dict, ref_nodes: dict, tol: float) -> tuple[dict, dict]:
    """
    KD-Tree ile uzamsal node eşleştirmesi yapar (ID bağımsız).
    Her RAW node için en yakın REFINED node'u bulur; tolerans içindeyse eşleştirir.

    Returns:
        (node_stats, node_mapping)
        node_mapping: {raw_nid: ref_nid} — eşleşen çiftler
    """
    raw_nids = list(raw_nodes.keys())
    ref_nids = list(ref_nodes.keys())

    raw_coords = np.array([raw_nodes[nid] for nid in raw_nids], dtype=np.float64)
    ref_coords = np.array([ref_nodes[nid] for nid in ref_nids], dtype=np.float64)

    # REFINED koordinatlarından KD-Tree oluştur
    tree = cKDTree(ref_coords)

    # Her RAW node için en yakın REFINED node'u bul
    dists, indices = tree.query(raw_coords, k=1)

    node_mapping: dict[int, int] = {}
    matched_ref_indices: set[int] = set()
    matched_dists: list[float] = []
    unmatched_raw = 0

    for i, (dist, ref_idx) in enumerate(zip(dists, indices)):
        if dist <= tol:
            raw_nid = raw_nids[i]
            ref_nid = ref_nids[ref_idx]
            node_mapping[raw_nid] = ref_nid
            matched_ref_indices.add(ref_idx)
            matched_dists.append(float(dist))
        else:
            unmatched_raw += 1

    matched_count = len(node_mapping)
    unmatched_ref = len(ref_nids) - len(matched_ref_indices)

    matched_dists_arr = np.array(matched_dists) if matched_dists else np.array([0.0])

    node_stats = {
        "raw_count": len(raw_nids),
        "ref_count": len(ref_nids),
        "matched": matched_count,
        "unmatched_raw": unmatched_raw,
        "unmatched_ref": unmatched_ref,
        "match_pct": 100.0 * matched_count / len(raw_nids) if raw_nids else 0.0,
        "dist_mean": float(np.mean(matched_dists_arr)),
        "dist_max": float(np.max(matched_dists_arr)),
        "dist_median": float(np.median(matched_dists_arr)),
    }

    return node_stats, node_mapping


def compare_elements(raw_elems: dict, ref_elems: dict, node_mapping: dict) -> dict:
    """
    Topolojik element eşleştirmesi yapar (ID bağımsız).
    RAW element node'larını node_mapping ile REFINED ID'lerine çevirir,
    sıralı tuple olarak karşılaştırır.
    """
    raw_tria = sum(1 for e in raw_elems.values() if e["type"] == "tria")
    raw_quad = sum(1 for e in raw_elems.values() if e["type"] == "quad")
    ref_tria = sum(1 for e in ref_elems.values() if e["type"] == "tria")
    ref_quad = sum(1 for e in ref_elems.values() if e["type"] == "quad")

    # REFINED tarafı: (type, sorted_node_tuple) kümesi oluştur
    ref_topo_set: set[tuple] = set()
    for elem in ref_elems.values():
        key = (elem["type"], tuple(sorted(elem["nodes"])))
        ref_topo_set.add(key)

    # RAW tarafı: node_mapping ile translate et, eşleşmeyenleri say
    translated = 0
    untranslatable = 0
    topo_matched = 0

    for elem in raw_elems.values():
        mapped_nodes = []
        ok = True
        for nid in elem["nodes"]:
            ref_nid = node_mapping.get(nid)
            if ref_nid is None:
                ok = False
                break
            mapped_nodes.append(ref_nid)

        if not ok:
            untranslatable += 1
            continue

        translated += 1
        key = (elem["type"], tuple(sorted(mapped_nodes)))
        if key in ref_topo_set:
            topo_matched += 1

    raw_count = len(raw_elems)
    ref_count = len(ref_elems)

    return {
        "raw_count": raw_count,
        "ref_count": ref_count,
        "translated": translated,
        "untranslatable": untranslatable,
        "topo_matched": topo_matched,
        "topo_unmatched": translated - topo_matched,
        "match_pct": 100.0 * topo_matched / raw_count if raw_count else 0.0,
        "raw_tria": raw_tria,
        "raw_quad": raw_quad,
        "ref_tria": ref_tria,
        "ref_quad": ref_quad,
    }


def print_report(file_name: str, node_stats: dict, elem_stats: dict) -> None:
    """Tekil dosya karşılaştırma raporunu yazdırır."""
    sep = "=" * 60
    print(f"\n{sep}")
    print(f"  {file_name}")
    print(sep)

    print(f"\n  NODE KARSILASTIRMASI (Uzamsal Eslestirme)")
    print(f"  {'RAW node sayısı':<35} {node_stats['raw_count']:>8}")
    print(f"  {'REFINED node sayısı':<35} {node_stats['ref_count']:>8}")
    print(f"  {'Uzamsal eslesen node':<35} {node_stats['matched']:>8}")
    print(f"  {'Eslesmeyen RAW node':<35} {node_stats['unmatched_raw']:>8}")
    print(f"  {'Eslesmeyen REFINED node':<35} {node_stats['unmatched_ref']:>8}")
    print(f"  {'Node esleme oranı':<35} {node_stats['match_pct']:>7.1f}%")

    if node_stats["matched"] > 0:
        print(f"\n  Mesafe istatistikleri (eslesen node'lar):")
        print(f"  {'Ortalama mesafe':<35} {node_stats['dist_mean']:>12.6f}")
        print(f"  {'Medyan mesafe':<35} {node_stats['dist_median']:>12.6f}")
        print(f"  {'Maksimum mesafe':<35} {node_stats['dist_max']:>12.6f}")

    print(f"\n  ELEMENT KARSILASTIRMASI (Topolojik Eslestirme)")
    print(f"  {'RAW element sayısı':<35} {elem_stats['raw_count']:>8}")
    print(f"    CTRIA3: {elem_stats['raw_tria']}, CQUAD4: {elem_stats['raw_quad']}")
    print(f"  {'REFINED element sayısı':<35} {elem_stats['ref_count']:>8}")
    print(f"    CTRIA3: {elem_stats['ref_tria']}, CQUAD4: {elem_stats['ref_quad']}")
    print(f"  {'Translate edilebilen':<35} {elem_stats['translated']:>8}")
    print(f"  {'Translate edilemeyen (node yok)':<35} {elem_stats['untranslatable']:>8}")
    print(f"  {'Topolojik eslesen element':<35} {elem_stats['topo_matched']:>8}")
    print(f"  {'Topolojik eslesmeyen':<35} {elem_stats['topo_unmatched']:>8}")
    print(f"  {'Element esleme oranı':<35} {elem_stats['match_pct']:>7.1f}%")


def write_csv(output_path: str, results: list[dict]) -> None:
    """Tüm sonuçları CSV dosyasına yazar."""
    if not results:
        return
    fieldnames = results[0].keys()
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    print(f"\nCSV yazıldı: {output_path}")


def main() -> None:
    args = parse_args()
    config = load_config()
    parser = ParserNas()

    if args.normalized:
        raw_dir = Path(config["raw_output_dir"])
        ref_dir = Path(config["refined_output_dir"])
        print("Mod: Normalize edilmiş çıktılar karşılaştırılıyor")
    else:
        raw_dir = Path(config["raw_dir"])
        ref_dir = Path(config["refined_dir"])
        print("Mod: Ham dosyalar karşılaştırılıyor")

    if not raw_dir.is_dir():
        sys.exit(f"Dizin bulunamadı: {raw_dir}")
    if not ref_dir.is_dir():
        sys.exit(f"Dizin bulunamadı: {ref_dir}")

    raw_files = sorted(raw_dir.glob("*.nas"))

    if args.filter:
        flt = args.filter.lower()
        raw_files = [f for f in raw_files if flt in f.name.lower()]

    if not raw_files:
        sys.exit("Eşleşen .nas dosyası bulunamadı.")

    print(f"Tolerans: {args.tol}")
    print(f"{len(raw_files)} dosya bulundu.\n")

    csv_rows = []

    for idx, raw_path in enumerate(raw_files, 1):
        ref_path = ref_dir / raw_path.name

        if not ref_path.exists():
            print(f"[{idx}/{len(raw_files)}] ATLA: Refined eşleşmesi yok -> {raw_path.name}")
            continue

        raw_data = parser.parse_file(raw_path)
        ref_data = parser.parse_file(ref_path)

        node_stats, node_mapping = compare_nodes(raw_data["nodes"], ref_data["nodes"], args.tol)
        elem_stats = compare_elements(raw_data["elements"], ref_data["elements"], node_mapping)

        print_report(raw_path.name, node_stats, elem_stats)

        csv_rows.append({
            "file_name": raw_path.name,
            **{f"node_{k}": v for k, v in node_stats.items()},
            **{f"elem_{k}": v for k, v in elem_stats.items()},
        })

    # Özet
    if len(csv_rows) > 1:
        print(f"\n{'=' * 60}")
        print(f"  GENEL OZET ({len(csv_rows)} dosya)")
        print(f"{'=' * 60}")
        total_raw_nodes = sum(r["node_raw_count"] for r in csv_rows)
        total_matched_nodes = sum(r["node_matched"] for r in csv_rows)
        total_raw_elems = sum(r["elem_raw_count"] for r in csv_rows)
        total_topo_matched = sum(r["elem_topo_matched"] for r in csv_rows)

        node_pct = 100.0 * total_matched_nodes / total_raw_nodes if total_raw_nodes else 0.0
        elem_pct = 100.0 * total_topo_matched / total_raw_elems if total_raw_elems else 0.0

        print(f"\n  {'Toplam RAW node':<35} {total_raw_nodes:>8}")
        print(f"  {'Uzamsal eslesen node':<35} {total_matched_nodes:>8}")
        print(f"  {'Node esleme oranı':<35} {node_pct:>7.1f}%")
        print(f"\n  {'Toplam RAW element':<35} {total_raw_elems:>8}")
        print(f"  {'Topolojik eslesen element':<35} {total_topo_matched:>8}")
        print(f"  {'Element esleme oranı':<35} {elem_pct:>7.1f}%")

    if args.output:
        write_csv(args.output, csv_rows)

    print("\nTamamlandı.")


if __name__ == "__main__":
    main()
