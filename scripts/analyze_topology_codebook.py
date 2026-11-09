"""
Seçilen WL-hash topolojileri için temsilci patch analizi yapar.

Tek bir *_patches.json dosyasından hash gruplarını ve Patch nesnelerini yükler,
her hash için ilk merkez düğümü temsilci olarak seçer ve yamanın node/element
özetini hizalı bir konsol raporu olarak yazdırır.

Kullanım:
    python scripts/analyze_topology_codebook.py --json data/foo_patches.json
    python scripts/analyze_topology_codebook.py --json data/foo_patches.json --top-k 5
    python scripts/analyze_topology_codebook.py --json data/foo_patches.json --hash abc123 --hash def456
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from utils.patch_extractor import Patch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Temsilci patch uzerinden top hash topoloji ozeti uretir."
    )
    parser.add_argument(
        "--json",
        required=True,
        help="Analiz edilecek *_patches.json dosya yolu.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Hash listesi verilmezse analiz edilecek en populer hash sayisi.",
    )
    parser.add_argument(
        "--hash",
        dest="hashes",
        action="append",
        default=None,
        help="Analiz edilecek WL-hash. Birden fazla kez verilebilir.",
    )
    parser.add_argument(
        "--full-hash",
        action="store_true",
        help="Hash degerlerini kisaltmadan yazdir.",
    )
    return parser.parse_args()


def patch_from_dict(payload: Dict[str, Any]) -> Patch:
    """JSON icindeki patch kaydini Patch nesnesine donusturur."""
    local_nodes = {
        int(local_id): tuple(coords)
        for local_id, coords in payload["local_nodes"].items()
    }
    local_elements = {
        int(element_id): {
            "type": element_data["type"],
            "nodes": [int(node_id) for node_id in element_data["nodes"]],
        }
        for element_id, element_data in payload["local_elements"].items()
    }
    global_to_local = {
        int(global_id): int(local_id)
        for global_id, local_id in payload["global_to_local"].items()
    }

    return Patch(
        center_global_id=int(payload["center_global_id"]),
        local_nodes=local_nodes,
        local_elements=local_elements,
        global_to_local=global_to_local,
        topology_hash=payload.get("topology_hash", ""),
    )


def load_patch_json(json_path: str | Path) -> tuple[str, Dict[str, List[int]], Dict[int, Patch]]:
    """Tek bir *_patches.json dosyasindan grup ve patch verilerini yukler."""
    json_path = Path(json_path)
    with json_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    hash_groups = {
        wl_hash: [int(node_id) for node_id in info.get("center_node_ids", [])]
        for wl_hash, info in data.get("topology_groups", {}).items()
    }
    all_patches = {
        int(node_id): patch_from_dict(patch_payload)
        for node_id, patch_payload in data.get("patches", {}).items()
    }

    return data.get("file_name", json_path.name), hash_groups, all_patches


def select_top_hashes(hash_groups: Dict[str, List[int]], top_k: int) -> List[str]:
    """Frekansa gore en populer K hash'i secer."""
    ranked = sorted(hash_groups.items(), key=lambda item: len(item[1]), reverse=True)
    return [wl_hash for wl_hash, _ in ranked[:top_k]]


def analyze_top_hashes(
    top_hashes: List[str],
    hash_groups: Dict[str, List[int]],
    all_patches: Dict[int, Patch],
) -> List[Dict[str, Any]]:
    """Her hash icin tek temsilci patch kullanarak topoloji ozeti cikarir.

    Analiz asamasi yalnizca secilen hash listesi uzerinde ilerler; bu nedenle
    karmasiklik O(K)'dir.
    """
    rows: List[Dict[str, Any]] = []

    for rank, wl_hash in enumerate(top_hashes, start=1):
        center_node_ids = hash_groups.get(wl_hash, [])
        if not center_node_ids:
            raise KeyError(f"Hash icin temsilci node bulunamadi: {wl_hash}")

        representative_node_id = center_node_ids[0]
        patch = all_patches.get(representative_node_id)
        if patch is None:
            raise KeyError(
                f"Temsilci patch bulunamadi | hash={wl_hash} | node={representative_node_id}"
            )

        element_counter = Counter(
            str(element_data.get("type", "unknown")).lower()
            for element_data in patch.local_elements.values()
        )

        rows.append(
            {
                "rank": rank,
                "wl_hash": wl_hash,
                "frequency": len(center_node_ids),
                "node_count": len(patch.local_nodes),
                "quad_count": element_counter.get("quad", 0),
                "tria_count": element_counter.get("tria", 0),
                "other_elements": {
                    elem_type: count
                    for elem_type, count in sorted(element_counter.items())
                    if elem_type not in {"quad", "tria"}
                },
            }
        )

    return rows


def format_element_summary(row: Dict[str, Any]) -> str:
    """Eleman tip dagilimini tek satirlik okunabilir metne donusturur."""
    parts = [f"{row['quad_count']} Quad", f"{row['tria_count']} Tria"]
    for elem_type, count in row["other_elements"].items():
        parts.append(f"{count} {elem_type.title()}")
    return ", ".join(parts)


def print_summary_report(rows: List[Dict[str, Any]], file_name: str, full_hash: bool = False) -> None:
    """Analiz sonucunu hizali konsol raporu olarak yazdirir."""
    if not rows:
        print("Analiz edilecek hash bulunamadi.")
        return

    header = f"TOP {len(rows)} YAMA TOPOLOJI ANALIZI"
    print(f"\n--- {header} ---")
    print(f"Kaynak Dosya: {file_name}")

    frequency_width = max(len(f"{row['frequency']:,}") for row in rows)
    node_width = max(len(str(row["node_count"])) for row in rows)

    for row in rows:
        hash_value = row["wl_hash"] if full_hash else f"{row['wl_hash'][:12]}..."
        element_summary = format_element_summary(row)
        print(
            f"Rank {row['rank']:>2} | "
            f"Hash: {hash_value:<15} | "
            f"Frekans: {row['frequency']:{frequency_width},} | "
            f"Toplam Node: {row['node_count']:{node_width}d} | "
            f"Elemanlar: {element_summary}"
        )


def main() -> None:
    args = parse_args()
    file_name, hash_groups, all_patches = load_patch_json(args.json)

    top_hashes = args.hashes if args.hashes else select_top_hashes(hash_groups, args.top_k)
    report_rows = analyze_top_hashes(top_hashes, hash_groups, all_patches)
    print_summary_report(report_rows, file_name=file_name, full_hash=args.full_hash)


if __name__ == "__main__":
    main()