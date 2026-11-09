"""
Normalize edilmiş NASTRAN mesh dosyalarını yan yana görselleştirir.
Sol: RAW normalized | Sağ: REFINED normalized

Kullanım:
    python scripts/visualize_mesh.py                  # tüm dosyalar sırayla
    python scripts/visualize_mesh.py -f TUNNEL        # sadece TUNNEL içerenler
    python scripts/visualize_mesh.py --no-edges       # edge'siz hızlı render
"""

import sys
import argparse
from pathlib import Path

import numpy as np

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from utils.config_utils import load_config
from utils.parser_utils import ParserNas

try:
    import pyvista as pv
except ImportError:
    sys.exit("PyVista kurulu değil. Kurmak için: pip install pyvista")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Normalize edilmiş NASTRAN mesh görselleştirici (RAW vs REFINED)."
    )
    ap.add_argument("-f", "--filter", type=str, default=None,
                    help="Dosya adı filtresi (büyük/küçük harf duyarsız).")
    ap.add_argument("--no-edges", action="store_true",
                    help="Edge çizgilerini devre dışı bırak (büyük mesh'lerde hızlı).")
    return ap.parse_args()


def parsed_to_pyvista(mesh_data: dict) -> pv.PolyData:
    """ParserNas çıktısını PyVista PolyData mesh'ine dönüştürür."""
    nodes = mesh_data["nodes"]
    elements = mesh_data["elements"]

    if not nodes:
        return pv.PolyData()

    sorted_nids = sorted(nodes.keys())
    nid_to_idx = {nid: idx for idx, nid in enumerate(sorted_nids)}

    # Points array (N x 3)
    points = np.array([nodes[nid] for nid in sorted_nids], dtype=np.float64)

    # Faces array: [n_verts, v0, v1, ..., n_verts, v0, v1, ...]
    valid_nids = set(sorted_nids)
    face_list = []
    for eid in sorted(elements):
        elem = elements[eid]
        elem_nodes = elem["nodes"]
        if not all(nid in valid_nids for nid in elem_nodes):
            continue
        n = len(elem_nodes)
        face_list.append(n)
        face_list.extend(nid_to_idx[nid] for nid in elem_nodes)

    if not face_list:
        return pv.PolyData(points)

    faces = np.array(face_list, dtype=np.int64)
    return pv.PolyData(points, faces)


def visualize_pair(
    raw_path: Path,
    refined_path: Path,
    file_name: str,
    show_edges: bool = True,
) -> None:
    """RAW ve REFINED mesh'i yan yana gösterir."""
    parser = ParserNas()
    raw_data = parser.parse_file(raw_path)
    refined_data = parser.parse_file(refined_path)

    raw_mesh = parsed_to_pyvista(raw_data)
    refined_mesh = parsed_to_pyvista(refined_data)

    plotter = pv.Plotter(
        shape=(1, 2),
        title=file_name,
        window_size=(1600, 800),
    )

    mesh_kwargs = dict(
        show_edges=show_edges,
        edge_color="black",
        line_width=0.5,
        opacity=1.0,
    )

    # Sol: RAW
    plotter.subplot(0, 0)
    plotter.add_text(f"RAW\n{file_name}", font_size=9)
    plotter.add_mesh(raw_mesh, color="lightblue", **mesh_kwargs)
    plotter.add_axes()
    plotter.view_isometric()

    # Sağ: REFINED
    plotter.subplot(0, 1)
    plotter.add_text(f"REFINED\n{file_name}", font_size=9)
    plotter.add_mesh(refined_mesh, color="lightyellow", **mesh_kwargs)
    plotter.add_axes()
    plotter.view_isometric()

    plotter.link_views()
    plotter.show()


def main() -> None:
    args = parse_args()
    config = load_config()

    raw_out = Path(config["raw_output_dir"])
    refined_out = Path(config["refined_output_dir"])

    if not raw_out.is_dir():
        sys.exit(f"raw_output_dir bulunamadı: {raw_out}")
    if not refined_out.is_dir():
        sys.exit(f"refined_output_dir bulunamadı: {refined_out}")

    raw_files = sorted(raw_out.glob("*.nas"))

    if args.filter:
        flt = args.filter.lower()
        raw_files = [f for f in raw_files if flt in f.name.lower()]

    if not raw_files:
        sys.exit("Eşleşen .nas dosyası bulunamadı.")

    print(f"{len(raw_files)} dosya bulundu.\n")

    for idx, raw_path in enumerate(raw_files, 1):
        refined_path = refined_out / raw_path.name

        if not refined_path.exists():
            print(f"[{idx}/{len(raw_files)}] ATLA: Refined eşleşmesi yok -> {raw_path.name}")
            continue

        print(f"[{idx}/{len(raw_files)}] {raw_path.name}")

        try:
            visualize_pair(raw_path, refined_path, raw_path.name, show_edges=not args.no_edges)
        except Exception as e:
            print(f"  HATA: {e}")
            continue

        if idx < len(raw_files):
            resp = input("  Sonraki için Enter, çıkmak için 'q': ").strip().lower()
            if resp == "q":
                break

    print("Tamamlandı.")


if __name__ == "__main__":
    main()
