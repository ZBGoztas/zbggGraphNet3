"""
utils/visualization.py  –  Üçlü Mesh Karşılaştırıcı
=====================================================
RAW | REFINED | PREDICTION meshlerini tek pencerede yan yana gösterir.

Kullanım (modül olarak):
    from utils.visualization import show_triple_comparison
    show_triple_comparison(raw_mesh, refined_mesh, pred_mesh)

Kullanım (doğrudan test):
    python utils/visualization.py
"""

from __future__ import annotations

from typing import Union

import numpy as np

try:
    import pyvista as pv
    _HAS_PYVISTA = True
except ImportError as exc:
    raise ImportError("PyVista kurulu değil.  pip install pyvista") from exc

MeshLike = Union[pv.PolyData, pv.UnstructuredGrid]

# ── Görsel sabitler ───────────────────────────────────────────────────────────
_PANEL_CONFIGS = (
    {"title": "RAW MESH  (Girdi)",                "color": "salmon"},
    {"title": "REFINED MESH  (Hedef)",            "color": "palegreen"},
    {"title": "PREDICTION  (Yapay Zeka Çıktısı)", "color": "cornflowerblue"},
)
_BACKGROUND  = "#1c1c2e"
_EDGE_COLOR  = "#555555"
_TEXT_COLOR  = "white"
_FONT_SIZE   = 11
_WINDOW_SIZE = (1800, 700)


def show_triple_comparison(
    raw_mesh:     MeshLike,
    refined_mesh: MeshLike,
    pred_mesh:    MeshLike,
    *,
    window_title: str  = "MeshCrafter – Üçlü Mesh Karşılaştırması",
    show_edges:   bool = True,
    pred_style:   str  = "surface",
) -> None:
    """
    Üç mesh'i yan yana, bağlantılı kamera görünümüyle gösteren interaktif
    pencere açar.

    Parameters
    ----------
    raw_mesh     : ham girdi mesh
    refined_mesh : ground-truth hedef mesh
    pred_mesh    : modelin tahmin ettiği mesh
    window_title : pencere başlığı
    show_edges   : mesh kenarlarını çiz (büyük mesh'lerde False geçilebilir)
    pred_style   : PREDICTION paneli render stili –
                   "surface"   → düz yüzey (varsayılan)
                   "wireframe" → sadece kenarlar (üst üste  binen yamalar için)
                   "points"    → nokta bulutu
    """
    meshes = (raw_mesh, refined_mesh, pred_mesh)

    plotter = pv.Plotter(
        shape=(1, 3),
        title=window_title,
        window_size=_WINDOW_SIZE,
    )

    for col, (mesh, cfg) in enumerate(zip(meshes, _PANEL_CONFIGS)):
        plotter.subplot(0, col)
        plotter.set_background(_BACKGROUND)
        plotter.add_text(
            cfg["title"],
            position="upper_edge",
            font_size=_FONT_SIZE,
            color=_TEXT_COLOR,
            font="courier",
        )

        if col == 2 and pred_style != "surface":
            # PREDICTION: özel render stili (z-fighting önleme)
            plotter.add_mesh(
                mesh,
                color=cfg["color"],
                style=pred_style,
                line_width=1.2,
                point_size=3,
                opacity=1.0,
            )
        else:
            plotter.add_mesh(
                mesh,
                color=cfg["color"],
                show_edges=show_edges,
                edge_color=_EDGE_COLOR,
                line_width=0.8,
                opacity=1.0,
            )

        plotter.add_axes()
        plotter.view_isometric()

    plotter.link_views()
    plotter.show()


# ── Örnek kullanım (dummy meshler) ───────────────────────────────────────────
if __name__ == "__main__":
    rng = np.random.default_rng(seed=42)
    base = dict(radius=1.0, theta_resolution=28, phi_resolution=28)

    raw     = pv.Sphere(**base)
    raw.points = raw.points + rng.normal(0.0, 0.12, raw.points.shape)

    refined = pv.Sphere(**base)

    pred    = pv.Sphere(**base)
    pred.points = pred.points + rng.normal(0.0, 0.035, pred.points.shape)

    print(f"RAW     : {raw.n_points} nokta")
    print(f"REFINED : {refined.n_points} nokta")
    print(f"PRED    : {pred.n_points} nokta")
    print("Pencereyi kapatmak için [Q]\n")

    show_triple_comparison(raw, refined, pred)
