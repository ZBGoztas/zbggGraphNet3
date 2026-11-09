"""
FAZ 2 – Veri Serileştirme ve Graf Dönüşümü Pipeline'ı
=======================================================
Ham (RAW) ve rafine (REFINED) mesh çiftlerini işleyerek GNN-Transformer
modelinin eğitiminde kullanılacak (Girdi, Hedef) çiftleri üretir.

Her çift için:
  - Girdi  : PyG Data nesnesi  (raw mesh  -> node features + edge_index + pos)
  - Hedef  : Token ID dizisi   (refined mesh -> canonical order -> WL-Hash -> vocab ID)

Çıktı: <processed_dir>/<stem>.pt  (torch.save ile serileştirilmiş Data nesnesi)

Kullanım:
    python faz_2_pipeline.py
    python faz_2_pipeline.py --raw-dir /path/to/raw --refined-dir /path/to/refined
    python faz_2_pipeline.py --vocab data_vocabulary/ai_vocabulary.json --output-dir ./out
"""

import csv
import json
import sys
import time
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Any

import numpy as np

# ── Proje kökünü sys.path'e ekle ──────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
from torch_geometric.data import Data

from utils.parser_utils import ParserNas
from utils.patch_extractor import PatchExtractor, PatchAnalyzer
from utils.canonical_order import apply_canonical_order
from utils.config_utils import load_config
from utils.logger_utils import get_context_logger


# ====================================================================== #
#  Sabitler                                                               #
# ====================================================================== #

DEFAULT_VOCAB_PATH = PROJECT_ROOT / "data_vocabulary" / "ai_vocabulary.json"
DEFAULT_OUTPUT_DIR  = PROJECT_ROOT / "data" / "processed"
DEFAULT_NORM_CSV    = Path(r"<local-path>")

# ai_vocabulary.json'daki özel token ID'leri (FAZ 1 ile uyumlu)
TOKEN_PAD   = 0
TOKEN_START = 1
TOKEN_END   = 2
TOKEN_UNK   = 3


# ====================================================================== #
#  ADIM 0: Dosya Eşleştirme                                              #
# ====================================================================== #

def _stem_key(path: Path) -> str:
    """
    Dosya adından eşleştirme anahtarı üretir.
    '_raw' veya '_refined' son eklerini temizler.

    Örnekler:
        "part_01_raw.nas"      -> "part_01"
        "part_01_refined.nas"  -> "part_01"
        "part_01.nas"          -> "part_01"
    """
    stem = path.stem
    for suffix in ("_raw", "_refined"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    return stem


def step0_match_pairs(
    raw_dir: Path,
    refined_dir: Path,
    logger,
) -> List[Tuple[str, Path, Path]]:
    """
    İki klasörü tarayıp aynı kök adına sahip (RAW, REFINED) .nas çiftlerini eşleştirir.

    Desteklenen isimlendirmeler:
        • raw_dir/part_01_raw.nas     <-> refined_dir/part_01_refined.nas
        • raw_dir/part_01.nas         <-> refined_dir/part_01.nas

    Returns:
        [(stem, raw_path, refined_path), ...]   — artan stem sırasına göre
    """
    raw_map: Dict[str, Path]     = {_stem_key(p): p for p in sorted(raw_dir.glob("*.nas"))}
    refined_map: Dict[str, Path] = {_stem_key(p): p for p in sorted(refined_dir.glob("*.nas"))}

    common_stems  = sorted(set(raw_map) & set(refined_map))
    unmatched_raw = sorted(set(raw_map) - set(refined_map))
    unmatched_ref = sorted(set(refined_map) - set(raw_map))

    if unmatched_raw:
        logger.warning(
            f"Eşleşmeyen RAW dosyaları ({len(unmatched_raw)}): "
            f"{unmatched_raw[:5]}{'...' if len(unmatched_raw) > 5 else ''}"
        )
    if unmatched_ref:
        logger.warning(
            f"Eşleşmeyen REFINED dosyaları ({len(unmatched_ref)}): "
            f"{unmatched_ref[:5]}{'...' if len(unmatched_ref) > 5 else ''}"
        )

    pairs = [(s, raw_map[s], refined_map[s]) for s in common_stems]
    logger.info(
        f"ADIM 0 Tamamlandı | "
        f"RAW={len(raw_map)} | REFINED={len(refined_map)} | Eşleşen çift={len(pairs)}"
    )
    return pairs


# ====================================================================== #
#  Vocabulary Yükleyici                                                   #
# ====================================================================== #

def load_vocabulary(vocab_path: Path, logger) -> Dict[str, int]:
    """
    ai_vocabulary.json dosyasından  wl_hash -> token_id  haritası oluşturur.
    Yalnızca PATCH tipi tokenlar dahil edilir; SPECIAL tokenlar atlanır.

    Returns:
        {wl_hash_string: token_id_int}
    """
    with open(vocab_path, "r", encoding="utf-8") as f:
        vocab_data = json.load(f)

    hash_to_id: Dict[str, int] = {}
    for token_id_str, entry in vocab_data.items():
        if token_id_str == "metadata":
            continue
        if entry.get("type") == "PATCH":
            wl_hash = entry.get("wl_hash")
            if wl_hash:
                hash_to_id[wl_hash] = int(token_id_str)

    logger.info(
        f"Sözlük yüklendi | {len(hash_to_id)} PATCH token | kaynak: {vocab_path}"
    )
    return hash_to_id


# ====================================================================== #
#  ADIM 1: Hedef (Target) Dizisi  — Refined Mesh -> Token Sequence       #
# ====================================================================== #

def step1_build_target_sequence(
    refined_path: Path,
    hash_to_id: Dict[str, int],
    logger,
) -> Optional[List[int]]:
    """
    Refined mesh dosyasını  canonical order'a sokup  her düğümün WL-Hash'ini
    hesaplar ve sözlükteki Token ID'sine çevirir.

    İşlem akışı:
        1) .nas parse
        2) apply_canonical_order  (BFS sıralama + ID yeniden numaralandırma)
        3) PatchExtractor  ->  PatchAnalyzer.compute_all_hashes()
        4) Sıralı düğüm listesi  ->  token_id  (bilinmiyorsa TOKEN_UNK)
        5) [TOKEN_START, id_0, id_1, ..., id_N, TOKEN_END]

    Returns:
        Hedef token dizisi (int listesi) veya mesh boşsa None.
    """
    file_label = refined_path.stem

    # 1a. Mesh'i parse et
    parser = ParserNas()
    mesh_data = parser.parse_file(refined_path)
    nodes    = mesh_data["nodes"]
    elements = mesh_data["elements"]

    if not nodes or not elements:
        logger.warning(f"[{file_label}] Refined mesh boş, atlanıyor.")
        return None

    # 1b. Canonical order + re-indexing
    canonical     = apply_canonical_order(nodes, elements, use_adjacency=True, file_name=file_label)
    canon_nodes   = canonical["nodes"]    # {0: (x,y,z), 1: ..., ...}
    canon_elements = canonical["elements"]

    # 1c. Her düğümün WL-Hash'ini hesapla
    extractor = PatchExtractor(file_name=file_label)
    patches_dict, node_skipped = extractor.extract_all_patches(canon_nodes, canon_elements)

    analyzer = PatchAnalyzer(file_name=file_label)
    analyzer.compute_all_hashes(patches_dict)

    # 1d. Sıralı düğüm listesi üzerinden token ID'lerini topla (0, 1, 2, ...)
    unk_count  = 0
    token_ids: List[int] = []

    for node_id in sorted(canon_nodes.keys()):
        if node_id in patches_dict:
            wl_hash  = patches_dict[node_id].topology_hash
            token_id = hash_to_id.get(wl_hash, TOKEN_UNK)
            if token_id == TOKEN_UNK:
                unk_count += 1
        else:
            # Patch çıkarılamadı (sınır/izole düğüm vb.)
            token_id = TOKEN_UNK
            unk_count += 1
        token_ids.append(token_id)

    # 1e. <START> ve <END> ekle
    target_sequence = [TOKEN_START] + token_ids + [TOKEN_END]

    logger.info(
        f"[{file_label}] ADIM 1 | "
        f"Düğüm={len(canon_nodes)} | Token={len(token_ids)} | "
        f"UNK={unk_count} ({unk_count/max(len(token_ids),1)*100:.1f}%) | Atlanan={node_skipped}"
    )
    return target_sequence


# ====================================================================== #
#  Geometrik Kalite Metrikleri (yardımcı fonksiyonlar)                   #
# ====================================================================== #

def _aspect_ratio(pts: List[np.ndarray]) -> float:
    """Kenar uz. bazlı Aspect Ratio = max_edge / min_edge (1 = ideal)."""
    n = len(pts)
    edges = [float(np.linalg.norm(pts[(i + 1) % n] - pts[i])) for i in range(n)]
    min_e = min(edges)
    return float(max(edges) / min_e) if min_e > 1e-12 else 0.0


def _skewness_triangle(pts: List[np.ndarray]) -> float:
    """
    Açı tabanlı skewness için üçgen: 1 - min_angle/60°
    0 = eşkenar (ideal),  1 = dejenere.
    """
    angles: List[float] = []
    for i in range(3):
        v1 = pts[(i + 1) % 3] - pts[i]
        v2 = pts[(i + 2) % 3] - pts[i]
        angle = np.degrees(np.arctan2(
            float(np.linalg.norm(np.cross(v1, v2))),
            float(np.dot(v1, v2))
        ))
        angles.append(angle)
    return float(np.clip(1.0 - min(angles) / 60.0, 0.0, 1.0)) if angles else 1.0


def _skewness_quad(pts: List[np.ndarray]) -> float:
    """
    Açı tabanlı skewness için dörtgen: 1 - min_angle/90°
    0 = dikdörtgen (ideal),  1 = dejenere.
    """
    angles: List[float] = []
    for i in range(4):
        v1 = pts[(i - 1) % 4] - pts[i]
        v2 = pts[(i + 1) % 4] - pts[i]
        angle = np.degrees(np.arctan2(
            float(np.linalg.norm(np.cross(v1, v2))),
            float(np.dot(v1, v2))
        ))
        angles.append(angle)
    return float(np.clip(1.0 - min(angles) / 90.0, 0.0, 1.0)) if angles else 1.0


def _scaled_jacobian_triangle(pts: List[np.ndarray]) -> float:
    """
    Ölçeklendirilmiş Jacobian (üçgen):  2·Area / (a² + b² + c²)
    1 = eşkenar (ideal),  0 = dejenere.
    """
    ab, ac = pts[1] - pts[0], pts[2] - pts[0]
    area2  = float(np.linalg.norm(np.cross(ab, ac)))
    denom  = (
        float(np.dot(ab, ab))
        + float(np.dot(ac, ac))
        + float(np.dot(pts[2] - pts[1], pts[2] - pts[1]))
    )
    return float(np.clip(area2 / denom, 0.0, 1.0)) if denom > 1e-12 else 0.0


def _compute_element_metrics(
    elem_data: Dict[str, Any],
    nodes: Dict[int, Tuple[float, float, float]],
) -> Tuple[float, float, float]:
    """
    Bir element için  (aspect_ratio, skewness, jacobian)  döndürür.

    Tüm değerler [0, ∞) ∪ [0, 1] aralığında ve float olarak üretilir.
    """
    node_coords = [np.array(nodes[nid]) for nid in elem_data["nodes"] if nid in nodes]
    n = len(node_coords)
    if n < 2:
        return 1.0, 1.0, 0.0

    ar = _aspect_ratio(node_coords)

    e_type = elem_data.get("type", "tria")
    if e_type == "tria" and n >= 3:
        sk  = _skewness_triangle(node_coords[:3])
        jac = _scaled_jacobian_triangle(node_coords[:3])
    elif e_type == "quad" and n >= 4:
        sk   = _skewness_quad(node_coords[:4])
        # Quad jacobian: iki üçgene böl, ortalamasını al
        jac1 = _scaled_jacobian_triangle(node_coords[:3])
        jac2 = _scaled_jacobian_triangle([node_coords[0], node_coords[2], node_coords[3]])
        jac  = (jac1 + jac2) / 2.0
    else:
        sk, jac = 1.0, 0.0

    return float(ar), float(sk), float(jac)


# ====================================================================== #
#  ADIM 2: Girdi Grafı  — Raw Mesh -> PyG Data                           #
# ====================================================================== #

def step2_build_input_graph(raw_path: Path, logger) -> Optional[Data]:
    """
    Ham mesh dosyasını oku ve GNN'in tüketeceği PyG Data nesnesini oluştur.

    Düğüm Özellik Vektörü  data.x  [num_nodes, 4]:
        [0] valence       – bağlı element sayısı (normalize edilmemiş)
        [1] aspect_ratio  – bağlı elementlerin ortalama AR'ı
        [2] skewness      – bağlı elementlerin ortalama skewness'ı
        [3] jacobian      – bağlı elementlerin ortalama scaled jacobian'ı

    data.pos        [num_nodes, 3]  – 3D koordinatlar (x, y, z)
    data.edge_index [2, num_edges]  – çift yönlü düğüm-düğüm komşuluğu

    Returns:
        torch_geometric.data.Data veya mesh boşsa None.
    """
    file_label = raw_path.stem

    # 2a. Mesh parse
    parser    = ParserNas()
    mesh_data = parser.parse_file(raw_path)
    nodes     = mesh_data["nodes"]
    elements  = mesh_data["elements"]

    if not nodes or not elements:
        logger.warning(f"[{file_label}] Raw mesh boş, atlanıyor.")
        return None

    # 2b. Düğüm -> bağlı elementler ters haritası
    node_to_elems: Dict[int, List[int]] = {nid: [] for nid in nodes}
    for eid, edata in elements.items():
        for nid in edata["nodes"]:
            if nid in node_to_elems:
                node_to_elems[nid].append(eid)

    # 2c. Element kalite metriklerini önceden hesapla (tekrar hesaplamayı önler)
    elem_metrics: Dict[int, Tuple[float, float, float]] = {
        eid: _compute_element_metrics(edata, nodes)
        for eid, edata in elements.items()
    }

    # 2d. Sıralı düğüm indeksi — deterministik çıktı için
    node_ids_sorted = sorted(nodes.keys())
    nid_to_idx      = {nid: idx for idx, nid in enumerate(node_ids_sorted)}

    pos_list:  List[List[float]] = []
    feat_list: List[List[float]] = []

    for nid in node_ids_sorted:
        x, y, z = nodes[nid]
        pos_list.append([x, y, z])

        connected = node_to_elems[nid]
        valence   = len(connected)

        if connected:
            mean_ar  = float(np.mean([elem_metrics[e][0] for e in connected]))
            mean_sk  = float(np.mean([elem_metrics[e][1] for e in connected]))
            mean_jac = float(np.mean([elem_metrics[e][2] for e in connected]))
        else:
            mean_ar = mean_sk = 1.0
            mean_jac = 0.0

        feat_list.append([float(valence), mean_ar, mean_sk, mean_jac])

    # 2e. Edge index — element kenarlarından çift yönlü kenarlar
    edge_set: Set[Tuple[int, int]] = set()
    for edata in elements.values():
        enids = edata["nodes"]
        n     = len(enids)
        for i in range(n):
            src = nid_to_idx.get(enids[i])
            dst = nid_to_idx.get(enids[(i + 1) % n])
            if src is not None and dst is not None and src != dst:
                edge_set.add((src, dst))
                edge_set.add((dst, src))

    if edge_set:
        src_list, dst_list = zip(*sorted(edge_set))
        edge_index = torch.tensor([list(src_list), list(dst_list)], dtype=torch.long)
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)

    # 2f. Data nesnesi
    data = Data(
        x          = torch.tensor(feat_list, dtype=torch.float),
        edge_index = edge_index,
        pos        = torch.tensor(pos_list,  dtype=torch.float),
    )

    logger.info(
        f"[{file_label}] ADIM 2 | "
        f"Düğüm={len(node_ids_sorted)} | Kenar={edge_index.shape[1]} | "
        f"x.shape={list(data.x.shape)}"
    )
    return data


# ====================================================================== #
#  ADIM 3: Birleştirme ve Kayıt                                          #
# ====================================================================== #

def step3_save_training_pair(
    stem: str,
    graph_data: Data,
    target_sequence: List[int],
    output_dir: Path,
    logger,
) -> Path:
    """
    Hedef dizi tensörünü  data.y  olarak ekler ve nesneyi diske yazar.

    Dosya formatı: torch.save(Data, path)  — .pt uzantısı

    Returns:
        Kaydedilen dosyanın tam yolu.
    """
    graph_data.y = torch.tensor(target_sequence, dtype=torch.long)

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{stem}.pt"
    torch.save(graph_data, out_path)

    logger.info(
        f"[{stem}] ADIM 3 | Kaydedildi: {out_path.name} | "
        f"x={list(graph_data.x.shape)} | "
        f"edge_index={list(graph_data.edge_index.shape)} | "
        f"y uzunluk={len(target_sequence)}"
    )
    return out_path


# ====================================================================== #
#  Konsol Raporu                                                          #
# ====================================================================== #

def print_summary(
    total_pairs: int,
    processed:   int,
    skipped:     int,
    output_dir:  Path,
    elapsed_sec: float,
) -> None:
    sep = "=" * 64
    print(f"\n{sep}")
    print("  FAZ 2 – Pipeline Raporu")
    print(sep)
    print(f"  Toplam eşleşen çift          : {total_pairs}")
    print(f"  Başarıyla işlenen            : {processed}")
    print(f"  Atlanan (hata / boş mesh)    : {skipped}")
    print(f"  Oluşturulan .pt dosyası      : {processed}")
    print(f"  Çıktı klasörü               : {output_dir}")
    print(f"  Toplam süre                  : {elapsed_sec:.1f} sn")
    print(f"{sep}\n")


# ====================================================================== #
#  CLI & Main                                                             #
# ====================================================================== #

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="FAZ 2: Veri Serileştirme ve Graf Dönüşümü",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--raw-dir",
        type=Path,
        default=None,
        help="Normalize edilmiş RAW .nas klasörü (varsayılan: config.yaml → raw_output_dir)",
    )
    p.add_argument(
        "--refined-dir",
        type=Path,
        default=None,
        help="Normalize edilmiş REFINED .nas klasörü (varsayılan: config.yaml → refined_output_dir)",
    )
    p.add_argument(
        "--vocab",
        type=Path,
        default=DEFAULT_VOCAB_PATH,
        help="ai_vocabulary.json yolu",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Çıktı .pt klasörü (varsayılan: config.yaml → torch_path_dir)",
    )
    p.add_argument(
        "--norm-csv",
        type=Path,
        default=DEFAULT_NORM_CSV,
        help="normalize_parameter.csv yolu. Sağlanırsa her .pt dosyasına "
             "norm_scale ve norm_center alanları eklenir.",
    )
    return p.parse_args()


def main() -> None:
    t0     = time.time()
    args   = _parse_args()
    logger = get_context_logger(name="faz2_pipeline", file_name="pipeline")

    # ── Yolları belirle: CLI argümanı > config.yaml > sabit varsayılan ──
    config = load_config()

    raw_dir     = (args.raw_dir     or Path(config["raw_output_dir"])).resolve()
    refined_dir = (args.refined_dir or Path(config["refined_output_dir"])).resolve()
    vocab_path  = args.vocab.resolve()

    # output-dir için önce CLI, sonra config'deki torch_path_dir, son çare DEFAULT_OUTPUT_DIR
    if args.output_dir is not None:
        output_dir = args.output_dir.resolve()
    elif "torch_path_dir" in config:
        output_dir = Path(config["torch_path_dir"]).resolve()
    else:
        output_dir = DEFAULT_OUTPUT_DIR.resolve()

    logger.info(f"RAW klasör     : {raw_dir}")
    logger.info(f"REFINED klasör : {refined_dir}")
    logger.info(f"Çıktı klasörü  : {output_dir}")
    logger.info(f"Sözlük         : {vocab_path}")

    # ── Normalizasyon parametrelerini yükle (opsiyonel) ───────────────────
    norm_params: Dict[str, Dict[str, float]] = {}
    norm_csv_path = args.norm_csv
    if norm_csv_path is not None and norm_csv_path.is_file():
        with open(norm_csv_path, newline="", encoding="utf-8") as _f:
            for row in csv.DictReader(_f):
                stem_key = Path(row["file_name"]).stem
                norm_params[stem_key] = {
                    "center_x":         float(row["center_x"]),
                    "center_y":         float(row["center_y"]),
                    "center_z":         float(row["center_z"]),
                    "scale_max_radius": float(row["scale_max_radius"]),
                }
        logger.info(f"Norm CSV yüklendi: {len(norm_params)} kayıt — {norm_csv_path}")
    else:
        logger.warning(
            f"Norm CSV bulunamadı: {norm_csv_path}  "
            f"→ .pt dosyalarına norm_scale/norm_center eklenmeyecek."
        )

    # ── Ön kontroller ────────────────────────────────────────────────────
    errors = False
    for label, path in [("RAW", raw_dir), ("REFINED", refined_dir)]:
        if not path.is_dir():
            logger.error(f"{label} klasörü bulunamadı: {path}")
            errors = True
    if not vocab_path.is_file():
        logger.error(f"Sözlük dosyası bulunamadı: {vocab_path}")
        errors = True
    if errors:
        sys.exit(1)

    # ── Vocabulary yükle ─────────────────────────────────────────────────
    hash_to_id = load_vocabulary(vocab_path, logger)

    # ── ADIM 0: Dosya eşleştirme ─────────────────────────────────────────
    pairs = step0_match_pairs(raw_dir, refined_dir, logger)
    if not pairs:
        logger.error("Eşleşen dosya çifti bulunamadı. Pipeline sonlandırılıyor.")
        sys.exit(1)

    # ── Ana döngü ────────────────────────────────────────────────────────
    processed = 0
    skipped   = 0

    for pair_idx, (stem, raw_path, refined_path) in enumerate(pairs, start=1):
        logger.info(f"─── [{pair_idx}/{len(pairs)}] İşleniyor: {stem} ───")

        # ADIM 1: Hedef dizisi (Refined -> Token ID sequence)
        target_seq = step1_build_target_sequence(refined_path, hash_to_id, logger)
        if target_seq is None:
            logger.warning(f"[{stem}] Hedef dizi oluşturulamadı, atlanıyor.")
            skipped += 1
            continue

        # ADIM 2: Girdi grafı (Raw -> PyG Data)
        graph_data = step2_build_input_graph(raw_path, logger)
        if graph_data is None:
            logger.warning(f"[{stem}] Girdi grafı oluşturulamadı, atlanıyor.")
            skipped += 1
            continue

        # ── Normalizasyon parametrelerini Data nesnesine göm ─────────────
        params = norm_params.get(stem)
        if params is not None:
            graph_data.norm_scale = torch.tensor(
                [params["scale_max_radius"]], dtype=torch.float
            )
            graph_data.norm_center = torch.tensor(
                [params["center_x"], params["center_y"], params["center_z"]],
                dtype=torch.float,
            )
            logger.debug(
                f"[{stem}] norm_scale={params['scale_max_radius']:.4f} | "
                f"norm_center=({params['center_x']:.2f}, "
                f"{params['center_y']:.2f}, {params['center_z']:.2f})"
            )
        else:
            logger.warning(f"[{stem}] Norm CSV'de kayıt yok; norm_scale/norm_center eklenmedi.")

        # ADIM 3: Birleştir ve kaydet
        step3_save_training_pair(stem, graph_data, target_seq, output_dir, logger)
        processed += 1

    # ── Özet ─────────────────────────────────────────────────────────────
    elapsed = time.time() - t0
    logger.info(
        f"Pipeline tamamlandı | "
        f"İşlenen={processed} | Atlanan={skipped} | Süre={elapsed:.1f}s"
    )
    print_summary(
        total_pairs = len(pairs),
        processed   = processed,
        skipped     = skipped,
        output_dir  = output_dir,
        elapsed_sec = elapsed,
    )


if __name__ == "__main__":
    main()
