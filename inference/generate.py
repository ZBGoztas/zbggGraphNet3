"""
scripts/generate.py  –  FAZ 5: Çıkarım (Inference) ve Yeniden İnşa
===================================================================
Eğitilmiş Graph2Seq modelini kullanarak RAW mesh grafiğinden yeni bir
Refined mesh üretir ve 3D uzayda birleştirir.

Pipeline Özeti:
    1. RAW .pt dosyası  →  PyG Data (edge_index, x, pos)
    2. GNN Encoder      →  Latent Context [1, hidden_dim]
    3. Greedy Decode (repetition_penalty)  →  Token ID Dizisi
    4. Token → Yama Geometrisi
         – Her yama, normalize uzaydaki RAW mesh düğümüne hizalanır
         – world = (local_normalized + raw_node_pos) * scale + center
    5. NAS Dışa Aktarma (Nastran BDF free-field)
    6. Üçlü PyVista Görselleştirme (RAW | REFINED | PREDICTION)

Düzeltilen Sorunlar (v2):
    – Greedy collapse: tekrar cezası (repetition_penalty) eklendi
    – Patch placement: her yama ilgili RAW mesh düğümüne konumlandırılır
    – Görselleştirme: boş mesh uyarısı ve sınır bilgisi eklendi

Kullanım:
    python scripts/generate.py
    python scripts/generate.py --pt-file /path/to/mesh.pt --output output.nas
    python scripts/generate.py --repetition-penalty 1.5 --temperature 0.9
"""

from __future__ import annotations
from pathlib import Path as _P
_REPO_ROOT = _P(__file__).resolve().parents[1]
_WORKSPACE = _REPO_ROOT.parent  # expected layout: <workspace>/DATA and <workspace>/<this repo>

import argparse
import csv
import json
import logging
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from torch_geometric.data import Data

# ── Proje kökünü sys.path'e ekle ─────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from models import HourGlassGNNEncoder, AutoregressiveMeshDecoder
from training.train import Config, Graph2Seq
from utils.parser_utils import ParserNas

try:
    import pyvista as pv
    from utils.visualization import show_triple_comparison
    _PYVISTA_AVAILABLE = True
except ImportError:
    _PYVISTA_AVAILABLE = False
    show_triple_comparison = None  # type: ignore[assignment]

try:
    import trimesh
    import trimesh.proximity
    _TRIMESH_AVAILABLE = True
except ImportError:
    _TRIMESH_AVAILABLE = False

from scipy.spatial import cKDTree


# =========================================================================== #
#  Loglama                                                                      #
# =========================================================================== #

def _build_logger(name: str = "generate") -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.DEBUG)
    fmt = logging.Formatter("%(asctime)s  [%(levelname)s]  %(message)s",
                            datefmt="%H:%M:%S")
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(ch)
    return logger


_log = _build_logger()


# =========================================================================== #
#  Sabitler                                                                     #
# =========================================================================== #

TOKEN_PAD   = 0
TOKEN_START = 1
TOKEN_END   = 2
TOKEN_UNK   = 3

SPECIAL_TOKENS = {TOKEN_PAD, TOKEN_START, TOKEN_END, TOKEN_UNK}

DEFAULT_MAX_GEN_LEN        = 2000
DEFAULT_EPSILON            = 1e-4
DEFAULT_REPETITION_PENALTY = 1.5   # >1.0 → geçmiş tokenları cezalandır
DEFAULT_TEMPERATURE        = 1.0   # <1.0 → keskin, >1.0 → rastgele


# =========================================================================== #
#  Normalizasyon Yardımcıları                                                   #
# =========================================================================== #

def load_normalization_params(csv_path: str) -> Dict[str, Dict[str, float]]:
    """normalize_parameter.csv dosyasını yükler: stem → {center_x/y/z, scale_max_radius}"""
    params: Dict[str, Dict[str, float]] = {}
    csv_file = Path(csv_path)
    if not csv_file.is_file():
        _log.warning(f"Normalizasyon CSV bulunamadı: {csv_path}  (denormalizasyon devre dışı)")
        return params
    with open(csv_file, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            stem = Path(row["file_name"]).stem
            params[stem] = {
                "center_x":         float(row["center_x"]),
                "center_y":         float(row["center_y"]),
                "center_z":         float(row["center_z"]),
                "scale_max_radius": float(row["scale_max_radius"]),
            }
    _log.info(f"Normalizasyon parametreleri yüklendi: {len(params)} kayıt")
    return params


# =========================================================================== #
#  SINIF 1: MeshGenerator                                                       #
# =========================================================================== #

class MeshGenerator:
    """
    Eğitilmiş Graph2Seq modelini kullanarak RAW mesh grafiğinden token dizisi üretir.

    v2 İyileştirmeleri
    ------------------
    repetition_penalty
        Üretilmiş tokenlara logit cezası uygular → greedy collapse önlenir.
        Uygulama kuralı:
            logit > 0  →  logit /= penalty
            logit <= 0 →  logit *= penalty

    temperature
        Logitleri ölçekler (argmax öncesi): düşük → keskin, yüksek → çeşitli.
    """

    def __init__(
        self,
        checkpoint_path: str,
        vocab_path: str,
        norm_csv_path: Optional[str] = None,
        device: Optional[str] = None,
    ) -> None:
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        _log.info(f"[MeshGenerator] Cihaz: {self.device}")

        ckpt = self._load_checkpoint(checkpoint_path)
        cfg: Config = ckpt["config"]
        self.config = cfg
        self.model: Graph2Seq = Graph2Seq(cfg).to(self.device)
        self.model.load_state_dict(ckpt["model_state"])
        self.model.eval()
        _log.info(
            f"[MeshGenerator] Model yüklendi "
            f"(epoch={ckpt.get('epoch', '?')}, "
            f"val_loss={ckpt.get('val_loss', float('nan')):.6f})"
        )

        self.vocabulary: Dict = self._load_vocabulary(vocab_path)
        _log.info(f"[MeshGenerator] Sözlük: {len(self.vocabulary)} giriş")

        self.norm_params: Dict[str, Dict[str, float]] = (
            load_normalization_params(norm_csv_path)
            if norm_csv_path else {}
        )

    # ── Özel yardımcılar ────────────────────────────────────────────────────

    @staticmethod
    def _load_checkpoint(path: str) -> dict:
        ckpt_path = Path(path)
        if not ckpt_path.is_file():
            raise FileNotFoundError(f"Checkpoint bulunamadı: {path}")
        _log.info(f"[MeshGenerator] Checkpoint yükleniyor: {path}")
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        missing = {"model_state", "config"} - set(ckpt.keys())
        if missing:
            raise KeyError(f"Checkpoint'te eksik anahtarlar: {missing}")
        return ckpt

    @staticmethod
    def _load_vocabulary(path: str) -> Dict:
        vocab_file = Path(path)
        if not vocab_file.is_file():
            raise FileNotFoundError(f"Sözlük bulunamadı: {path}")
        with open(vocab_file, encoding="utf-8") as f:
            return json.load(f)

    # ── Token Dizisi Üretimi ─────────────────────────────────────────────────

    @torch.no_grad()
    def generate_sequence(
        self,
        raw_mesh_data: Data,
        max_len: int = DEFAULT_MAX_GEN_LEN,
        repetition_penalty: float = DEFAULT_REPETITION_PENALTY,
        temperature: float = DEFAULT_TEMPERATURE,
    ) -> List[int]:
        """
        Otoregressif greedy decoding — repetition_penalty ile greedy collapse önlenir.

        Her adımda üretilmiş tokenların logitleri ceza katsayısıyla baskılanır;
        bu sayede model farklı yama tiplerini keşfetmeye zorlanır.
        """
        _log.info("[generate_sequence] Başlatılıyor …")
        t_start = time.time()

        data = raw_mesh_data.to(self.device)
        batch_vector = torch.zeros(data.x.size(0), dtype=torch.long, device=self.device)

        # GNN Encoder → Latent Context [1, hidden_dim]
        context: torch.Tensor = self.model.encoder(
            data.x,
            data.edge_index,
            batch_vector,
        )
        _log.info(f"[generate_sequence] Latent context: {list(context.shape)}")

        # Greedy Decoding — <START> ile başla
        generated = torch.tensor([[TOKEN_START]], dtype=torch.long, device=self.device)
        step = 0

        while step < max_len:
            logits = self.model.decoder(
                tgt_tokens=generated,
                memory=context,
            )  # [1, cur_len, vocab_size]

            next_logits = logits[0, -1, :].clone()  # [vocab_size]

            # Repetition Penalty: geçmiş tokenların logitlerini baskıla
            if repetition_penalty != 1.0:
                for prev_id in set(generated[0].tolist()):
                    if 0 <= prev_id < next_logits.size(0):
                        if next_logits[prev_id] > 0:
                            next_logits[prev_id] /= repetition_penalty
                        else:
                            next_logits[prev_id] *= repetition_penalty

            # Temperature: logit ölçekleme
            if temperature != 1.0 and temperature > 0:
                next_logits = next_logits / temperature

            next_token_id: int = int(next_logits.argmax().item())
            generated = torch.cat(
                [generated, torch.tensor([[next_token_id]], dtype=torch.long, device=self.device)],
                dim=1,
            )
            step += 1

            if next_token_id == TOKEN_END:
                _log.info(f"[generate_sequence] <END> alındı. Adım: {step}")
                break

        if step >= max_len:
            _log.warning(f"[generate_sequence] max_len={max_len} sınırına ulaşıldı.")

        token_list: List[int] = generated[0].cpu().tolist()
        elapsed = time.time() - t_start

        # Token dağılım özeti
        patch_tokens = [t for t in token_list if t not in SPECIAL_TOKENS]
        counter = Counter(patch_tokens)
        _log.info(
            f"[generate_sequence] Tamamlandı | toplam={len(token_list)} | "
            f"patch={len(patch_tokens)} | benzersiz={len(counter)} | "
            f"en sık={counter.most_common(5)} | süre={elapsed:.2f}s"
        )
        return token_list


# =========================================================================== #
#  SINIF 2: MeshReconstructor                                                   #
# =========================================================================== #

class MeshReconstructor:
    """
    Token ID dizisini 3D mesh'e dönüştürür ve NAS dosyasına yazar.

    v2 Patch Placement (Yama Konumlandırma)
    ----------------------------------------
    Önceki sürümde tüm yamalar tek global merkeze konuluyordu; bu nedenle
    greedy collapse durumunda binlerce yama üst üste yığılarak PyVista'da
    siyah görünüyordu.

    v2'de her yama, sırasına göre karşılık gelen RAW mesh düğümüne hizalanır:

        world = (local_normalized + raw_node_normalized) * scale + center

    Bu sayede yamalar tüm mesh yüzeyi üzerine dağıtılır.
    """

    def __init__(
        self,
        vocabulary: Dict,
        norm_params: Optional[Dict[str, Dict[str, float]]] = None,
        epsilon: float = DEFAULT_EPSILON,
    ) -> None:
        self.vocabulary  = vocabulary
        self.norm_params = norm_params or {}

    # ── Yüzey Normal Hizalama Yardımcıları ──────────────────────────────────

    @staticmethod
    def _rotation_matrix_from_vectors(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
        """
        Rodrigues formülü: src birim vektörünü dst birim vektörüne döndüren 3×3 matris.

        Özel durumlar:
            • Paralel (dot ≈ +1)  → Birim matris (döndürme yok).
            • Ters paralel (dot ≈-1) → Dik bir eksen etrafında 180° rotasyon.
        """
        src = src / (np.linalg.norm(src) + 1e-12)
        dst = dst / (np.linalg.norm(dst) + 1e-12)
        v = np.cross(src, dst)
        c = float(np.dot(src, dst))
        s = np.linalg.norm(v)

        if s < 1e-10:
            if c > 0.0:
                return np.eye(3, dtype=np.float64)
            # 180° ters yönde: src'ye dik bir eksen etrafında döndür
            perp = np.array([1.0, 0.0, 0.0]) if abs(src[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
            axis = np.cross(src, perp)
            axis /= np.linalg.norm(axis)
            kmat = np.array([[0.0, -axis[2], axis[1]],
                              [axis[2], 0.0, -axis[0]],
                              [-axis[1], axis[0], 0.0]])
            return np.eye(3, dtype=np.float64) + 2.0 * kmat @ kmat

        kmat = np.array([[0.0, -v[2], v[1]],
                         [v[2], 0.0, -v[0]],
                         [-v[1], v[0], 0.0]])
        return np.eye(3, dtype=np.float64) + kmat + kmat @ kmat * ((1.0 - c) / (s ** 2))

    @staticmethod
    def _build_surface_normals(
        raw_pos: np.ndarray,
        raw_mesh_data: Optional[Data] = None,
    ) -> np.ndarray:
        """
        raw_pos'taki her düğüm için yüzey normal vektörü döndürür (birim uzunluk).

        Öncelik Sırası
        --------------
        1. raw_mesh_data.face mevcutsa  →  trimesh yüzey normallerini kullanır
           (trimesh.proximity.closest_point ile her düğümün en yakın üçgeninin normali).
        2. Trimesh yoksa veya face verisi eksikse  →  KDTree + PCA tabanlı tahmin:
           Her nokta için k en yakın komşudan kovaryans matrisi oluşturulur;
           en küçük öz vektör (minimum varyans yönü) normal olarak seçilir.
        """
        n = len(raw_pos)

        # --- Yol 1: trimesh yüzey normalleri -----------------------------------
        if (_TRIMESH_AVAILABLE
                and raw_mesh_data is not None
                and hasattr(raw_mesh_data, "face")
                and raw_mesh_data.face is not None):
            try:
                face_np = raw_mesh_data.face.cpu().numpy().T  # [n_faces, 3]
                tm = trimesh.Trimesh(vertices=raw_pos, faces=face_np, process=False)
                tm.fix_normals()
                _, _, tri_ids = trimesh.proximity.closest_point(tm, raw_pos)
                normals = tm.face_normals[tri_ids].astype(np.float64)
                norms = np.linalg.norm(normals, axis=1, keepdims=True)
                norms = np.where(norms < 1e-12, 1.0, norms)
                _log.info("[surface_normals] trimesh yüzey normalleri hesaplandı.")
                return normals / norms
            except Exception as exc:
                _log.debug(f"[surface_normals] trimesh başarısız ({exc}), PCA'ya geçiliyor.")

        # --- Yol 2: KDTree + PCA normal tahmini --------------------------------
        _log.info(f"[surface_normals] KDTree+PCA ile {n} düğüm için normal tahmin ediliyor …")
        k = min(15, n)
        tree = cKDTree(raw_pos)
        _, all_idx = tree.query(raw_pos, k=k)  # [n, k]
        normals = np.zeros((n, 3), dtype=np.float64)
        for i in range(n):
            nbrs = raw_pos[all_idx[i]]
            centered = nbrs - nbrs.mean(axis=0)
            cov = centered.T @ centered
            _, eigvecs = np.linalg.eigh(cov)
            normals[i] = eigvecs[:, 0]  # en küçük öz değer → normal yönü

        norms = np.linalg.norm(normals, axis=1, keepdims=True)
        norms = np.where(norms < 1e-12, 1.0, norms)
        return normals / norms

    # ── Token → Yama Geometrisi ──────────────────────────────────────────────

    def _tokens_to_patches(
        self,
        token_ids: List[int],
        file_stem: str = "",
        raw_mesh_data: Optional[Data] = None,
    ) -> Tuple[np.ndarray, List[List[int]]]:
        """
        Her PATCH tokeni için vocabulary'den yerel koordinatları alır ve
        ilgili RAW mesh düğümüne hizalayarak NORMALİZE uzayda döndürür.

        Döndürülen koordinatlar NORMALİZE uzaydadır ([-1, 1] civarı).
        NAS dışa aktarma için denormalizasyon reconstruct() içinde yapılır.

        Konumlandırma:
            RAW mesh pos mevcut  → norm_coords = local_coords + raw_anchor
            RAW mesh pos yok     → norm_coords = local_coords  (origin çevresinde)
        """
        # RAW mesh normalize koordinatları (anchor olarak kullanılır)
        raw_pos: Optional[np.ndarray] = None
        if (raw_mesh_data is not None
                and hasattr(raw_mesh_data, "pos")
                and raw_mesh_data.pos is not None):
            raw_pos = raw_mesh_data.pos.cpu().numpy().astype(np.float64)
            _log.info(
                f"[tokens_to_patches] RAW mesh anchor: {raw_pos.shape[0]} düğüm"
            )

        # Yüzey normalleri: her RAW düğümü için hedef normal vektörü
        raw_normals: Optional[np.ndarray] = None
        if raw_pos is not None and len(raw_pos) > 0:
            raw_normals = self._build_surface_normals(raw_pos, raw_mesh_data)

        # Yamanın varsayılan normali  (sözlükteki yerel geometri Z-eksenine yaslanır)
        _DEFAULT_PATCH_NORMAL = np.array([0.0, 0.0, 1.0], dtype=np.float64)

        # Sadece PATCH tokenları
        patch_token_ids = [t for t in token_ids if t not in SPECIAL_TOKENS]
        n_patches = len(patch_token_ids)
        n_raw = raw_pos.shape[0] if raw_pos is not None else 0

        all_verts: List[np.ndarray] = []
        all_faces: List[List[int]] = []
        global_vertex_offset = 0
        patch_count = 0
        skipped = 0

        for patch_idx, token_id in enumerate(patch_token_ids):
            entry = self.vocabulary.get(str(token_id))
            if entry is None or entry.get("type") != "PATCH":
                skipped += 1
                continue

            local_nodes_raw: Dict[str, List[float]] = entry.get("local_nodes", {})
            if not local_nodes_raw:
                skipped += 1
                continue

            # Yerel koordinatlar (normalize uzayda, küçük ofsetler)
            sorted_keys = sorted(local_nodes_raw.keys(), key=lambda k: int(k))
            local_coords = np.array(
                [local_nodes_raw[k] for k in sorted_keys], dtype=np.float64
            )  # [patch_nodes, 3]

            # Yüzey listesi (normal hesabı için döngü öncesinde lazım)
            local_elements_raw: Dict[str, Dict] = entry.get("local_elements", {})

            # Anchor belirleme: yamayı RAW mesh üzerindeki ilgili bölgeye hizala
            # Tüm koordinatlar NORMALİZE uzayda kalır ([-1,1] civarı)
            if raw_pos is not None and n_raw > 0 and n_patches > 0:
                raw_idx = min(int(patch_idx * n_raw / n_patches), n_raw - 1)
                anchor = raw_pos[raw_idx]

                # ── Yüzey Normaline Göre Döndürme (Rotate → Translate) ────────
                # 1) Yamanın kendi normalini yerel geometrisinden hesapla
                patch_normal = _DEFAULT_PATCH_NORMAL.copy()
                if local_elements_raw:
                    first_elem = next(iter(local_elements_raw.values()))
                    lni_first: List[int] = first_elem.get("nodes", [])
                    if len(lni_first) >= 3:
                        try:
                            v0 = local_coords[lni_first[0]]
                            v1 = local_coords[lni_first[1]]
                            v2 = local_coords[lni_first[2]]
                            pn = np.cross(v1 - v0, v2 - v0)
                            pn_len = np.linalg.norm(pn)
                            if pn_len > 1e-10:
                                patch_normal = pn / pn_len
                        except IndexError:
                            pass  # local_coords yeterince büyük değilse varsayılanı kullan

                # 2) Hedef normal: RAW mesh yüzeyinin bu noktadaki normali
                if raw_normals is not None:
                    target_normal = raw_normals[raw_idx]
                    tn_len = np.linalg.norm(target_normal)
                    if tn_len > 1e-10:
                        target_normal = target_normal / tn_len
                        # 3) Rotasyon matrisini hesapla (Rodrigues)
                        rot_mat = self._rotation_matrix_from_vectors(patch_normal, target_normal)
                        # 4) Yamayı kendi merkezine göre döndür, sonra anchor'a ötele
                        patch_center = local_coords.mean(axis=0)
                        centered = local_coords - patch_center
                        rotated = (rot_mat @ centered.T).T
                        norm_coords = rotated + anchor
                    else:
                        norm_coords = local_coords + anchor
                else:
                    norm_coords = local_coords + anchor
                # ─────────────────────────────────────────────────────────────

            else:
                # Fallback: orijin çevresinde (local koordinatlar olduğu gibi)
                norm_coords = local_coords.copy()

            # Yüzey listesi
            for elem_data in local_elements_raw.values():
                lni: List[int] = elem_data.get("nodes", [])
                if lni:
                    all_faces.append([global_vertex_offset + ni for ni in lni])

            all_verts.append(norm_coords)
            global_vertex_offset += len(norm_coords)
            patch_count += 1

        if skipped:
            _log.debug(f"[tokens_to_patches] {skipped} token atlandı (PATCH değil).")

        if not all_verts:
            _log.warning("Hiçbir PATCH token geçerli geometri üretmedi!")
            return np.empty((0, 3), dtype=np.float64), []

        final_verts = np.vstack(all_verts)
        _log.info(
            f"[tokens_to_patches] {patch_count} yama işlendi – "
            f"köşe={len(final_verts)}, yüzey={len(all_faces)}"
        )
        return final_verts, all_faces

    # ── NAS Dışa Aktarma ─────────────────────────────────────────────────────

    @staticmethod
    def _export_nas(
        vertices: np.ndarray,
        faces: List[List[int]],
        output_path: str,
    ) -> None:
        """Mesh'i Nastran BDF free-field formatında diske yazar."""
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            f.write("$ MeshCrafter zbggGraphNet3 – FAZ 5 Çıktısı\n")
            f.write(f"$ Düğüm sayısı: {len(vertices)}\n")
            f.write(f"$ Yüzey sayısı: {len(faces)}\n")
            f.write("BEGIN BULK\n")
            for i, (vx, vy, vz) in enumerate(vertices, start=1):
                f.write(f"GRID,{i},,{vx:.8f},{vy:.8f},{vz:.8f}\n")
            eid = 1
            for face in faces:
                n = len(face)
                if n < 3:
                    continue
                if n == 3:
                    n1, n2, n3 = (vi + 1 for vi in face)
                    f.write(f"CTRIA3,{eid},1,{n1},{n2},{n3}\n")
                    eid += 1
                elif n == 4:
                    n1, n2, n3, n4 = (vi + 1 for vi in face)
                    f.write(f"CQUAD4,{eid},1,{n1},{n2},{n3},{n4}\n")
                    eid += 1
                else:
                    for i in range(1, n - 1):
                        f.write(
                            f"CTRIA3,{eid},1,"
                            f"{face[0]+1},{face[i]+1},{face[i+1]+1}\n"
                        )
                        eid += 1
            f.write("ENDDATA\n")
        _log.info(f"[export_nas] Dosya kaydedildi: {out.resolve()}")

    # ── Ana Yeniden İnşa Metodu ──────────────────────────────────────────────

    def reconstruct(
        self,
        token_ids: List[int],
        raw_mesh_data: Data,
        output_path: str,
        file_stem: str = "generated_mesh",
        smooth: bool = True,
    ) -> Tuple[np.ndarray, List[List[int]]]:
        """
        Token ID dizisini 3D mesh'e çevirir ve NAS dosyasına kaydeder.

        _tokens_to_patches() NORMALİZE koordinatlar döndürür (görselleştirme için).
        NAS dışa aktarma için koordinatlar bu metot içinde dünya uzayına çevrilir.

        Dönüş değerleri vizüalizasyon için normalize uzaydadır:
            viz_verts ≈ [-1, 1]  (raw_data.pos ve REFINED_NAS ile aynı uzay)
        """
        _log.info("[reconstruct] Token → Yama Geometrisi (normalize uzay, RAW mesh anchor)")
        norm_verts, final_faces = self._tokens_to_patches(
            token_ids, file_stem, raw_mesh_data
        )
        if len(norm_verts) == 0:
            _log.error("[reconstruct] Geçerli yama geometrisi üretilemedi.")
            return norm_verts, final_faces

        _log.info(
            f"[reconstruct] Toplam: köşe={len(norm_verts)}, yüzey={len(final_faces)}"
        )

        # NAS için denormalize et: norm → dünya koordinatları
        nas_verts = norm_verts.copy()
        if self.norm_params and file_stem:
            params = self.norm_params.get(file_stem)
            if params is not None:
                center = np.array(
                    [params["center_x"], params["center_y"], params["center_z"]],
                    dtype=np.float64,
                )
                scale = float(params["scale_max_radius"])
                nas_verts = norm_verts * scale + center
                _log.info(
                    f"[reconstruct] NAS denormalizasyon: "
                    f"center={center.tolist()}, scale={scale:.4f}"
                )
            else:
                _log.warning(
                    f"[reconstruct] '{file_stem}' için norm parametresi bulunamadı. "
                    "NAS normalize uzayda kaydedilecek."
                )

        _log.info("[reconstruct] NAS Dışa Aktarma")
        self._export_nas(nas_verts, final_faces, output_path)

        # Görselleştirme için normalize uzay koordinatlarını döndür
        return norm_verts, final_faces


# =========================================================================== #
#  PyVista Yardımcıları                                                         #
# =========================================================================== #

def _build_pyvista_from_arrays(
    vertices: np.ndarray,
    faces: List[List[int]],
) -> "pv.PolyData":
    """(vertices, faces) çiftini PyVista PolyData'ya çevirir."""
    if not _PYVISTA_AVAILABLE:
        raise ImportError("PyVista kurulu değil: pip install pyvista")
    if len(vertices) == 0:
        return pv.PolyData()
    face_arr: List[int] = []
    for face in faces:
        n = len(face)
        if n >= 3:
            face_arr.extend([n] + face)
    if not face_arr:
        return pv.PolyData(vertices.astype(np.float32))
    return pv.PolyData(
        vertices.astype(np.float32),
        np.array(face_arr, dtype=np.int_),
    )


def _build_pyvista_from_raw_graph(data: Data) -> "pv.PolyData":
    """PyG Data (pos + edge_index) nesnesinden PyVista wireframe mesh üretir."""
    if not _PYVISTA_AVAILABLE:
        raise ImportError("PyVista kurulu değil: pip install pyvista")
    pos = data.pos.cpu().numpy().astype(np.float32)
    edge_index = data.edge_index.cpu().numpy()
    edges: set = set()
    for k in range(edge_index.shape[1]):
        u, v = int(edge_index[0, k]), int(edge_index[1, k])
        edges.add((min(u, v), max(u, v)))
    line_cells: List[int] = []
    for u, v in edges:
        line_cells.extend([2, u, v])
    mesh = pv.PolyData()
    mesh.points = pos
    if line_cells:
        mesh.lines = np.array(line_cells, dtype=np.int_)
    return mesh


def _load_nas_as_pyvista(nas_path: Path) -> "pv.PolyData":
    """NAS dosyasını ParserNas ile okur ve PyVista PolyData döndürür."""
    if not _PYVISTA_AVAILABLE:
        raise ImportError("PyVista kurulu değil: pip install pyvista")
    parser = ParserNas()
    mesh_data = parser.parse_file(nas_path)
    nodes    = mesh_data["nodes"]
    elements = mesh_data["elements"]
    node_ids = sorted(nodes.keys())
    node_idx = {nid: i for i, nid in enumerate(node_ids)}
    vertices = np.array([nodes[nid] for nid in node_ids], dtype=np.float32)
    face_arr: List[int] = []
    for elem in elements.values():
        global_ids = [node_idx[nid] for nid in elem["nodes"] if nid in node_idx]
        n = len(global_ids)
        if n >= 3:
            face_arr.extend([n] + global_ids)
    if not face_arr:
        return pv.PolyData(vertices)
    return pv.PolyData(vertices, np.array(face_arr, dtype=np.int_))


# =========================================================================== #
#  Komut Satırı Arayüzü                                                         #
# =========================================================================== #

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="FAZ 5: Graf2Seq Çıkarım ve Mesh Yeniden İnşa  (v2)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--pt-file", type=str, default=None,
                   help="Giriş RAW .pt dosyası. Verilmezse klasörden rastgele seçilir.")
    p.add_argument("--checkpoint", type=str,
                   default=str(PROJECT_ROOT / "checkpoints" / "best_model.pth"),
                   help="Eğitilmiş model checkpoint (.pth) yolu.")
    p.add_argument("--vocab", type=str,
                   default=str(PROJECT_ROOT / "data_vocabulary" / "ai_vocabulary.json"),
                   help="ai_vocabulary.json yolu.")
    p.add_argument("--norm-csv", type=str,
                   default=str(_WORKSPACE / "DATA" / "normalize_parameter.csv"),
                   help="normalize_parameter.csv yolu.")
    p.add_argument("--pt-dir", type=str,
                   default=str(_WORKSPACE / "DATA" / "PT_FILES"),
                   help="Test .pt dosyalarının bulunduğu klasör.")
    p.add_argument("--refined-dir", type=str,
                   default=str(_WORKSPACE / "DATA" / "REFINED_NAS"),
                   help="Ground-truth REFINED .nas dosyalarının klasörü.")
    p.add_argument("--output", type=str,
                   default=str(PROJECT_ROOT / "output_generated.nas"),
                   help="Çıktı NAS dosyası yolu.")
    p.add_argument("--no-vis", action="store_true",
                   help="PyVista görselleştirmeyi devre dışı bırak.")
    p.add_argument("--max-gen-len", type=int, default=DEFAULT_MAX_GEN_LEN,
                   help="Maksimum üretilecek token sayısı.")
    p.add_argument("--repetition-penalty", type=float, default=DEFAULT_REPETITION_PENALTY,
                   help="Tekrar cezası (1.0=devre dışı, >1.0=çeşitliliği artır).")
    p.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE,
                   help="Logit sıcaklık katsayısı (<1.0=keskin, >1.0=rastgele).")
    p.add_argument("--epsilon", type=float, default=DEFAULT_EPSILON,
                   help="Vertex welding mesafe toleransı.")
    p.add_argument("--no-smooth", action="store_true",
                   help="Laplacian smoothing'i devre dışı bırak.")
    p.add_argument("--seed", type=int, default=None,
                   help="Rastgele PT dosyası seçimi için seed.")
    return p.parse_args()


# =========================================================================== #
#  Ana Giriş Noktası                                                            #
# =========================================================================== #

if __name__ == "__main__":
    args = _parse_args()

    print("\n" + "=" * 68)
    print("  MeshCrafter zbggGraphNet3 — FAZ 5: Çıkarım & Yeniden İnşa  v2")
    print("=" * 68)

    # ── 1. Giriş PT dosyasını belirle ────────────────────────────────────────
    if args.pt_file:
        pt_path = Path(args.pt_file)
        if not pt_path.is_file():
            _log.error(f"Belirtilen PT dosyası bulunamadı: {pt_path}")
            sys.exit(1)
    else:
        pt_dir = Path(args.pt_dir)
        if not pt_dir.is_dir():
            _log.error(f"PT dizini bulunamadı: {pt_dir}")
            sys.exit(1)
        pt_files = sorted(pt_dir.glob("*.pt"))
        if not pt_files:
            _log.error(f"'{pt_dir}' klasöründe .pt dosyası bulunamadı.")
            sys.exit(1)
        rng = random.Random(args.seed)
        pt_path = rng.choice(pt_files)
        _log.info(f"Rastgele seçilen PT dosyası: {pt_path.name}")

    file_stem = pt_path.stem
    print(f"\n[Giriş]              {pt_path}")
    print(f"[Çıktı]              {args.output}")
    print(f"[Stem]               {file_stem}")
    print(f"[Repetition Penalty] {args.repetition_penalty}")
    print(f"[Temperature]        {args.temperature}")
    print()

    # ── 2. RAW mesh verisini yükle ────────────────────────────────────────────
    _log.info("RAW mesh yükleniyor …")
    raw_data: Data = torch.load(pt_path, map_location="cpu", weights_only=False)

    _log.info(f"  x.shape         : {list(raw_data.x.shape)}")
    _log.info(f"  edge_index.shape: {list(raw_data.edge_index.shape)}")
    if hasattr(raw_data, "pos") and raw_data.pos is not None:
        _log.info(f"  pos.shape       : {list(raw_data.pos.shape)}")
    if hasattr(raw_data, "y") and raw_data.y is not None:
        _log.info(f"  y (hedef dizi)  : {raw_data.y.shape}  "
                  f"→ toplam {raw_data.y.numel()} token")

    # ── 3. MeshGenerator ─────────────────────────────────────────────────────
    _log.info("\nMeshGenerator başlatılıyor …")
    generator = MeshGenerator(
        checkpoint_path=args.checkpoint,
        vocab_path=args.vocab,
        norm_csv_path=args.norm_csv,
    )

    # ── 4. Token Dizisi Üret ──────────────────────────────────────────────────
    _log.info("\nToken dizisi üretiliyor …")
    t0 = time.time()
    token_ids: List[int] = generator.generate_sequence(
        raw_mesh_data=raw_data,
        max_len=args.max_gen_len,
        repetition_penalty=args.repetition_penalty,
        temperature=args.temperature,
    )
    gen_time = time.time() - t0

    patch_tokens  = [t for t in token_ids if t not in SPECIAL_TOKENS]
    special_count = len(token_ids) - len(patch_tokens)
    token_counter = Counter(patch_tokens)
    unique_tokens = len(token_counter)

    print(f"\n{'─'*60}")
    print("  ★  ÜRETİM İSTATİSTİKLERİ")
    print(f"{'─'*60}")
    print(f"  Toplam üretilen token sayısı  : {len(token_ids)}")
    print(f"  Özel token (START/END/PAD/UNK): {special_count}")
    print(f"  PATCH token sayısı            : {len(patch_tokens)}")
    print(f"  Benzersiz PATCH token türü    : {unique_tokens}")
    print(f"  Üretim süresi                 : {gen_time:.3f} s")
    if token_counter:
        print(f"\n  Token Frekans Dağılımı (İlk 10):")
        max_freq = max(token_counter.values())
        for tid, freq in token_counter.most_common(10):
            bar = "█" * min(int(freq * 30 / max(max_freq, 1)), 30)
            print(f"    Token {tid:3d} : {freq:5d}  {bar}")
    print(f"{'─'*60}\n")

    # ── 5. MeshReconstructor ─────────────────────────────────────────────────
    _log.info("MeshReconstructor başlatılıyor …")
    reconstructor = MeshReconstructor(
        vocabulary=generator.vocabulary,
        norm_params=generator.norm_params,
        epsilon=args.epsilon,
    )

    _log.info("\nMesh yeniden inşa ediliyor …")
    t1 = time.time()
    final_verts, final_faces = reconstructor.reconstruct(
        token_ids=token_ids,
        raw_mesh_data=raw_data,
        output_path=args.output,
        file_stem=file_stem,
        smooth=not args.no_smooth,
    )
    recon_time = time.time() - t1

    # ── 6. Özet Rapor ─────────────────────────────────────────────────────────
    print(f"\n{'─'*60}")
    print("  ★  YENİDEN İNŞA RAPORU")
    print(f"{'─'*60}")
    print(f"  Düğüm sayısı          : {len(final_verts)}")
    print(f"  Yüzey sayısı          : {len(final_faces)}")
    print(f"  Yeniden inşa süresi   : {recon_time:.3f} s")
    print(f"  Çıktı dosyası (NAS)   : {Path(args.output).resolve()}")
    print(f"{'─'*60}")
    print(f"\n  [TAMAMLANDI] Toplam süre: {gen_time + recon_time:.3f} s")
    print("=" * 68 + "\n")

    # ── 7. PyVista Görselleştirme ─────────────────────────────────────────────
    if args.no_vis:
        _log.info("Görselleştirme '--no-vis' ile devre dışı bırakıldı.")
    elif not _PYVISTA_AVAILABLE:
        _log.warning("PyVista kurulu değil; pip install pyvista")
    else:
        _log.info("\nPyVista görselleştirme hazırlanıyor …")

        # RAW mesh
        if hasattr(raw_data, "pos") and raw_data.pos is not None:
            pv_raw = _build_pyvista_from_raw_graph(raw_data)
            _log.info(f"  RAW      : {pv_raw.n_points} nokta, {pv_raw.n_cells} kenar")
        else:
            _log.warning("  RAW mesh 'pos' alanı yok; boş mesh.")
            pv_raw = pv.PolyData()

        # REFINED mesh
        refined_dir = Path(args.refined_dir)
        refined_nas = refined_dir / f"{file_stem}.nas"
        if refined_nas.is_file():
            try:
                pv_refined = _load_nas_as_pyvista(refined_nas)
                _log.info(
                    f"  REFINED  : {pv_refined.n_points} nokta, "
                    f"{pv_refined.n_cells} element"
                )
            except Exception as exc:
                _log.warning(f"  REFINED NAS yüklenemedi ({exc}).")
                pv_refined = pv.PolyData()
        else:
            _log.warning(f"  REFINED NAS bulunamadı: {refined_nas}")
            pv_refined = pv.PolyData()

        # PREDICTION mesh
        pv_pred = _build_pyvista_from_arrays(final_verts, final_faces)
        _log.info(
            f"  PREDICTION: {pv_pred.n_points} nokta, {pv_pred.n_cells} element"
        )

        if pv_pred.n_points == 0:
            _log.error(
                "  PREDICTION mesh boş! "
                "Token üretimi veya sözlük eşleşmesi başarısız."
            )
        else:
            b = pv_pred.bounds
            _log.info(
                f"  PREDICTION sınırları: "
                f"X=[{b[0]:.1f},{b[1]:.1f}] "
                f"Y=[{b[2]:.1f},{b[3]:.1f}] "
                f"Z=[{b[4]:.1f},{b[5]:.1f}]"
            )

        print("\nGörselleştirici açılıyor — kapatmak için [Q] tuşuna bas.\n")
        show_triple_comparison(
            pv_raw,
            pv_refined,
            pv_pred,
            window_title=(
                f"MeshCrafter – {file_stem}  |  RAW | REFINED | PREDICTION"
            ),
        )
