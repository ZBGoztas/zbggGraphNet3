"""
inference/generate_3.py  –  FAZ 5 v4: Ölçeklendirilmiş Yama + Gelişmiş Birleştirme
====================================================================================
generate_2.py'nin üzerine inşa edilmiş; yamalar arasındaki devasa boşlukları
kapatan ve pürüzsüz tek parça mesh elde etmeyi sağlayan iki kritik güncelleyi
içerir:

Yeni MeshReconstructor v4 Pipeline
───────────────────────────────────
    ADIM 0 – Yama Lokal Ölçeklendirme  (YENİ)
        local_coords → merkezle → patch_scale_factor ile büyüt
        Yamalar birbirine temas edene kadar genişletilir.

    ADIM 1 – Yüzey Normaline Göre Döndürme  (v3'ten devralındı)
        trimesh / KDTree+PCA → surface_normal → Rodrigues Rotasyon

    ADIM 2 – Dünya Uzayına Öteleme  (v3'ten devralındı)
        Büyütülmüş+döndürülmüş koordinatlar → anchor (X, Y, Z) noktasına taşı

    ADIM 3a – Gelişmiş KDTree Welding  (YENİ: dinamik epsilon)
        Yama yarıçapına göre otomatik epsilon hesabı
        cKDTree.query_pairs → Union-Find → Laplacian Smoothing

    ADIM 3b – Open3D Poisson / Ball Pivoting  (YENİ: alternatif yöntem)
        Küresel vertex + normal verilerinden tam kapalı yüzey örer
        Kullanıcı --merge-method parametresi ile seçer

Parametrik Kontrol
──────────────────
    --patch-scale-factor  : None (bbox/√n_patches otomatik) veya sabit çarpan
    --merge-method        : kdtree | poisson | ball_pivot
    --epsilon             : KDTree welding toleransı (auto = dinamik)
    --open3d-depth        : Poisson yüzey derinliği (varsayılan 8)

Bağımlılıklar
─────────────
    pip install torch torch_geometric trimesh scipy pyvista open3d

Kullanım
────────
    python inference/generate_3.py
    python inference/generate_3.py --patch-scale-factor 5.0 --merge-method poisson
    python inference/generate_3.py --epsilon 0.15 --smooth-iterations 5
    python inference/generate_3.py --merge-method ball_pivot --no-vis
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
from scipy.spatial import cKDTree
from torch_geometric.data import Data

# ── Proje kökünü sys.path'e ekle ─────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from models import HourGlassGNNEncoder, AutoregressiveMeshDecoder
from training.train import Config, Graph2Seq
from utils.parser_utils import ParserNas

# ── İsteğe bağlı: PyVista ────────────────────────────────────────────────────
try:
    import pyvista as pv
    from utils.visualization import show_triple_comparison
    _PYVISTA_AVAILABLE = True
except ImportError:
    _PYVISTA_AVAILABLE = False
    show_triple_comparison = None  # type: ignore[assignment]

# ── İsteğe bağlı: trimesh (yüzey normal sorgulama) ───────────────────────────
try:
    import trimesh
    import trimesh.proximity
    _TRIMESH_AVAILABLE = True
except ImportError:
    _TRIMESH_AVAILABLE = False

# ── İsteğe bağlı: Open3D (Poisson / Ball Pivoting rekonstrüksiyon) ───────────
try:
    import open3d as o3d
    _OPEN3D_AVAILABLE = True
except ImportError:
    _OPEN3D_AVAILABLE = False


# =========================================================================== #
#  Loglama                                                                      #
# =========================================================================== #

def _build_logger(name: str = "generate3") -> logging.Logger:
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
DEFAULT_EPSILON            = "auto"   # "auto" → dinamik hesap, float → sabit
DEFAULT_REPETITION_PENALTY = 1.5
DEFAULT_TEMPERATURE        = 1.0
DEFAULT_SMOOTH_ITERS       = 3
DEFAULT_SMOOTH_FACTOR      = 0.5
DEFAULT_OPEN3D_DEPTH       = 8        # Poisson rekonstrüksiyon derinliği
# patch_scale_factor için None → reconstruct() içinde norm_scale kullanılır
DEFAULT_PATCH_SCALE_FACTOR: Optional[float] = None


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
#  SINIF 1: MeshGenerator  (generate_2.py ile aynı, değişiklik yok)            #
# =========================================================================== #

class MeshGenerator:
    """Eğitilmiş Graph2Seq modelini kullanarak RAW mesh grafiğinden token dizisi üretir."""

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
            load_normalization_params(norm_csv_path) if norm_csv_path else {}
        )

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

    @torch.no_grad()
    def generate_sequence(
        self,
        raw_mesh_data: Data,
        max_len: int = DEFAULT_MAX_GEN_LEN,
        repetition_penalty: float = DEFAULT_REPETITION_PENALTY,
        temperature: float = DEFAULT_TEMPERATURE,
    ) -> List[int]:
        """Otoregressif greedy decoding — repetition_penalty ile greedy collapse önlenir."""
        _log.info("[generate_sequence] Başlatılıyor …")
        t_start = time.time()

        data = raw_mesh_data.to(self.device)
        batch_vector = torch.zeros(data.x.size(0), dtype=torch.long, device=self.device)

        context: torch.Tensor = self.model.encoder(data.x, data.edge_index, batch_vector)
        _log.info(f"[generate_sequence] Latent context: {list(context.shape)}")

        generated = torch.tensor([[TOKEN_START]], dtype=torch.long, device=self.device)

        for step in range(max_len):
            logits = self.model.decoder(tgt_tokens=generated, memory=context)
            next_logits = logits[0, -1, :].clone()

            if repetition_penalty != 1.0:
                for prev_id in set(generated[0].tolist()):
                    if 0 <= prev_id < next_logits.size(0):
                        if next_logits[prev_id] > 0:
                            next_logits[prev_id] /= repetition_penalty
                        else:
                            next_logits[prev_id] *= repetition_penalty

            if temperature != 1.0 and temperature > 0:
                next_logits = next_logits / temperature

            next_token_id: int = int(next_logits.argmax().item())
            generated = torch.cat(
                [generated,
                 torch.tensor([[next_token_id]], dtype=torch.long, device=self.device)],
                dim=1,
            )

            if next_token_id == TOKEN_END:
                _log.info(f"[generate_sequence] <END> alındı. Adım: {step + 1}")
                break
        else:
            _log.warning(f"[generate_sequence] max_len={max_len} sınırına ulaşıldı.")

        token_list: List[int] = generated[0].cpu().tolist()
        elapsed = time.time() - t_start
        patch_tokens = [t for t in token_list if t not in SPECIAL_TOKENS]
        counter = Counter(patch_tokens)
        _log.info(
            f"[generate_sequence] Tamamlandı | toplam={len(token_list)} | "
            f"patch={len(patch_tokens)} | benzersiz={len(counter)} | "
            f"en sık={counter.most_common(5)} | süre={elapsed:.2f}s"
        )
        return token_list


# =========================================================================== #
#  SINIF 2: MeshReconstructor v4                                                #
# =========================================================================== #

class MeshReconstructor:
    """
    Token ID dizisini 4 adımlı pipeline ile pürüzsüz 3D mesh'e dönüştürür.

    v4 Pipeline
    ───────────
    ADIM 0 — Yama Lokal Ölçeklendirme  [YENİ]
        Yama köşeleri, merkeze çekildikten sonra `patch_scale_factor` ile
        çarpılır. Bu, minyatür yamalar arasındaki boşlukları kapatır.
        Formül: scaled = (local - patch_center) × patch_scale_factor

    ADIM 1 — Yüzey Normaline Göre Döndürme  [v3'ten devralındı]
        Rodrigues formülü: [0,0,1] → RAW mesh surface_normal
        Büyütülmüş koordinatlar kendi merkezi etrafında döndürülür.

    ADIM 2 — Dünya Uzayına Öteleme  [v3'ten devralındı + iyileştirildi]
        Döndürülmüş koordinatlar → anchor (X, Y, Z) noktasına ötele.
        Normalizasyon geri alımı NAS export aşamasında uygulanır.

    ADIM 3a — Gelişmiş KDTree Welding  [YENİ: dinamik epsilon]
        Yama ortalama yarıçapından epsilon otomatik hesaplanır:
            auto_epsilon = medyan_yama_yarıçapı × 0.5
        Manuel epsilon de desteklenir (--epsilon 0.15 gibi).
        cKDTree → Union-Find → Laplacian Smoothing

    ADIM 3b — Open3D Poisson / Ball Pivoting  [YENİ: alternatif yöntem]
        Tüm global vertex + surface_normal verilerinden yeni bir ağ örer.
        --merge-method poisson  : Poisson Surface Reconstruction (tam kapalı)
        --merge-method ball_pivot: Ball Pivoting Algorithm (açık yüzey uyumlu)

    Parametreler
    ────────────
    vocabulary        : ai_vocabulary.json içeriği (dict)
    norm_params       : normalize_parameter.csv'den yüklenen parametreler
    raw_mesh_trimesh  : trimesh.Trimesh objesi; None ise KDTree+PCA kullanılır
    epsilon           : Vertex welding toleransı. "auto" → dinamik hesap.
    patch_scale_factor: Yama büyütme katsayısı (sabit çarpan modu).
                        None (varsayılan) → RAW mesh bbox / √n_patches otomatik hedef yarıçap.
                        Örn: 5.0 → centered × 5 uygulanır.
    """

    _DEFAULT_PATCH_NORMAL = np.array([0.0, 0.0, 1.0], dtype=np.float64)

    def __init__(
        self,
        vocabulary: Dict,
        norm_params: Optional[Dict[str, Dict[str, float]]] = None,
        raw_mesh_trimesh: Optional["trimesh.Trimesh"] = None,
        epsilon: float | str = DEFAULT_EPSILON,
        patch_scale_factor: Optional[float] = DEFAULT_PATCH_SCALE_FACTOR,
    ) -> None:
        self.vocabulary         = vocabulary
        self.norm_params        = norm_params or {}
        self.raw_mesh_trimesh   = raw_mesh_trimesh
        self.epsilon            = epsilon           # float veya "auto"
        self.patch_scale_factor = patch_scale_factor  # None → otomatik bbox/√n modu

        # Global havuz – reconstruct() başında sıfırlanır
        self._all_verts:   List[np.ndarray] = []
        self._all_faces:   List[List[int]]  = []
        self._all_normals: List[np.ndarray] = []   # Open3D için: her vertex'in normali
        self._vert_offset: int              = 0

        # Yama boyut istatistikleri – dinamik epsilon için
        self._patch_radii: List[float]      = []

    # ── Rodrigues Rotasyon Matrisi ────────────────────────────────────────────

    @staticmethod
    def _rotation_matrix_from_vectors(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
        """
        src birim vektörünü dst birim vektörüne döndüren 3×3 rotasyon matrisi.
        Rodrigues formülü; paralel ve ters-paralel özel durumları işler.
        """
        src = src / (np.linalg.norm(src) + 1e-12)
        dst = dst / (np.linalg.norm(dst) + 1e-12)
        v   = np.cross(src, dst)
        c   = float(np.dot(src, dst))
        s   = np.linalg.norm(v)

        if s < 1e-10:
            if c > 0.0:
                return np.eye(3, dtype=np.float64)
            # 180° ters yönde: src'ye dik bir eksen etrafında döndür
            perp = (np.array([1.0, 0.0, 0.0]) if abs(src[0]) < 0.9
                    else np.array([0.0, 1.0, 0.0]))
            axis = np.cross(src, perp)
            axis /= np.linalg.norm(axis)
            kmat = np.array([
                [ 0.0,      -axis[2],  axis[1]],
                [ axis[2],   0.0,     -axis[0]],
                [-axis[1],   axis[0],  0.0    ],
            ])
            return np.eye(3, dtype=np.float64) + 2.0 * kmat @ kmat

        kmat = np.array([
            [ 0.0,  -v[2],  v[1]],
            [ v[2],  0.0,  -v[0]],
            [-v[1],  v[0],  0.0 ],
        ])
        return (np.eye(3, dtype=np.float64)
                + kmat
                + kmat @ kmat * ((1.0 - c) / (s ** 2)))

    # ── Yüzey Normal Sorgulama ────────────────────────────────────────────────

    def _surface_normal_at(
        self,
        target_xyz: np.ndarray,
        raw_pos: Optional[np.ndarray] = None,
        raw_kdtree: Optional[cKDTree] = None,
    ) -> np.ndarray:
        """
        target_xyz konumundaki tahmini yüzey normalini döndürür.

        Önce raw_mesh_trimesh ile trimesh.proximity sorgusu dener.
        trimesh yoksa KDTree+PCA ile komşulardan normal tahmin eder.
        Fallback: Z-ekseni [0, 0, 1].
        """
        # ── Yol 1: trimesh proximity ─────────────────────────────────────────
        if self.raw_mesh_trimesh is not None and _TRIMESH_AVAILABLE:
            try:
                query = target_xyz.reshape(1, 3)
                _, _, tri_ids = trimesh.proximity.closest_point(
                    self.raw_mesh_trimesh, query
                )
                normal = self.raw_mesh_trimesh.face_normals[tri_ids[0]].astype(np.float64)
                n_len = np.linalg.norm(normal)
                if n_len > 1e-12:
                    return normal / n_len
            except Exception as exc:
                _log.debug(f"[surface_normal_at] trimesh hatası: {exc}")

        # ── Yol 2: KDTree + PCA ──────────────────────────────────────────────
        if raw_pos is not None and raw_kdtree is not None:
            k = min(15, len(raw_pos))
            _, idx = raw_kdtree.query(target_xyz, k=k)
            idx = np.atleast_1d(idx)
            nbrs = raw_pos[idx]
            centered = nbrs - nbrs.mean(axis=0)
            cov = centered.T @ centered
            _, eigvecs = np.linalg.eigh(cov)
            normal = eigvecs[:, 0]  # en küçük öz değer → yüzey normali
            n_len = np.linalg.norm(normal)
            if n_len > 1e-12:
                return normal / n_len

        # ── Fallback ──────────────────────────────────────────────────────────
        return self._DEFAULT_PATCH_NORMAL.copy()

    # ── ADIM 0 + 1 + 2: Tek Yama İşleme ─────────────────────────────────────

    def _process_patch(
        self,
        local_coords: np.ndarray,                    # [n_verts, 3] — normalize yerel uzay
        local_face_indices: List[List[int]],         # yama içi yüzey bağlantıları
        target_xyz: np.ndarray,                      # RAW mesh anchor (normalize uzay)
        patch_scale: Optional[float] = None,         # Sabit çarpan modu (ADIM 0)
        target_patch_radius: Optional[float] = None, # Hedef yarıçap modu (ADIM 0)
        raw_pos: Optional[np.ndarray] = None,
        raw_kdtree: Optional[cKDTree] = None,
    ) -> None:
        """
        Tek bir yamayı Scale → Rotate → Translate pipeline'ından geçirir
        ve global havuza ekler.

        ADIM 0 — Lokal Ölçeklendirme (iki mod)
        ────────────────────────────────────────
        Mod 1: target_patch_radius (otomatik, önerilen)
            Her yama kendi vocab boyutuna bağımsız olarak
            `target_patch_radius` birim yarıçapa normalleştirilir.
            effective_scale = target_patch_radius / vocab_patch_radius
            Bu mod tüm yamaları eşit "hücre boyutuna" getirir.

        Mod 2: patch_scale (sabit çarpan, --patch-scale-factor ile)
            scaled = centered × patch_scale
            Tüm yamalara aynı çarpan uygulanır; vocab içi boyut farkları korunur.

        Dönüşüm Sırası
        ──────────────
        1. Yama merkezini origin'e taşı        → centered
        2. Ölçekle (mod 1 veya mod 2)          → scaled   (ADIM 0)
        3. Rodrigues rotasyonu uygula           → rotated  (ADIM 1)
        4. target_xyz'e ötele                  → norm_coords (ADIM 2)
        """
        if len(local_coords) == 0:
            return

        # ── ADIM 0: Yama merkezini origin'e taşı ─────────────────────────────
        patch_center = local_coords.mean(axis=0)
        centered     = local_coords - patch_center        # yerel merkez = origin

        # Vocab'daki asıl yama yarıçapı (ölçekleme öncesi, ham vocab boyutu)
        distances = np.linalg.norm(centered, axis=1)
        # Ortanca kullan (max yerine): tek bir aykırı köşe tüm yamanın
        # effective_scale'ini patlatmasın (ince çubuk/spike artefaktı).
        patch_radius = float(np.median(distances) + 1e-12)

        # ── Ölçekleme: iki mod ────────────────────────────────────────────────
        if target_patch_radius is not None:
            # Mod 1 (otomatik): her yamayı hedef yarıçapa normalleştir.
            # Sınırla: çok kompakt vocab yamalarının patlamaması (spike önleme).
            raw_scale = target_patch_radius / patch_radius
            effective_scale = float(np.clip(raw_scale, 0.5, 12.0))
        elif patch_scale is not None:
            # Mod 2 (sabit): kullanıcı tarafından verilen sabit çarpan
            effective_scale = float(patch_scale)
        else:
            effective_scale = 1.0  # ölçekleme yok (fallback)

        # Epsilon için ÖLÇEKLENMIŞ yarıçapı kaydet (ölçekleme sonrası gerçek boyut).
        # _compute_auto_epsilon bu değeri kullanır → epsilon gerçek patch boyutuna göre.
        self._patch_radii.append(effective_scale * patch_radius)

        scaled = centered * effective_scale               # [n_verts, 3]

        # ── ADIM 1: Hedef konumun yüzey normalini sorgula, yamayı döndür ──────
        surface_normal = self._surface_normal_at(
            target_xyz, raw_pos=raw_pos, raw_kdtree=raw_kdtree
        )
        rot_mat = self._rotation_matrix_from_vectors(
            self._DEFAULT_PATCH_NORMAL, surface_normal
        )
        rotated = (rot_mat @ scaled.T).T                  # [n_verts, 3]

        # ── ADIM 2: Normalize uzay anchor'una ötele ───────────────────────────
        # Her köşe surface_normal yönünde döndürülmüş ve target_xyz etrafında
        # yerleştirilmiştir. Normalize uzayda çalışıyoruz; NAS export için
        # denormalizasyon reconstruct() içinde yapılır.
        norm_coords = rotated + target_xyz                # [n_verts, 3]

        # Open3D için: her köşeye bu yamanın surface_normal'ını ata ───────────
        vert_normals = np.tile(surface_normal, (len(norm_coords), 1))  # [n, 3]

        # ── Global havuza ekle (yüzey indeksleri ofseti güncelle) ─────────────
        offset = self._vert_offset
        self._all_verts.append(norm_coords)
        self._all_normals.append(vert_normals)
        for face in local_face_indices:
            self._all_faces.append([offset + vi for vi in face])
        self._vert_offset += len(norm_coords)

    # ── ADIM 3a: Gelişmiş KDTree Vertex Welding ───────────────────────────────

    def _compute_auto_epsilon(self) -> float:
        """
        Ölçeklenmiş yama yarıçaplarından welding toleransı (epsilon) hesaplanır.

        v4'te `_patch_radii` artık ölçekleme SONRASI efektif yarıçapları içerir
        (genellikle target_patch_radius civarı). İki bitişik yamayı birleştirmek için
        epsilon, efektif yarıçapın küçük bir fraksiyonu olmalıdır:
            • Çok büyük epsilon → alakasız yamalar kaynaşır ("over-welding")
            • Çok küçük epsilon → dokunuşan kenarlar kaynaşmaz (kopuk mesh)

        Hedef: komşu yamaların paylaşılan kenar köşeleri birleşsin.
        Paylaşılan kenar pozisyon hatası tipik olarak efektif_r × 0.1–0.2 mertebesindedir.
        """
        if not self._patch_radii:
            return 0.05  # varsayılan güvenli değer

        median_radius = float(np.median(self._patch_radii))
        # Fraksiyon 0.40: FPS-NN tabanlı ölçekleme ile yamalar ~%10 örtüşür.
        # Sınır köşe pozisyon hatası ≈ 0.10 × d = 0.18 × r → epsilon 0.40×r > 0.18×r
        # Bu, temas eden yamalar arasındaki boşluğu kapatır (over-welding olmadan).
        auto_eps = median_radius * 0.40
        # Aralık sınırı: 0.003 (çok küçük = hiç kaynaşma yok) – 0.6 (aşırı kaynaşma)
        auto_eps = float(np.clip(auto_eps, 0.003, 0.6))
        _log.info(
            f"[auto_epsilon] scaled_median_radius={median_radius:.5f} → "
            f"epsilon={auto_eps:.5f}"
        )
        return auto_eps

    def _weld_vertices(
        self,
        epsilon: float,
    ) -> Tuple[np.ndarray, List[List[int]]]:
        """
        Gelişmiş KDTree tabanlı Vertex Welding.

        v4 Yenilikleri
        ──────────────
        • epsilon artık dışarıdan parametre olarak alınır (auto veya sabit).
        • Aşırı welding tespitinde (%80+ köşe kaybı) uyarı ve epsilon önerisi.

        Algoritma
        ---------
        1. Tüm köşeleri birleştir → numpy dizisi [N, 3]
        2. cKDTree oluştur
        3. query_pairs(r=epsilon) → yakın çift tespiti
        4. Union-Find ile bağlı bileşenleri grupla
        5. Her grup için ortalama konum → yeni köşe dizisi
        6. Eski yüzey indekslerini yeni indekslerle güncelle
        7. Dejenere yüzeyleri (< 3 benzersiz köşe) kaldır
        """
        if not self._all_verts:
            _log.warning("[weld_vertices] Boş vertex havuzu.")
            return np.empty((0, 3), dtype=np.float64), []

        all_v = np.vstack(self._all_verts)   # [N, 3]
        all_f = self._all_faces
        n     = len(all_v)

        _log.info(
            f"[weld_vertices] {n} köşe, {len(all_f)} yüzey, epsilon={epsilon:.6f}"
        )

        # ── KDTree + yakın çift tespiti ───────────────────────────────────────
        tree  = cKDTree(all_v)
        pairs = tree.query_pairs(r=epsilon)
        _log.info(f"[weld_vertices] {len(pairs)} yakın çift bulundu.")

        # ── Union-Find (path compression + union by rank) ─────────────────────
        parent = list(range(n))
        rank   = [0] * n

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]  # path halving
                x = parent[x]
            return x

        def union(x: int, y: int) -> None:
            rx, ry = find(x), find(y)
            if rx == ry:
                return
            if rank[rx] < rank[ry]:
                rx, ry = ry, rx
            parent[ry] = rx
            if rank[rx] == rank[ry]:
                rank[rx] += 1

        for i, j in pairs:
            union(i, j)

        # ── Grup → yeni köşe üret (ortalama konum) ───────────────────────────
        groups: Dict[int, List[int]] = defaultdict(list)
        for i in range(n):
            groups[find(i)].append(i)

        # Yeni köşe indeksi: eski indeks → yeni indeks
        new_idx: Dict[int, int] = {}
        new_verts_list: List[np.ndarray] = []
        for new_id, (_, members) in enumerate(sorted(groups.items())):
            avg_pos = all_v[members].mean(axis=0)
            new_verts_list.append(avg_pos)
            for old_id in members:
                new_idx[old_id] = new_id

        new_verts = np.array(new_verts_list, dtype=np.float64)  # [M, 3]

        # ── Yüzey indekslerini güncelle, dejenere yüzeyleri ayıkla ───────────
        new_faces: List[List[int]] = []
        removed = 0
        for face in all_f:
            remapped = [new_idx[vi] for vi in face]
            if len(set(remapped)) >= 3:   # dejenere değil
                new_faces.append(remapped)
            else:
                removed += 1

        n_before = len(all_v)
        n_after  = len(new_verts)
        reduction_pct = 100.0 * (n_before - n_after) / max(n_before, 1)
        _log.info(
            f"[weld_vertices] {n_before} → {n_after} köşe "
            f"(%{reduction_pct:.1f} birleştirildi) | "
            f"yüzey: {len(all_f)} → {len(new_faces)} "
            f"({removed} dejenere kaldırıldı)."
        )

        if reduction_pct > 80:
            _log.warning(
                f"[weld_vertices] ⚠  Aşırı welding tespit edildi "
                f"(%{reduction_pct:.0f} köşe kaybı)! "
                f"epsilon={epsilon:.4f} çok büyük olabilir. "
                f"--epsilon 0.01 ile tekrar deneyin."
            )

        return new_verts, new_faces

    # ── ADIM 3b: Laplacian Smoothing ─────────────────────────────────────────

    @staticmethod
    def _laplacian_smooth(
        vertices: np.ndarray,
        faces: List[List[int]],
        iterations: int = DEFAULT_SMOOTH_ITERS,
        factor: float = DEFAULT_SMOOTH_FACTOR,
    ) -> np.ndarray:
        """
        Basit Laplacian Smoothing.

        Her vertex'e, bağlı komşu vertex'lerin ortalamasına doğru
        (factor) katsayısıyla iteratif yaklaştırma uygulanır.
        factor=0.5, iterations=3 genellikle gürültüyü gidermekte yeterlidir.
        """
        n = len(vertices)
        if n == 0:
            return vertices

        # Komşuluk grafiğini oluştur
        neighbors: Dict[int, set] = defaultdict(set)
        for face in faces:
            for k in range(len(face)):
                vi = face[k]
                vj = face[(k + 1) % len(face)]
                if 0 <= vi < n and 0 <= vj < n:
                    neighbors[vi].add(vj)
                    neighbors[vj].add(vi)

        verts = vertices.copy()
        for _ in range(iterations):
            new_verts = verts.copy()
            for i in range(n):
                nbrs = list(neighbors.get(i, []))
                if nbrs:
                    avg = verts[nbrs].mean(axis=0)
                    new_verts[i] = verts[i] + factor * (avg - verts[i])
            verts = new_verts

        _log.info(f"[laplacian_smooth] {iterations} iterasyon tamamlandı.")
        return verts

    # ── Farthest Point Sampling (FPS) ────────────────────────────────────────

    @staticmethod
    def _farthest_point_sample(pos: np.ndarray, n_samples: int) -> np.ndarray:
        """
        Farthest Point Sampling: n_samples noktayı uzayda maksimum mesafeyle seç.

        pos'taki noktalardan yinelemeli olarak en uzak noktayı seçerek
        geometrik olarak eşit dağılımlı bir alt küme üretir.
        Naive O(N×K) implementasyonu; numpy vektörizasyonu ile hızlandırılmış.

        Dönüş: seçilen indeksleri içeren int64 dizisi [n_samples]
        """
        n = len(pos)
        if n == 0 or n_samples <= 0:
            return np.empty(0, dtype=np.int64)
        n_samples = min(n_samples, n)
        indices  = np.zeros(n_samples, dtype=np.int64)
        dist_sq  = np.full(n, np.inf, dtype=np.float64)
        current  = 0
        for i in range(n_samples):
            indices[i] = current
            diff = pos - pos[current]               # [N, 3]
            d    = (diff * diff).sum(axis=1)        # [N] — kare-uzaklık (FPS için yeterli)
            np.minimum(dist_sq, d, out=dist_sq)
            current = int(np.argmax(dist_sq))
        return indices

    # ── ADIM 3c (alternatif): Open3D Poisson / Ball Pivoting ─────────────────

    def reconstruct_open3d(
        self,
        method: str = "poisson",
        poisson_depth: int = DEFAULT_OPEN3D_DEPTH,
        ball_radii: Optional[List[float]] = None,
    ) -> Tuple[np.ndarray, List[List[int]]]:
        """
        Open3D ile Global Nokta Bulutundan Yüzey Rekonstrüksiyonu.

        Yöntemler
        ---------
        method = "poisson"
            Poisson Surface Reconstruction (kapalı, pürüzsüz yüzey).
            depth parametresi ayrıntı seviyesini kontrol eder (6–10 arası önerilir).
            Düşük yoğunluklu bölgeler trim_factor ile budanır.

        method = "ball_pivot"
            Ball Pivoting Algorithm (açık yüzeylere daha uyumlu).
            ball_radii listesi verilmezse ortalama yama yarıçapından otomatik belirlenir.

        Bağımlılık
        ----------
        pip install open3d

        Dönüş
        -----
        (vertices [N, 3], faces [[a, b, c], ...])
        faces triangüle edilmiştir (hepsi 3-vertex).
        """
        if not _OPEN3D_AVAILABLE:
            raise ImportError(
                "Open3D kurulu değil: pip install open3d\n"
                "Alternatif: --merge-method kdtree kullanın."
            )

        if not self._all_verts:
            _log.warning("[reconstruct_open3d] Vertex havuzu boş.")
            return np.empty((0, 3), dtype=np.float64), []

        # ── Tüm vertex + normal verilerini birleştir ──────────────────────────
        all_v = np.vstack(self._all_verts).astype(np.float64)   # [N, 3]
        all_n = np.vstack(self._all_normals).astype(np.float64) # [N, 3]

        _log.info(
            f"[reconstruct_open3d] method={method}, "
            f"{len(all_v)} nokta → Open3D PointCloud oluşturuluyor …"
        )

        # ── Open3D PointCloud nesnesi oluştur ─────────────────────────────────
        pcd = o3d.geometry.PointCloud()
        pcd.points  = o3d.utility.Vector3dVector(all_v)
        pcd.normals = o3d.utility.Vector3dVector(all_n)

        # ── Normal tutarlılığını düzelt (hepsi aynı yöne baksın) ─────────────
        # Büyük nokta bulutlarında bazı normaller ters yönde olabilir.
        pcd.orient_normals_consistent_tangent_plane(k=10)

        if method == "poisson":
            _log.info(
                f"[reconstruct_open3d] Poisson Surface Reconstruction "
                f"(depth={poisson_depth}) başlıyor …"
            )
            # Poisson rekonstrüksiyonu: tam kapalı, pürüzsüz mesh üretir
            mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
                pcd,
                depth=poisson_depth,
                width=0,
                scale=1.1,
                linear_fit=False,
            )
            # Düşük yoğunluklu (nokta bulutu tarafından desteklenmeyen) üçgenleri buda
            densities_np = np.asarray(densities)
            trim_threshold = np.percentile(densities_np, 10)  # alt %10'u kaldır
            verts_to_remove = densities_np < trim_threshold
            mesh.remove_vertices_by_mask(verts_to_remove)

            _log.info(
                f"[reconstruct_open3d] Poisson tamamlandı: "
                f"{len(np.asarray(mesh.vertices))} köşe, "
                f"{len(np.asarray(mesh.triangles))} üçgen."
            )

        elif method == "ball_pivot":
            # Yarıçap listesi otomatik belirlenmezse yama istatistiklerinden türet
            if ball_radii is None:
                if self._patch_radii:
                    med = float(np.median(self._patch_radii))
                    ball_radii = [med * 0.5, med * 1.0, med * 2.0]
                else:
                    ball_radii = [0.05, 0.1, 0.2]
            _log.info(
                f"[reconstruct_open3d] Ball Pivoting (radii={ball_radii}) başlıyor …"
            )
            radii_o3d = o3d.utility.DoubleVector(ball_radii)
            mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
                pcd, radii_o3d
            )
            _log.info(
                f"[reconstruct_open3d] Ball Pivoting tamamlandı: "
                f"{len(np.asarray(mesh.vertices))} köşe, "
                f"{len(np.asarray(mesh.triangles))} üçgen."
            )

        else:
            raise ValueError(
                f"Bilinmeyen Open3D yöntemi: '{method}'. "
                "Geçerli değerler: 'poisson', 'ball_pivot'"
            )

        # ── Open3D Mesh → numpy dizilerine dönüştür ───────────────────────────
        out_verts = np.asarray(mesh.vertices, dtype=np.float64)
        out_tris  = np.asarray(mesh.triangles, dtype=np.int64)

        if len(out_verts) == 0 or len(out_tris) == 0:
            _log.warning(
                "[reconstruct_open3d] Boş mesh üretildi. "
                "Nokta bulutunun yeterli yoğunlukta olduğundan emin olun. "
                "--patch-scale-factor değerini artırmayı deneyin."
            )
            return np.empty((0, 3), dtype=np.float64), []

        out_faces = out_tris.tolist()
        return out_verts, out_faces

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
            f.write("$ MeshCrafter zbggGraphNet3 – FAZ 5 v4 Çıktısı\n")
            f.write(f"$ Düğüm sayısı : {len(vertices)}\n")
            f.write(f"$ Yüzey sayısı : {len(faces)}\n")
            f.write("BEGIN BULK\n")
            for i, (vx, vy, vz) in enumerate(vertices, start=1):
                f.write(f"GRID,{i},,{vx:.8f},{vy:.8f},{vz:.8f}\n")
            eid = 1
            for face in faces:
                nv = len(face)
                if nv < 3:
                    continue
                if nv == 3:
                    n1, n2, n3 = (vi + 1 for vi in face)
                    f.write(f"CTRIA3,{eid},1,{n1},{n2},{n3}\n")
                    eid += 1
                elif nv == 4:
                    n1, n2, n3, n4 = (vi + 1 for vi in face)
                    f.write(f"CQUAD4,{eid},1,{n1},{n2},{n3},{n4}\n")
                    eid += 1
                else:
                    # n > 4: fan triangülasyon
                    for k in range(1, nv - 1):
                        f.write(
                            f"CTRIA3,{eid},1,"
                            f"{face[0]+1},{face[k]+1},{face[k+1]+1}\n"
                        )
                        eid += 1
            f.write("ENDDATA\n")
        _log.info(f"[export_nas] Kaydedildi: {out.resolve()}")

    # ── Ana Yeniden İnşa Metodu ──────────────────────────────────────────────

    def reconstruct(
        self,
        token_ids: List[int],
        raw_mesh_data: Data,
        output_path: str,
        file_stem: str = "generated_mesh",
        merge_method: str = "kdtree",
        smooth: bool = True,
        smooth_iterations: int = DEFAULT_SMOOTH_ITERS,
        smooth_factor: float = DEFAULT_SMOOTH_FACTOR,
        poisson_depth: int = DEFAULT_OPEN3D_DEPTH,
        ball_radii: Optional[List[float]] = None,
    ) -> Tuple[np.ndarray, List[List[int]]]:
        """
        Token ID dizisini v4 pipeline'dan geçirerek NAS dosyasına dönüştürür.

        Pipeline
        ────────
        1. Her patch token için:
               ADIM 0  Lokal Ölçeklendirme
                       Otomatik mod: hedef_r = bbox_ext / (2 × √n_patches)
                       Sabit mod   : centered × --patch-scale-factor
               ADIM 1  Yüzey Normaline Göre Döndürme
               ADIM 2  Dünya Uzayına Öteleme
           → normalize uzayda çalışır

        2. merge_method seçimine göre:
               "kdtree"     → ADIM 3a: KDTree Welding (dinamik epsilon) + Smooth
               "poisson"    → ADIM 3b: Open3D Poisson Surface Reconstruction
               "ball_pivot" → ADIM 3b: Open3D Ball Pivoting Algorithm

        3. Denormalizasyon: world = normalized × norm_scale + norm_center
        4. NAS dışa aktarma

        Dönüş değerleri normalize uzaydadır (RAW mesh ile aynı ölçek, viz için).

        Parametreler
        ────────────
        merge_method    : "kdtree" | "poisson" | "ball_pivot"
        smooth          : KDTree yöntemi için Laplacian Smoothing uygula
        smooth_iterations: Smoothing iterasyon sayısı
        smooth_factor   : Smoothing katsayısı (0 < f < 1)
        poisson_depth   : Open3D Poisson derinliği (daha yüksek = daha detaylı)
        ball_radii      : Ball Pivoting için yarıçap listesi (None → otomatik)
        """
        # ── Durum sıfırlama ─────────────────────────────────────────────────
        self._all_verts   = []
        self._all_faces   = []
        self._all_normals = []
        self._vert_offset = 0
        self._patch_radii = []

        # ── RAW mesh verilerini hazırla ──────────────────────────────────────
        raw_pos: Optional[np.ndarray] = None
        raw_kdtree: Optional[cKDTree] = None

        if hasattr(raw_mesh_data, "pos") and raw_mesh_data.pos is not None:
            raw_pos    = raw_mesh_data.pos.cpu().numpy().astype(np.float64)
            raw_kdtree = cKDTree(raw_pos)
            _log.info(f"[reconstruct] RAW mesh: {raw_pos.shape[0]} düğüm")

        # ── Normalizasyon parametrelerini yükle ─────────────────────────────
        norm_scale  = 1.0
        norm_center = np.zeros(3, dtype=np.float64)

        if self.norm_params and file_stem:
            params = self.norm_params.get(file_stem)
            if params is not None:
                norm_scale  = float(params["scale_max_radius"])
                norm_center = np.array(
                    [params["center_x"], params["center_y"], params["center_z"]],
                    dtype=np.float64,
                )
                _log.info(
                    f"[reconstruct] Norm: scale={norm_scale:.4f}, "
                    f"center={norm_center.tolist()}"
                )
            else:
                _log.warning(
                    f"[reconstruct] '{file_stem}' norm parametresi bulunamadı; "
                    "scale=1, center=0 kullanılıyor."
                )

        # ── PATCH tokenlarını filtrele (n_patches scale hesabı için önce lazım) ─
        patch_token_ids = [t for t in token_ids if t not in SPECIAL_TOKENS]
        n_patches = len(patch_token_ids)
        n_raw     = raw_pos.shape[0] if raw_pos is not None else 0

        # ── FPS: Uzayda eşit dağılımlı anchor noktaları seç ─────────────────
        # Doğrusal indeksleme (patch_idx * n_raw / n_patches) yerine FPS kullanılır.
        # Bu, yamaların RAW mesh yüzeyine geometrik olarak eşit dağılmasını sağlar;
        # kümelenmeden kaçınır ve tüm yüzeyin kapsanmasını garanti eder.
        # FPS ÖNCE hesaplanır — scale için FPS komşu mesafeleri kullanılır.
        fps_anchors: Optional[np.ndarray] = None
        if raw_pos is not None and n_patches > 0:
            _log.info(f"[reconstruct] FPS başlatılıyor: {n_patches} anchor, {len(raw_pos)} RAW düğüm …")
            fps_anchors = self._farthest_point_sample(raw_pos, n_patches)
            _log.info(f"[reconstruct] FPS tamamlandı: {len(fps_anchors)} anchor seçildi.")

        # ── Patch Scale Factor belirleme ─────────────────────────────────────
        # İki mod:
        #   Mod 1 (önerilen, otomatik): FPS anchor noktaları arasındaki en yakın
        #     komşu mesafelerinden hedef yarıçap hesaplanır.
        #     target_r = median_NN_dist × 0.55  → yamalar birbirine %10 örtüşerek değer
        #     Bu, her yamanın komşusuna tam olarak temas etmesini (touching) sağlar.
        #
        #   Mod 2 (sabit çarpan): --patch-scale-factor ile kullanıcı tarafından
        #     verilen sabit katsayı (örn. 5.0). centered × factor uygulanır.
        #
        # NOT: Eskiden kullanılan `norm_scale` (100–500 arası) doğrudan çarpan
        # olarak kullanılıyordu. Bu, yamaları birkaç birimlik devlere dönüştürdü.
        # norm_scale artık YALNIZCA NAS dışa aktarma (denormalizasyon) için kullanılır.
        fixed_patch_scale: Optional[float] = None
        target_patch_radius: Optional[float] = None

        if self.patch_scale_factor is not None:
            # Kullanıcı sabit çarpan verdi
            fixed_patch_scale = float(self.patch_scale_factor)
            _log.info(
                f"[reconstruct] patch_scale_factor={fixed_patch_scale} (sabit çarpan modu)"
            )
        elif fps_anchors is not None and len(fps_anchors) > 1:
            # Otomatik: FPS anchor NN mesafesine göre hedef yarıçap hesapla.
            # Komşu yamalar birbirine temas edecek şekilde: r = 0.55 × median_NN_dist
            fps_positions = raw_pos[fps_anchors]
            fps_kd        = cKDTree(fps_positions)
            k_nn          = min(2, len(fps_positions))
            dists_nn, _   = fps_kd.query(fps_positions, k=k_nn)
            if dists_nn.ndim > 1 and dists_nn.shape[1] > 1:
                median_nn = float(np.median(dists_nn[:, 1]))
            else:
                median_nn = float(np.median(dists_nn.ravel()))
            target_patch_radius = max(median_nn * 0.55, 1e-6)
            _log.info(
                f"[reconstruct] FPS-NN tabanlı hedef yarıçap: "
                f"median_NN={median_nn:.6f}, "
                f"target_patch_radius={target_patch_radius:.6f}"
            )
        else:
            fixed_patch_scale = 1.0  # verisi yoksa ölçekleme yapma
            _log.warning("[reconstruct] FPS yok; patch_scale=1 (ölçekleme yok).")

        # ── İşlem döngüsüne başla ─────────────────────────────────────────────
        skipped   = 0
        processed = 0

        _log.info(
            f"[reconstruct] ADIM 0+1+2 başlıyor: "
            f"{n_patches} patch token, {n_raw} RAW düğümü"
        )

        for patch_idx, token_id in enumerate(patch_token_ids):
            entry = self.vocabulary.get(str(token_id))
            if entry is None or entry.get("type") != "PATCH":
                skipped += 1
                continue

            # ── Yerel köşe koordinatları ────────────────────────────────────
            local_nodes_raw: Dict[str, List[float]] = entry.get("local_nodes", {})
            if not local_nodes_raw:
                skipped += 1
                continue

            sorted_keys  = sorted(local_nodes_raw.keys(), key=lambda k: int(k))
            local_coords = np.array(
                [local_nodes_raw[k] for k in sorted_keys], dtype=np.float64
            )  # [patch_n_verts, 3]

            # ── Yerel yüzey bağlantıları ─────────────────────────────────────
            local_elements_raw: Dict[str, Dict] = entry.get("local_elements", {})
            local_face_indices: List[List[int]] = []
            for elem_data in local_elements_raw.values():
                lni: List[int] = elem_data.get("nodes", [])
                if lni:
                    local_face_indices.append(lni)

            # ── Hedef konum: FPS ile seçilen RAW mesh anchor noktası ─────────
            # FPS garantili eşit dağılım; tüm yüzeyin kapsanmasını sağlar.
            # Modulo: n_patches > n_raw durumunda (FPS anchor sayısı < patch sayısı)
            # patch'ler mevcut anchor'lar arasında döngüsel olarak dağıtılır.
            if fps_anchors is not None:
                raw_idx    = int(fps_anchors[patch_idx % len(fps_anchors)])
                target_xyz = raw_pos[raw_idx]
            else:
                target_xyz = np.zeros(3, dtype=np.float64)

            # ── ADIM 0 + 1 + 2: Scale → Rotate → Translate ──────────────────
            self._process_patch(
                local_coords        = local_coords,
                local_face_indices  = local_face_indices,
                target_xyz          = target_xyz,
                patch_scale         = fixed_patch_scale,
                target_patch_radius = target_patch_radius,
                raw_pos             = raw_pos,
                raw_kdtree          = raw_kdtree,
            )
            processed += 1

        _log.info(
            f"[reconstruct] ADIM 0+1+2 tamamlandı: "
            f"{processed} yama işlendi, {skipped} atlandı."
        )

        if not self._all_verts:
            _log.error("[reconstruct] Hiçbir yama için geçerli geometri üretilemedi!")
            return np.empty((0, 3), dtype=np.float64), []

        # ── ADIM 3: Birleştirme yöntemi seç ─────────────────────────────────
        if merge_method in ("poisson", "ball_pivot"):
            _log.info(
                f"[reconstruct] ADIM 3: Open3D {merge_method.upper()} başlıyor …"
            )
            # Open3D normalize uzayda çalışır; denormalizasyon sonra yapılır
            final_verts, final_faces = self.reconstruct_open3d(
                method       = merge_method,
                poisson_depth= poisson_depth,
                ball_radii   = ball_radii,
            )

        else:  # merge_method == "kdtree" (varsayılan)
            # Epsilon: "auto" → dinamik, float → sabit
            if self.epsilon == "auto":
                effective_epsilon = self._compute_auto_epsilon()
            else:
                effective_epsilon = float(self.epsilon)

            _log.info(
                f"[reconstruct] ADIM 3: KDTree Welding "
                f"(epsilon={effective_epsilon:.6f}) başlıyor …"
            )
            final_verts, final_faces = self._weld_vertices(epsilon=effective_epsilon)

            if len(final_verts) == 0:
                _log.error("[reconstruct] Vertex Welding sonrası mesh boş!")
                return final_verts, final_faces

            # İsteğe bağlı: Laplacian Smoothing ─────────────────────────────
            if smooth and smooth_iterations > 0:
                _log.info(
                    f"[reconstruct] Laplacian Smoothing: "
                    f"{smooth_iterations} iter, factor={smooth_factor}"
                )
                final_verts = self._laplacian_smooth(
                    final_verts, final_faces,
                    iterations=smooth_iterations,
                    factor=smooth_factor,
                )

        _log.info(
            f"[reconstruct] Final mesh: "
            f"{len(final_verts)} köşe, {len(final_faces)} yüzey"
        )

        # ── Denormalizasyon: normalize uzay → dünya koordinatları ────────────
        # final_verts normalize uzayda (RAW mesh ile aynı ölçek ~[-2,2]).
        # NAS için world = norm × scale + center
        nas_verts = final_verts * norm_scale + norm_center
        _log.info(
            f"[reconstruct] NAS denormalizasyon: "
            f"scale={norm_scale:.4f}, center={norm_center.tolist()}"
        )

        self._export_nas(nas_verts, final_faces, output_path)

        # Görselleştirme için normalize uzay koordinatlarını döndür ───────────
        return final_verts, final_faces


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
    pos        = data.pos.cpu().numpy().astype(np.float32)
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
    parser    = ParserNas()
    mesh_data = parser.parse_file(nas_path)
    nodes     = mesh_data["nodes"]
    elements  = mesh_data["elements"]
    node_ids  = sorted(nodes.keys())
    node_idx  = {nid: i for i, nid in enumerate(node_ids)}
    vertices  = np.array([nodes[nid] for nid in node_ids], dtype=np.float32)
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
        description="FAZ 5 v4: Ölçeklendirilmiş Yama + Gelişmiş Birleştirme",
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
                   default=str(PROJECT_ROOT / "output_generated_v4.nas"),
                   help="Çıktı NAS dosyası yolu.")
    p.add_argument("--no-vis", action="store_true",
                   help="PyVista görselleştirmeyi devre dışı bırak.")
    p.add_argument("--max-gen-len", type=int, default=DEFAULT_MAX_GEN_LEN,
                   help="Maksimum üretilecek token sayısı.")
    p.add_argument("--repetition-penalty", type=float, default=DEFAULT_REPETITION_PENALTY,
                   help="Tekrar cezası (1.0=devre dışı, >1.0=çeşitliliği artır).")
    p.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE,
                   help="Logit sıcaklık katsayısı (<1.0=keskin, >1.0=rastgele).")
    p.add_argument("--seed", type=int, default=None,
                   help="Rastgele PT dosyası seçimi için seed.")

    # ── Yeni v4 Parametreleri ─────────────────────────────────────────────────
    p.add_argument(
        "--patch-scale-factor", type=float, default=None,
        help=(
            "Yama büyütme katsayısı (sabit çarpan modu). "
            "None (varsayılan) → her yama, RAW mesh bbox / √n_patches "
            "formülüyle otomatik hedef yarıçapa normalleştirilir. "
            "Örn: 5.0 → tüm yamalara centered × 5 uygulanır."
        ),
    )
    p.add_argument(
        "--merge-method", type=str, default="kdtree",
        choices=["kdtree", "poisson", "ball_pivot"],
        help=(
            "Yüzey birleştirme yöntemi:\n"
            "  kdtree    – KDTree Welding + Laplacian Smooth (hızlı)\n"
            "  poisson   – Open3D Poisson Reconstruction (pürüzsüz, kapalı)\n"
            "  ball_pivot– Open3D Ball Pivoting (açık yüzeyler için)"
        ),
    )
    p.add_argument(
        "--epsilon", type=str, default="auto",
        help=(
            "Vertex welding mesafe toleransı (kdtree yöntemi).\n"
            "'auto' (varsayılan) → yama yarıçapına göre dinamik hesap.\n"
            "Sayı: örn. 0.1 → sabit tolerans."
        ),
    )
    p.add_argument("--no-smooth", action="store_true",
                   help="Laplacian smoothing'i devre dışı bırak (kdtree yöntemi).")
    p.add_argument("--smooth-iterations", type=int, default=DEFAULT_SMOOTH_ITERS,
                   help="Laplacian smoothing iterasyon sayısı.")
    p.add_argument("--smooth-factor", type=float, default=DEFAULT_SMOOTH_FACTOR,
                   help="Laplacian smoothing katsayısı.")
    p.add_argument("--open3d-depth", type=int, default=DEFAULT_OPEN3D_DEPTH,
                   help="Open3D Poisson rekonstrüksiyon derinliği (6–10 önerilir).")
    return p.parse_args()


# =========================================================================== #
#  Ana Giriş Noktası                                                            #
# =========================================================================== #

if __name__ == "__main__":
    args = _parse_args()

    # epsilon: "auto" string veya float
    epsilon_value: float | str
    if args.epsilon.lower() == "auto":
        epsilon_value = "auto"
    else:
        try:
            epsilon_value = float(args.epsilon)
        except ValueError:
            _log.error(f"--epsilon için geçersiz değer: '{args.epsilon}'. 'auto' veya sayı girin.")
            sys.exit(1)

    print("\n" + "=" * 70)
    print("  MeshCrafter zbggGraphNet3 — FAZ 5 v4: Ölçeklendirilmiş Yama")
    print("=" * 70)

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
        rng     = random.Random(args.seed)
        pt_path = rng.choice(pt_files)
        _log.info(f"Rastgele seçilen PT dosyası: {pt_path.name}")

    file_stem = pt_path.stem
    print(f"\n[Giriş]                {pt_path}")
    print(f"[Çıktı]                {args.output}")
    print(f"[Stem]                 {file_stem}")
    print(f"[Merge Method]         {args.merge_method}")
    patch_sf_display = args.patch_scale_factor if args.patch_scale_factor else "auto (bbox/√n)"
    print(f"[Patch Scale Factor]   {patch_sf_display}")
    print(f"[Epsilon]              {epsilon_value}")
    print(f"[Repetition Penalty]   {args.repetition_penalty}")
    print(f"[Temperature]          {args.temperature}")
    print()

    # ── 2. RAW mesh verisini yükle ────────────────────────────────────────────
    _log.info("RAW mesh yükleniyor …")
    raw_data: Data = torch.load(pt_path, map_location="cpu", weights_only=False)
    _log.info(f"  x.shape         : {list(raw_data.x.shape)}")
    _log.info(f"  edge_index.shape: {list(raw_data.edge_index.shape)}")
    if hasattr(raw_data, "pos") and raw_data.pos is not None:
        _log.info(f"  pos.shape       : {list(raw_data.pos.shape)}")

    # ── 3. trimesh nesnesi oluştur (yüzey normal sorgulama için) ─────────────
    raw_trimesh: Optional["trimesh.Trimesh"] = None
    if _TRIMESH_AVAILABLE and hasattr(raw_data, "face") and raw_data.face is not None:
        try:
            raw_pos_np  = raw_data.pos.cpu().numpy()
            face_np     = raw_data.face.cpu().numpy().T
            raw_trimesh = trimesh.Trimesh(vertices=raw_pos_np, faces=face_np, process=False)
            raw_trimesh.fix_normals()
            _log.info(f"  trimesh         : {len(raw_trimesh.faces)} yüzey (yüzey normal modu)")
        except Exception as exc:
            _log.warning(f"  trimesh oluşturulamadı ({exc}); KDTree+PCA modu kullanılacak.")

    # ── 4. MeshGenerator ─────────────────────────────────────────────────────
    _log.info("\nMeshGenerator başlatılıyor …")
    generator = MeshGenerator(
        checkpoint_path=args.checkpoint,
        vocab_path     =args.vocab,
        norm_csv_path  =args.norm_csv,
    )

    # ── 5. Token Dizisi Üret ──────────────────────────────────────────────────
    _log.info("\nToken dizisi üretiliyor …")
    t0 = time.time()
    token_ids: List[int] = generator.generate_sequence(
        raw_mesh_data    =raw_data,
        max_len          =args.max_gen_len,
        repetition_penalty=args.repetition_penalty,
        temperature      =args.temperature,
    )
    gen_time = time.time() - t0

    patch_tokens  = [t for t in token_ids if t not in SPECIAL_TOKENS]
    special_count = len(token_ids) - len(patch_tokens)
    token_counter = Counter(patch_tokens)

    print(f"\n{'─'*62}")
    print("  ★  ÜRETİM İSTATİSTİKLERİ")
    print(f"{'─'*62}")
    print(f"  Toplam üretilen token sayısı  : {len(token_ids)}")
    print(f"  Özel token (START/END/PAD/UNK): {special_count}")
    print(f"  PATCH token sayısı            : {len(patch_tokens)}")
    print(f"  Benzersiz PATCH token türü    : {len(token_counter)}")
    print(f"  Üretim süresi                 : {gen_time:.3f} s")
    if token_counter:
        print(f"\n  Token Frekans Dağılımı (İlk 10):")
        max_freq = max(token_counter.values())
        for tid, freq in token_counter.most_common(10):
            bar = "█" * min(int(freq * 30 / max(max_freq, 1)), 30)
            print(f"    Token {tid:3d} : {freq:5d}  {bar}")
    print(f"{'─'*62}\n")

    # ── 6. MeshReconstructor v4 ───────────────────────────────────────────────
    _log.info("MeshReconstructor v4 başlatılıyor …")
    reconstructor = MeshReconstructor(
        vocabulary         =generator.vocabulary,
        norm_params        =generator.norm_params,
        raw_mesh_trimesh   =raw_trimesh,
        epsilon            =epsilon_value,
        patch_scale_factor =args.patch_scale_factor,
    )

    _log.info("v4 Mesh yeniden inşa ediliyor …")
    t1 = time.time()
    final_verts, final_faces = reconstructor.reconstruct(
        token_ids        =token_ids,
        raw_mesh_data    =raw_data,
        output_path      =args.output,
        file_stem        =file_stem,
        merge_method     =args.merge_method,
        smooth           =not args.no_smooth,
        smooth_iterations=args.smooth_iterations,
        smooth_factor    =args.smooth_factor,
        poisson_depth    =args.open3d_depth,
    )
    recon_time = time.time() - t1

    # ── 7. Özet Rapor ─────────────────────────────────────────────────────────
    print(f"\n{'─'*62}")
    print("  ★  YENİDEN İNŞA RAPORU  (v4)")
    print(f"{'─'*62}")
    print(f"  Merge Method          : {args.merge_method}")
    print(f"  Patch Scale Factor    : {patch_sf_display}")
    print(f"  Düğüm sayısı          : {len(final_verts)}")
    print(f"  Yüzey sayısı          : {len(final_faces)}")
    print(f"  Yeniden inşa süresi   : {recon_time:.3f} s")
    print(f"  Çıktı dosyası (NAS)   : {Path(args.output).resolve()}")
    print(f"{'─'*62}")
    print(f"\n  [TAMAMLANDI] Toplam süre: {gen_time + recon_time:.3f} s")
    print("=" * 70 + "\n")

    # ── 8. PyVista Görselleştirme ─────────────────────────────────────────────
    if args.no_vis:
        _log.info("Görselleştirme '--no-vis' ile devre dışı bırakıldı.")
    elif not _PYVISTA_AVAILABLE:
        _log.warning("PyVista kurulu değil; pip install pyvista")
    else:
        _log.info("\nPyVista görselleştirme hazırlanıyor …")

        # Normalizasyon parametreleri (RAW mesh'i world-space'e çevirmek için)
        _vis_scale  = 1.0
        _vis_center = np.zeros(3, dtype=np.float64)
        if generator.norm_params:
            _p = generator.norm_params.get(file_stem)
            if _p:
                _vis_scale  = float(_p["scale_max_radius"])
                _vis_center = np.array(
                    [_p["center_x"], _p["center_y"], _p["center_z"]], dtype=np.float64
                )

        # RAW mesh — normalize uzay → dünya koordinatları (REFINED ile aynı ölçek)
        if hasattr(raw_data, "pos") and raw_data.pos is not None:
            raw_pos_world = (
                raw_data.pos.cpu().numpy().astype(np.float64) * _vis_scale + _vis_center
            ).astype(np.float32)
            edge_index = raw_data.edge_index.cpu().numpy()
            _raw_edges: set = set()
            for _k in range(edge_index.shape[1]):
                _u, _v = int(edge_index[0, _k]), int(edge_index[1, _k])
                _raw_edges.add((min(_u, _v), max(_u, _v)))
            _raw_lines: List[int] = []
            for _u, _v in _raw_edges:
                _raw_lines.extend([2, _u, _v])
            pv_raw = pv.PolyData()
            pv_raw.points = raw_pos_world
            if _raw_lines:
                pv_raw.lines = np.array(_raw_lines, dtype=np.int_)
            _log.info(f"  RAW        : {pv_raw.n_points} nokta (dünya koordinatları)")
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
                    f"  REFINED    : {pv_refined.n_points} nokta, "
                    f"{pv_refined.n_cells} element"
                )
            except Exception as exc:
                _log.warning(f"  REFINED NAS yüklenemedi ({exc}).")
                pv_refined = pv.PolyData()
        else:
            _log.warning(f"  REFINED NAS bulunamadı: {refined_nas}")
            pv_refined = pv.PolyData()

        # PREDICTION mesh — NAS dosyasından yükle (dünya koordinatları = REFINED ile aynı ölçek)
        # Eski yaklaşım (final_verts normalize uzayda) kamerayı REFINED'a göre çok küçük
        # gösteriyordu. Kaydedilen NAS dosyasını doğrudan okuyarak ölçek uyumu sağlanır.
        pv_pred_path = Path(args.output)
        if pv_pred_path.is_file():
            try:
                pv_pred = _load_nas_as_pyvista(pv_pred_path)
                _log.info(
                    f"  PREDICTION : {pv_pred.n_points} nokta, "
                    f"{pv_pred.n_cells} element (NAS dünya koordinatları)"
                )
            except Exception as exc:
                _log.warning(f"  PREDICTION NAS yüklenemedi ({exc}); normalize uzay kullanılıyor.")
                pv_pred = _build_pyvista_from_arrays(final_verts, final_faces)
        else:
            _log.warning("  PREDICTION NAS dosyası bulunamadı; normalize uzay kullanılıyor.")
            pv_pred = _build_pyvista_from_arrays(final_verts, final_faces)

        if pv_pred.n_points == 0:
            _log.error(
                "  PREDICTION mesh boş! "
                "Token üretimi veya sözlük eşleşmesi başarısız."
            )
        else:
            b = pv_pred.bounds
            _log.info(
                f"  Sınırlar: "
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
                f"MeshCrafter v4 – {file_stem}  |  RAW | REFINED | PREDICTION"
            ),
        )
