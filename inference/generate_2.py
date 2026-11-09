"""
inference/generate_2.py  –  FAZ 5 v3: Geometrik Hizalama ile Mesh Yeniden İnşa
================================================================================
Mevcut generate.py'nin üzerine inşa edilmiş, üç temel geometrik düzeltmeyi
içeren geliştirilmiş yeniden inşa scripti.

Yeni MeshReconstructor v3 Pipeline:
    ADIM 1 – Yüzey Normaline Göre Döndürme
        trimesh.proximity.closest_point → yüzey normali → Rodrigues Rotasyon
    ADIM 2 – Öteleme + Denormalizasyon + Havuzlama
        Scale → Rotate → Translate sırasıyla 3D yerleştirme
    ADIM 3 – KDTree ile Vertex Welding
        cKDTree.query_pairs → Union-Find birleştirme → Laplacian Smoothing

Bağımlılıklar:
    pip install torch torch_geometric trimesh scipy pyvista

Kullanım:
    python inference/generate_2.py
    python inference/generate_2.py --pt-file /path/to/mesh.pt --output output_v3.nas
    python inference/generate_2.py --epsilon 0.05 --smooth-iterations 3
"""

from __future__ import annotations

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


# =========================================================================== #
#  Loglama                                                                      #
# =========================================================================== #

def _build_logger(name: str = "generate_v3") -> logging.Logger:
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
DEFAULT_EPSILON            = 0.005  # Vertex welding mesafe toleransı
                                    # Normalize uzayda (~[-1,1]) yama iç kenar
                                    # mesafeleri 0.01–0.05 → epsilon bu değerin
                                    # altında olmalı, yoksa intra-patch köşeler
                                    # kaynaşıp tüm yüzeyler dejenere olur.
DEFAULT_REPETITION_PENALTY = 1.5
DEFAULT_TEMPERATURE        = 1.0
DEFAULT_SMOOTH_ITERS       = 2
DEFAULT_SMOOTH_FACTOR      = 0.5


# =========================================================================== #
#  Normalizasyon Yardımcıları                                                   #
# =========================================================================== #

def load_normalization_params(csv_path: str) -> Dict[str, Dict[str, float]]:
    """normalize_parameter.csv → stem: {center_x/y/z, scale_max_radius}"""
    params: Dict[str, Dict[str, float]] = {}
    csv_file = Path(csv_path)
    if not csv_file.is_file():
        _log.warning(f"Normalizasyon CSV bulunamadı: {csv_path}")
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
    _log.info(f"Normalizasyon parametreleri: {len(params)} kayıt")
    return params


# =========================================================================== #
#  SINIF 1: MeshGenerator  (generate.py ile aynı — değiştirilmedi)             #
# =========================================================================== #

class MeshGenerator:
    """
    Eğitilmiş Graph2Seq modelini kullanarak RAW mesh grafiğinden token dizisi üretir.

    repetition_penalty  : Greedy collapse önleyici logit cezası (>1.0 aktif).
    temperature         : Logit ölçekleme (<1.0 keskin, >1.0 rastgele).
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

    @staticmethod
    def _load_checkpoint(path: str) -> dict:
        ckpt_path = Path(path)
        if not ckpt_path.is_file():
            raise FileNotFoundError(f"Checkpoint bulunamadı: {path}")
        _log.info(f"[MeshGenerator] Checkpoint: {path}")
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
        """Otoregresif greedy decoding — repetition_penalty ile greedy collapse önlenir."""
        _log.info("[generate_sequence] Başlatılıyor …")
        t_start = time.time()

        data = raw_mesh_data.to(self.device)
        batch_vector = torch.zeros(data.x.size(0), dtype=torch.long, device=self.device)

        context: torch.Tensor = self.model.encoder(
            data.x, data.edge_index, batch_vector
        )
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
                [generated, torch.tensor([[next_token_id]], dtype=torch.long,
                                          device=self.device)],
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
#  SINIF 2: MeshReconstructor v3                                                #
# =========================================================================== #

class MeshReconstructor:
    """
    Token ID dizisini geometrik olarak hizalanmış, kaynaştırılmış 3D mesh'e
    dönüştürür.

    v3 Üç Adımlı Pipeline
    ─────────────────────
    ADIM 1 — Yüzey Normaline Göre Döndürme
        raw_mesh (trimesh) üzerinde closest_point sorgusu → yüzey normali
        Rodrigues formülü ile Rotasyon Matrisi: Z-eksen → hedef normal
        Köşeler yama merkezi etrafında döndürülür

    ADIM 2 — Scale → Rotate → Translate + Denormalizasyon
        Dönüşüm sırası: local_coords → (× scale) → R × → (+ world_target)
        Denormalizasyon: world = normalized × scale + center
        Global vertex/face havuzuna ekleme (index ofseti güncellenir)

    ADIM 3 — KDTree ile Vertex Welding
        cKDTree.query_pairs(r=epsilon) → yakın çift tespiti
        Union-Find → bağlı bileşen grupları
        Her grup için ortalama konum → yeniden indeksleme
        Bozulan (dejenere) yüzeyler ayıklanır
        İsteğe bağlı: Laplacian Smoothing

    Parametre
    ---------
    vocabulary        : ai_vocabulary.json içeriği (dict)
    norm_params       : normalize_parameter.csv'den yüklenen parametreler
    raw_mesh_trimesh  : trimesh.Trimesh objesi (yüzey normal sorguları için)
                        None ise KDTree+PCA tabanlı tahmine geri dönülür
    epsilon           : Vertex welding mesafe toleransı (varsayılan 0.05)
    """

    _DEFAULT_PATCH_NORMAL = np.array([0.0, 0.0, 1.0], dtype=np.float64)

    def __init__(
        self,
        vocabulary: Dict,
        norm_params: Optional[Dict[str, Dict[str, float]]] = None,
        raw_mesh_trimesh: Optional["trimesh.Trimesh"] = None,
        epsilon: float = DEFAULT_EPSILON,
    ) -> None:
        self.vocabulary         = vocabulary
        self.norm_params        = norm_params or {}
        self.raw_mesh_trimesh   = raw_mesh_trimesh
        self.epsilon            = epsilon

        # Global havuz – reconstruct() başında sıfırlanır
        self._all_verts:   List[np.ndarray] = []
        self._all_faces:   List[List[int]]  = []
        self._vert_offset: int              = 0

    # ── Statik Araçlar ───────────────────────────────────────────────────────

    @staticmethod
    def build_trimesh_from_data(
        raw_data: Data,
    ) -> Optional["trimesh.Trimesh"]:
        """
        PyG Data → trimesh.Trimesh dönüştürücü.

        raw_data.face mevcutsa triangulated mesh oluşturur.
        Yoksa None döndürür; çağıran  KDTree+PCA'ya geçer.
        """
        if not _TRIMESH_AVAILABLE:
            _log.warning("[build_trimesh] trimesh kurulu değil; pip install trimesh")
            return None

        if not (hasattr(raw_data, "pos") and raw_data.pos is not None):
            _log.warning("[build_trimesh] raw_data.pos yok; trimesh oluşturulamadı.")
            return None

        pos = raw_data.pos.cpu().numpy().astype(np.float64)

        if hasattr(raw_data, "face") and raw_data.face is not None:
            faces = raw_data.face.cpu().numpy().T  # [n_faces, 3]
            try:
                tm = trimesh.Trimesh(vertices=pos, faces=faces, process=True)
                tm.fix_normals()
                _log.info(
                    f"[build_trimesh] Trimesh oluşturuldu: "
                    f"{len(pos)} köşe, {len(faces)} yüzey"
                )
                return tm
            except Exception as exc:
                _log.debug(f"[build_trimesh] Trimesh oluşturma hatası: {exc}")

        _log.info(
            "[build_trimesh] face verisi yok; "
            "yüzey normalleri için KDTree+PCA kullanılacak."
        )
        return None

    @staticmethod
    def _rotation_matrix_from_vectors(
        src: np.ndarray,
        dst: np.ndarray,
    ) -> np.ndarray:
        """
        Rodrigues formülü: src birim vektörünü dst'ye döndüren 3×3 matris.

        Özel durumlar
        -------------
        Paralel  (dot ≈ +1) → birim matris (döndürme yok)
        Antiparalel (dot ≈ -1) → dik eksende 180° rotasyon
        """
        src = src / (np.linalg.norm(src) + 1e-12)
        dst = dst / (np.linalg.norm(dst) + 1e-12)

        v = np.cross(src, dst)
        c = float(np.dot(src, dst))
        s = np.linalg.norm(v)

        if s < 1e-10:
            if c > 0.0:
                return np.eye(3, dtype=np.float64)
            # 180° döndürme: src'ye dik arbitrary eksen
            perp = (np.array([1.0, 0.0, 0.0])
                    if abs(src[0]) < 0.9
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

    # ── ADIM 1/2 Yardımcısı: Yüzey Normali Sorgulama ─────────────────────────

    def _surface_normal_at(
        self,
        target_xyz: np.ndarray,
        raw_pos: Optional[np.ndarray] = None,
        raw_kdtree: Optional[cKDTree] = None,
    ) -> np.ndarray:
        """
        target_xyz konumundaki tahmini yüzey normalini döndürür.

        Önce raw_mesh_trimesh ile trimesh.proximity.closest_point dener.
        trimesh yoksa KDTree+PCA ile komşulardan normal tahmin eder.
        """
        # ── Yol 1: trimesh proximity ─────────────────────────────────────────
        if self.raw_mesh_trimesh is not None and _TRIMESH_AVAILABLE:
            try:
                query = target_xyz.reshape(1, 3)
                _, _, tri_ids = trimesh.proximity.closest_point(
                    self.raw_mesh_trimesh, query
                )
                normal = self.raw_mesh_trimesh.face_normals[tri_ids[0]].astype(
                    np.float64
                )
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
            normal = eigvecs[:, 0]
            n_len = np.linalg.norm(normal)
            if n_len > 1e-12:
                return normal / n_len

        # Fallback: Z-ekseni
        return self._DEFAULT_PATCH_NORMAL.copy()

    # ── ADIM 1 + 2: Tek Yama İşleme ──────────────────────────────────────────

    def _process_patch(
        self,
        local_coords: np.ndarray,           # [n_verts, 3] — normalize yerel uzay
        local_face_indices: List[List[int]], # yama içi yüzey bağlantıları
        target_xyz: np.ndarray,             # RAW mesh anchor (normalize uzay)
        raw_pos: Optional[np.ndarray] = None,
        raw_kdtree: Optional[cKDTree] = None,
    ) -> None:
        """
        Bir yamayı NORMALİZE uzayda Rotate → Translate sırası ile işler
        ve global havuza ekler. Denormalizasyon NAS export aşamasında yapılır.

        Dönüşüm Sırası
        ──────────────
        Bu adım normalize uzayda çalışır (scale = 1).
        Matematiksel eşdeğerlik:
            (R × centered + anchor) × scale + center
          = R × (centered × scale) + anchor × scale + center   [Scale→Rotate→Translate]
        Yani dünya uzayına geçişten önce veya sonra yapılması aynı sonucu verir.

        1. Yama merkezi hesaplanır ve lokal koordinatlar merkeze çekilir
        2. target_xyz konumundaki yüzey normali sorgulanır (trimesh / KDTree+PCA)
        3. Rotasyon matrisi hesaplanır: [0,0,1] → surface_normal  (Rodrigues)
        4. Merkezlenmiş koordinatlar döndürülür
        5. target_xyz'e ötelenir → normalize uzay koordinatı
        """
        if len(local_coords) == 0:
            return

        # ── 1. Yama merkezini origin'e taşı ──────────────────────────────────
        patch_center = local_coords.mean(axis=0)
        centered     = local_coords - patch_center   # origin'de, normalize ölçek

        # ── 2. Hedef konumun yüzey normalini sorgula ─────────────────────────
        surface_normal = self._surface_normal_at(
            target_xyz, raw_pos=raw_pos, raw_kdtree=raw_kdtree
        )

        # ── 3. Rodrigues: [0,0,1] → surface_normal ───────────────────────────
        rot_mat = self._rotation_matrix_from_vectors(
            self._DEFAULT_PATCH_NORMAL, surface_normal
        )

        # ── 4. Rotate (normalize ölçekte) ────────────────────────────────────
        rotated = (rot_mat @ centered.T).T           # [n_verts, 3]

        # ── 5. Translate → normalize uzay anchor'una ─────────────────────────
        norm_coords = rotated + target_xyz           # görselleştirme ile aynı uzay

        # ── Global havuza ekle (yüzey indeksleri ofseti güncelle) ─────────────
        offset = self._vert_offset
        self._all_verts.append(norm_coords)
        for face in local_face_indices:
            self._all_faces.append([offset + vi for vi in face])
        self._vert_offset += len(norm_coords)

    # ── ADIM 3a: Vertex Welding ───────────────────────────────────────────────

    def _weld_vertices(
        self,
    ) -> Tuple[np.ndarray, List[List[int]]]:
        """
        KDTree tabanlı Vertex Welding.

        Algoritma
        ---------
        1. Tüm köşeleri birleştir → numpy dizisi [N, 3]
        2. cKDTree oluştur
        3. query_pairs(r=epsilon) → birbirine yakın köşe çiftleri
        4. Union-Find ile bağlı bileşenleri grupla
        5. Her grup için ortalama konum hesapla → yeni köşe dizisi
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
            f"[weld_vertices] Welding başlıyor: {n} köşe, "
            f"{len(all_f)} yüzey, epsilon={self.epsilon}"
        )

        # ── KDTree + yakın çift tespiti ───────────────────────────────────────
        tree  = cKDTree(all_v)
        pairs = tree.query_pairs(r=self.epsilon)
        _log.info(f"[weld_vertices] {len(pairs)} yakın çift bulundu.")

        # ── Union-Find (path compression + union by rank) ─────────────────────
        parent = list(range(n))
        rank   = [0] * n

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]   # path halving
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

        new_idx_map: np.ndarray = np.full(n, -1, dtype=np.int64)
        new_verts_list: List[np.ndarray] = []

        for root, members in groups.items():
            new_idx = len(new_verts_list)
            avg_pos = all_v[members].mean(axis=0)
            new_verts_list.append(avg_pos)
            for m in members:
                new_idx_map[m] = new_idx

        # ── Yüzey indekslerini güncelle ve dejenere yüzeyleri kaldır ──────────
        new_faces: List[List[int]] = []
        removed = 0
        for face in all_f:
            remapped = [int(new_idx_map[vi]) for vi in face]
            if len(set(remapped)) >= 3:
                new_faces.append(remapped)
            else:
                removed += 1

        new_verts = np.array(new_verts_list, dtype=np.float64)

        reduction_pct = 100.0 * (n - len(new_verts)) / max(n, 1)
        log_fn = _log.warning if reduction_pct > 80 else _log.info
        log_fn(
            f"[weld_vertices] Tamamlandı: "
            f"{n} → {len(new_verts)} köşe "
            f"({reduction_pct:.1f}% azalma), "
            f"{len(new_faces)} yüzey kaldı "
            f"({removed} dejenere kaldırıldı)."
        )
        if reduction_pct > 80:
            _log.warning(
                f"[weld_vertices] ⚠  Aşırı welding tespit edildi "
                f"(%{reduction_pct:.0f} köşe kaybı)! "
                f"Epsilon={self.epsilon} çok büyük olabilir. "
                f"--epsilon 0.001 ile tekrar deneyin."
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
        Laplacian Smoothing: köşeleri komşularının ortalamasına doğru iter.

        Her iterasyonda:  v_new = v + factor × (mean(neighbors) − v)

        factor ∈ (0, 1) → 1.0 tam komşu ortalaması (agresif düzleştirme)
                           0.3 hafif düzleştirme (sivrilikleri giderir)
        """
        if len(vertices) == 0 or len(faces) == 0 or iterations <= 0:
            return vertices

        n = len(vertices)
        verts = vertices.copy()

        # Komşuluk yapısı
        adjacency: List[List[int]] = [[] for _ in range(n)]
        for face in faces:
            for i, vi in enumerate(face):
                for vj in face:
                    if vi != vj:
                        adjacency[vi].append(vj)

        # Tekrarlı komşulardan kaynaklanan ağırlık gürültüsünü azalt
        adj_sets = [list(set(nbrs)) for nbrs in adjacency]

        for it in range(iterations):
            new_verts = verts.copy()
            for i in range(n):
                nbrs = adj_sets[i]
                if nbrs:
                    avg = verts[nbrs].mean(axis=0)
                    new_verts[i] = verts[i] + factor * (avg - verts[i])
            verts = new_verts

        _log.info(f"[laplacian_smooth] {iterations} iterasyon tamamlandı.")
        return verts

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
            f.write("$ MeshCrafter zbggGraphNet3 – FAZ 5 v3 Çıktısı\n")
            f.write(f"$ Düğüm sayısı: {len(vertices)}\n")
            f.write(f"$ Yüzey sayısı: {len(faces)}\n")
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
                    # Fan triangülasyon
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
        smooth: bool = True,
        smooth_iterations: int = DEFAULT_SMOOTH_ITERS,
        smooth_factor: float = DEFAULT_SMOOTH_FACTOR,
    ) -> Tuple[np.ndarray, List[List[int]]]:
        """
        Token ID dizisini 3 adımlı pipeline'dan geçirerek NAS dosyasına dönüştürür.

        1. Her patch token için: ADIM 1 (Rotate) + ADIM 2 (Translate) — normalize uzayda
        2. Tüm yamalar işlendikten sonra: ADIM 3 (Vertex Weld + Smooth) — normalize uzayda
        3. Denormalizasyon → NAS dışa aktarma (world koordinatlar)

        Dönüş değerleri normalize uzaydadır (RAW mesh ile aynı ölçek, viz için):
            final_verts : [N, 3] float64  — normalize uzay (~[-2, 2])
            final_faces : List[List[int]]
        """
        # ── Durum sıfırlama ────────────────────────────────────────────────────
        self._all_verts   = []
        self._all_faces   = []
        self._vert_offset = 0

        # ── RAW mesh verilerini hazırla ────────────────────────────────────────
        raw_pos: Optional[np.ndarray] = None
        raw_kdtree: Optional[cKDTree] = None

        if hasattr(raw_mesh_data, "pos") and raw_mesh_data.pos is not None:
            raw_pos = raw_mesh_data.pos.cpu().numpy().astype(np.float64)
            raw_kdtree = cKDTree(raw_pos)
            _log.info(f"[reconstruct] RAW mesh: {raw_pos.shape[0]} düğüm")

        # ── Normalizasyon parametrelerini yükle ───────────────────────────────
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

        # ── Sadece PATCH token'larını filtrele ────────────────────────────────
        patch_token_ids = [t for t in token_ids if t not in SPECIAL_TOKENS]
        n_patches = len(patch_token_ids)
        n_raw     = raw_pos.shape[0] if raw_pos is not None else 0

        skipped   = 0
        processed = 0

        _log.info(
            f"[reconstruct] ADIM 1+2 başlıyor: "
            f"{n_patches} patch token, {n_raw} RAW düğümü"
        )

        for patch_idx, token_id in enumerate(patch_token_ids):
            entry = self.vocabulary.get(str(token_id))
            if entry is None or entry.get("type") != "PATCH":
                skipped += 1
                continue

            # Yerel köşe koordinatları —————————————————————————————————————
            local_nodes_raw: Dict[str, List[float]] = entry.get("local_nodes", {})
            if not local_nodes_raw:
                skipped += 1
                continue

            sorted_keys  = sorted(local_nodes_raw.keys(), key=lambda k: int(k))
            local_coords = np.array(
                [local_nodes_raw[k] for k in sorted_keys], dtype=np.float64
            )  # [patch_n_verts, 3]

            # Yerel yüzey bağlantıları —————————————————————————————————————
            local_elements_raw: Dict[str, Dict] = entry.get("local_elements", {})
            local_face_indices: List[List[int]] = []
            for elem_data in local_elements_raw.values():
                lni: List[int] = elem_data.get("nodes", [])
                if lni:                      # en az 1 düğüm yeterli; dejenere kontrol weld'de yapılır
                    local_face_indices.append(lni)

            # Hedef konum (X, Y, Z): yamayı RAW mesh yüzeyine hizala ─────────
            # Modelin patch_idx-inci patch'i için ilgili RAW mesh bölgesi
            if raw_pos is not None and n_raw > 0 and n_patches > 0:
                raw_idx    = min(int(patch_idx * n_raw / n_patches), n_raw - 1)
                target_xyz = raw_pos[raw_idx]   # normalize uzayda anchor
            else:
                target_xyz = np.zeros(3, dtype=np.float64)

            # ADIM 1 + 2: Rotate + Translate (normalize uzayda) ───────────────
            self._process_patch(
                local_coords       = local_coords,
                local_face_indices = local_face_indices,
                target_xyz         = target_xyz,
                raw_pos            = raw_pos,
                raw_kdtree         = raw_kdtree,
            )
            processed += 1

        _log.info(
            f"[reconstruct] ADIM 1+2 tamamlandı: "
            f"{processed} yama işlendi, {skipped} atlandı."
        )

        if not self._all_verts:
            _log.error("[reconstruct] Hiçbir yama için geçerli geometri üretilemedi!")
            return np.empty((0, 3), dtype=np.float64), []

        # ADIM 3: Vertex Welding ───────────────────────────────────────────────
        _log.info("[reconstruct] ADIM 3: Vertex Welding başlıyor …")
        final_verts, final_faces = self._weld_vertices()

        if len(final_verts) == 0:
            _log.error("[reconstruct] Vertex Welding sonrası mesh boş!")
            return final_verts, final_faces

        # İsteğe bağlı: Laplacian Smoothing ───────────────────────────────────
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

        # NAS için denormalize et: world = normalize × scale + center ──────────
        nas_verts = final_verts * norm_scale + norm_center
        _log.info(
            f"[reconstruct] NAS denormalizasyon: "
            f"scale={norm_scale:.4f}, center={norm_center.tolist()}"
        )
        self._export_nas(nas_verts, final_faces, output_path)

        # Görselleştirme için normalize uzay koordinatlarını döndür ─────────────
        # (RAW mesh ve REFINED mesh ile aynı koordinat uzayı → kamera uyumu)
        return final_verts, final_faces


# =========================================================================== #
#  PyVista Yardımcıları                                                         #
# =========================================================================== #

def _build_pyvista_from_arrays(
    vertices: np.ndarray,
    faces: List[List[int]],
) -> "pv.PolyData":
    if not _PYVISTA_AVAILABLE:
        raise ImportError("pip install pyvista")
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
    if not _PYVISTA_AVAILABLE:
        raise ImportError("pip install pyvista")
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
    if not _PYVISTA_AVAILABLE:
        raise ImportError("pip install pyvista")
    parser   = ParserNas()
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
        description="FAZ 5 v3: Graf2Seq Çıkarım ve Geometrik Mesh Yeniden İnşa",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--pt-file", type=str, default=None)
    p.add_argument(
        "--checkpoint",
        type=str,
        default=str(PROJECT_ROOT / "checkpoints" / "best_model.pth"),
    )
    p.add_argument(
        "--vocab",
        type=str,
        default=str(PROJECT_ROOT / "data_vocabulary" / "ai_vocabulary.json"),
    )
    p.add_argument(
        "--norm-csv",
        type=str,
        default=r"<local-path>",
    )
    p.add_argument(
        "--pt-dir",
        type=str,
        default=r"<local-path>",
    )
    p.add_argument(
        "--refined-dir",
        type=str,
        default=r"<local-path>",
    )
    p.add_argument(
        "--output",
        type=str,
        default=str(PROJECT_ROOT / "output_generated_v3.nas"),
    )
    p.add_argument("--no-vis",            action="store_true")
    p.add_argument("--max-gen-len",       type=int,   default=DEFAULT_MAX_GEN_LEN)
    p.add_argument("--repetition-penalty",type=float, default=DEFAULT_REPETITION_PENALTY)
    p.add_argument("--temperature",       type=float, default=DEFAULT_TEMPERATURE)
    p.add_argument("--epsilon",           type=float, default=DEFAULT_EPSILON,
                   help="Vertex welding mesafe toleransı")
    p.add_argument("--no-smooth",         action="store_true",
                   help="Laplacian smoothing'i devre dışı bırak")
    p.add_argument("--smooth-iterations", type=int,   default=DEFAULT_SMOOTH_ITERS)
    p.add_argument("--smooth-factor",     type=float, default=DEFAULT_SMOOTH_FACTOR)
    p.add_argument("--seed",              type=int,   default=None)
    return p.parse_args()


# =========================================================================== #
#  Ana Giriş Noktası                                                            #
# =========================================================================== #

if __name__ == "__main__":
    args = _parse_args()

    print("\n" + "=" * 68)
    print("  MeshCrafter zbggGraphNet3 — FAZ 5 v3: Geometrik Hizalama")
    print("=" * 68)

    # ── 1. Giriş PT dosyasını belirle ────────────────────────────────────────
    if args.pt_file:
        pt_path = Path(args.pt_file)
        if not pt_path.is_file():
            _log.error(f"PT dosyası bulunamadı: {pt_path}")
            sys.exit(1)
    else:
        pt_dir = Path(args.pt_dir)
        if not pt_dir.is_dir():
            _log.error(f"PT dizini bulunamadı: {pt_dir}")
            sys.exit(1)
        pt_files = sorted(pt_dir.glob("*.pt"))
        if not pt_files:
            _log.error(f"'{pt_dir}' içinde .pt dosyası yok.")
            sys.exit(1)
        rng     = random.Random(args.seed)
        pt_path = rng.choice(pt_files)
        _log.info(f"Rastgele seçilen: {pt_path.name}")

    file_stem = pt_path.stem
    print(f"\n[Giriş]               {pt_path}")
    print(f"[Çıktı]               {args.output}")
    print(f"[Stem]                {file_stem}")
    print(f"[Epsilon (welding)]   {args.epsilon}")
    print(f"[Smooth iter]         {args.smooth_iterations}")
    print(f"[Repetition Penalty]  {args.repetition_penalty}")
    print(f"[Temperature]         {args.temperature}")
    print()

    # ── 2. RAW mesh verisini yükle ────────────────────────────────────────────
    _log.info("RAW mesh yükleniyor …")
    raw_data: Data = torch.load(pt_path, map_location="cpu", weights_only=False)

    _log.info(f"  x.shape          : {list(raw_data.x.shape)}")
    _log.info(f"  edge_index.shape : {list(raw_data.edge_index.shape)}")
    if hasattr(raw_data, "pos") and raw_data.pos is not None:
        _log.info(f"  pos.shape        : {list(raw_data.pos.shape)}")

    # ── 3. Trimesh oluştur (yüzey normal sorguları için) ──────────────────────
    _log.info("Trimesh yüzey modeli oluşturuluyor …")
    raw_trimesh = MeshReconstructor.build_trimesh_from_data(raw_data)
    if raw_trimesh is None:
        _log.warning(
            "Trimesh oluşturulamadı (face verisi yok veya trimesh kurulu değil). "
            "Yüzey normalleri KDTree+PCA ile tahmin edilecek."
        )

    # ── 4. MeshGenerator ─────────────────────────────────────────────────────
    _log.info("MeshGenerator başlatılıyor …")
    generator = MeshGenerator(
        checkpoint_path=args.checkpoint,
        vocab_path=args.vocab,
        norm_csv_path=args.norm_csv,
    )

    # ── 5. Token Dizisi Üret ──────────────────────────────────────────────────
    _log.info("Token dizisi üretiliyor …")
    t0 = time.time()
    token_ids: List[int] = generator.generate_sequence(
        raw_mesh_data     = raw_data,
        max_len           = args.max_gen_len,
        repetition_penalty= args.repetition_penalty,
        temperature       = args.temperature,
    )
    gen_time = time.time() - t0

    patch_tokens  = [t for t in token_ids if t not in SPECIAL_TOKENS]
    token_counter = Counter(patch_tokens)

    print(f"\n{'─'*60}")
    print("  ★  ÜRETİM İSTATİSTİKLERİ")
    print(f"{'─'*60}")
    print(f"  Toplam token            : {len(token_ids)}")
    print(f"  PATCH token sayısı      : {len(patch_tokens)}")
    print(f"  Benzersiz PATCH türü    : {len(token_counter)}")
    print(f"  Üretim süresi           : {gen_time:.3f} s")
    if token_counter:
        print(f"\n  Token Frekans Dağılımı (İlk 10):")
        max_freq = max(token_counter.values())
        for tid, freq in token_counter.most_common(10):
            bar = "█" * min(int(freq * 30 / max(max_freq, 1)), 30)
            print(f"    Token {tid:3d} : {freq:5d}  {bar}")
    print(f"{'─'*60}\n")

    # ── 6. MeshReconstructor v3 ───────────────────────────────────────────────
    _log.info("MeshReconstructor v3 başlatılıyor …")
    reconstructor = MeshReconstructor(
        vocabulary      = generator.vocabulary,
        norm_params     = generator.norm_params,
        raw_mesh_trimesh= raw_trimesh,   # trimesh normals için
        epsilon         = args.epsilon,
    )

    _log.info("v3 Mesh yeniden inşa ediliyor …")
    t1 = time.time()
    final_verts, final_faces = reconstructor.reconstruct(
        token_ids        = token_ids,
        raw_mesh_data    = raw_data,
        output_path      = args.output,
        file_stem        = file_stem,
        smooth           = not args.no_smooth,
        smooth_iterations= args.smooth_iterations,
        smooth_factor    = args.smooth_factor,
    )
    recon_time = time.time() - t1

    # ── 7. Özet Rapor ─────────────────────────────────────────────────────────
    print(f"\n{'─'*60}")
    print("  ★  YENİDEN İNŞA RAPORU (v3)")
    print(f"{'─'*60}")
    print(f"  Düğüm sayısı          : {len(final_verts)}")
    print(f"  Yüzey sayısı          : {len(final_faces)}")
    print(f"  Yeniden inşa süresi   : {recon_time:.3f} s")
    print(f"  Toplam süre           : {gen_time + recon_time:.3f} s")
    print(f"  Çıktı (NAS)           : {Path(args.output).resolve()}")
    print(f"{'─'*60}\n")

    # ── 8. PyVista Görselleştirme ─────────────────────────────────────────────
    if args.no_vis:
        _log.info("Görselleştirme '--no-vis' ile devre dışı.")
    elif not _PYVISTA_AVAILABLE:
        _log.warning("PyVista kurulu değil: pip install pyvista")
    else:
        _log.info("PyVista görselleştirme hazırlanıyor …")

        pv_raw = (
            _build_pyvista_from_raw_graph(raw_data)
            if (hasattr(raw_data, "pos") and raw_data.pos is not None)
            else pv.PolyData()
        )
        _log.info(f"  RAW       : {pv_raw.n_points} nokta")

        refined_dir = Path(args.refined_dir)
        refined_nas = refined_dir / f"{file_stem}.nas"
        if refined_nas.is_file():
            try:
                pv_refined = _load_nas_as_pyvista(refined_nas)
                _log.info(
                    f"  REFINED   : {pv_refined.n_points} nokta, "
                    f"{pv_refined.n_cells} element"
                )
            except Exception as exc:
                _log.warning(f"  REFINED yüklenemedi: {exc}")
                pv_refined = pv.PolyData()
        else:
            _log.warning(f"  REFINED bulunamadı: {refined_nas}")
            pv_refined = pv.PolyData()

        pv_pred = _build_pyvista_from_arrays(final_verts, final_faces)
        _log.info(
            f"  PREDICTION: {pv_pred.n_points} nokta, {pv_pred.n_cells} element"
        )

        if pv_pred.n_points == 0:
            _log.error("PREDICTION mesh boş! Token–sözlük eşleşmesi başarısız.")
        else:
            b = pv_pred.bounds
            _log.info(
                f"  Sınırlar: "
                f"X=[{b[0]:.2f},{b[1]:.2f}] "
                f"Y=[{b[2]:.2f},{b[3]:.2f}] "
                f"Z=[{b[4]:.2f},{b[5]:.2f}]"
            )

        print("\nGörselleştirici açılıyor — kapatmak için [Q].\n")
        show_triple_comparison(
            pv_raw,
            pv_refined,
            pv_pred,
            window_title=(
                f"MeshCrafter v3 – {file_stem}  |  RAW | REFINED | PREDICTION"
            ),
        )
