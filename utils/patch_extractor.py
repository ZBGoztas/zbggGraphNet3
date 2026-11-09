import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import networkx as nx
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, Tuple, Any, List, Optional
from utils.logger_utils import get_context_logger


# ====================================================================== #
#  Veri Yapıları (Data Classes)
# ====================================================================== #

@dataclass
class Patch:
    """
    Tek bir düğüm etrafından koparılmış 1-Ring Neighborhood yamasını temsil eder.

    Attributes:
        center_global_id:   Merkez düğümün orijinal (global) mesh ID'si.
        local_nodes:        Lokal ID -> (x, y, z) koordinat sözlüğü.
                            Lokal ID 0 her zaman merkez düğümdür.
                            Koordinatlar merkeze göre normalize edilmiştir (merkez = 0,0,0).
        local_elements:     Lokal element ID -> {'type': str, 'nodes': List[int]} sözlüğü.
                            Element node referansları lokal ID'lere çevrilmiştir.
        global_to_local:    Global Node ID -> Lokal Node ID eşleme tablosu.
    """
    center_global_id: int
    local_nodes: Dict[int, Tuple[float, float, float]]
    local_elements: Dict[int, Dict[str, Any]]
    global_to_local: Dict[int, int] = field(repr=False)
    topology_hash: str = ""


# ====================================================================== #
#  Ana Sınıf
# ====================================================================== #

class PatchExtractor:
    """
    Mesh üzerindeki her düğümün 1-Ring Neighborhood (Şemsiye/Umbrella)
    yapısını koparıp, bağımsız ve standartlaştırılmış küçük yamalar (Patch)
    haline getiren sınıf.

    Algoritma Adımları:
        A) Ters Haritalama  – Node -> Element ters indeksi oluşturulur.
        B) Yamayı Koparma   – Merkez düğüme bağlı elemanlar ve komşu düğümler toplanır.
        C) Lokal İndeksleme – Global ID'ler 0'dan başlayarak yeniden numaralandırılır (merkez = 0).
        D) Lokal Normalizasyon – Koordinatlar merkez düğüme göre orijine taşınır.
    """

    def __init__(self, file_name: Optional[str] = None) -> None:
        self.logger = get_context_logger(name="patch_extractor", file_name=file_name)
        self._node_to_elements: Dict[int, List[int]] = {}

    # ------------------------------------------------------------------ #
    #  ADIM A: Ters Haritalama (Reverse Mapping)
    # ------------------------------------------------------------------ #

    def _build_reverse_map(self, elements: Dict[int, Dict[str, Any]]) -> Dict[int, List[int]]:
        """
        Element -> Node bağlantısını tersine çevirip Node -> Element sözlüğü oluşturur.
        Bu sayede herhangi bir node'a bağlı elemanlar O(1) ile bulunabilir.

        Args:
            elements: {eid: {'type': 'tria'|'quad', 'nodes': [n1, n2, ...]}}

        Returns:
            {node_id: [eid1, eid2, ...]}
        """
        reverse_map: Dict[int, List[int]] = defaultdict(list)

        for eid, edata in elements.items():
            for nid in edata.get("nodes", []):
                reverse_map[nid].append(eid)

        self.logger.info(f"Ters haritalama tamamlandı | {len(reverse_map)} node indekslendi")
        return dict(reverse_map)

    # ------------------------------------------------------------------ #
    #  ADIM B: Yamayı Koparma (Extraction)
    # ------------------------------------------------------------------ #

    def _extract_ring(
        self,
        center_nid: int,
        elements: Dict[int, Dict[str, Any]],
    ) -> Tuple[List[int], set]:
        """
        Merkez düğüme bağlı elemanları ve bu elemanların kullandığı
        tüm düğümleri (tekrarsız küme) toplar.

        Args:
            center_nid: Merkez düğümün global ID'si.
            elements:   Global element sözlüğü.

        Returns:
            (ring_element_ids, ring_node_ids)
            ring_element_ids : Merkeze bağlı element ID'leri listesi.
            ring_node_ids    : Bu elemanların kullandığı tüm düğüm ID'leri kümesi.
        """
        ring_eids: List[int] = self._node_to_elements.get(center_nid, [])

        if not ring_eids:
            return [], set()

        ring_nids: set = set()
        for eid in ring_eids:
            for nid in elements[eid].get("nodes", []):
                ring_nids.add(nid)

        return ring_eids, ring_nids

    # ------------------------------------------------------------------ #
    #  ADIM C: Lokal İndeksleme (Re-indexing)
    # ------------------------------------------------------------------ #

    @staticmethod
    def _build_local_index(center_nid: int, ring_nids: set) -> Dict[int, int]:
        """
        Global düğüm ID'lerini sıfırdan başlayarak yeniden numaralandırır.
        Merkez düğüm kesinlikle Lokal ID: 0 alır.

        Args:
            center_nid: Merkez düğümün global ID'si.
            ring_nids:  Yamadaki tüm düğüm ID'leri kümesi.

        Returns:
            {global_nid: local_id}  (center_nid -> 0 garantili)
        """
        global_to_local: Dict[int, int] = {center_nid: 0}

        local_counter = 1
        for nid in sorted(ring_nids):
            if nid == center_nid:
                continue
            global_to_local[nid] = local_counter
            local_counter += 1

        return global_to_local

    # ------------------------------------------------------------------ #
    #  ADIM D: Lokal Normalizasyon (Merkeze Çekme)
    # ------------------------------------------------------------------ #

    @staticmethod
    def _normalize_coords(
        global_to_local: Dict[int, int],
        nodes: Dict[int, Tuple[float, float, float]],
        center_nid: int,
    ) -> Dict[int, Tuple[float, float, float]]:
        """
        Yamanın koordinatlarını merkez düğümü orijine oturtacak şekilde kaydırır.
        Merkez düğümün koordinatları tüm düğümlerden çıkarılır.

        Sonuç: Lokal ID 0 → (0.0, 0.0, 0.0)

        Args:
            global_to_local: Global -> Lokal ID eşleme tablosu.
            nodes:           Global node koordinat sözlüğü.
            center_nid:      Merkez düğümün global ID'si.

        Returns:
            {local_id: (dx, dy, dz)}  (merkez = (0, 0, 0))
        """
        cx, cy, cz = nodes[center_nid]
        center_vec = np.array([cx, cy, cz], dtype=np.float64)

        local_nodes: Dict[int, Tuple[float, float, float]] = {}

        for global_nid, local_id in global_to_local.items():
            gx, gy, gz = nodes[global_nid]
            shifted = np.array([gx, gy, gz], dtype=np.float64) - center_vec
            local_nodes[local_id] = (float(shifted[0]), float(shifted[1]), float(shifted[2]))

        return local_nodes

    # ------------------------------------------------------------------ #
    #  Public API
    # ------------------------------------------------------------------ #

    def extract_patch(
        self,
        center_nid: int,
        nodes: Dict[int, Tuple[float, float, float]],
        elements: Dict[int, Dict[str, Any]],
    ) -> Optional[Patch]:
        """
        Tek bir düğüm etrafındaki 1-Ring Patch'i çıkarır.

        Ters harita henüz oluşturulmadıysa otomatik olarak oluşturulur.

        Args:
            center_nid: Merkez düğümün global ID'si.
            nodes:      Global node koordinat sözlüğü.
            elements:   Global element sözlüğü.

        Returns:
            Patch nesnesi veya eleman bulunamazsa None.
        """
        # Lazy: ters haritayı ilk çağrıda oluştur
        if not self._node_to_elements:
            self._node_to_elements = self._build_reverse_map(elements)

        # --- ADIM B: Yamayı Kopar ---
        ring_eids, ring_nids = self._extract_ring(center_nid, elements)

        if not ring_eids:
            self.logger.debug(f"Node {center_nid}: Bağlı element bulunamadı, atlanıyor")
            return None

        # --- ADIM C: Lokal İndeksleme ---
        global_to_local = self._build_local_index(center_nid, ring_nids)

        # Element connectivity'yi lokal ID'lere çevir
        local_elements: Dict[int, Dict[str, Any]] = {}
        for local_eid, global_eid in enumerate(ring_eids):
            edata = elements[global_eid]
            local_element_nodes = [global_to_local[nid] for nid in edata["nodes"]]
            local_elements[local_eid] = {
                "type": edata["type"],
                "nodes": local_element_nodes,
            }

        # --- ADIM D: Lokal Normalizasyon ---
        local_nodes = self._normalize_coords(global_to_local, nodes, center_nid)

        return Patch(
            center_global_id=center_nid,
            local_nodes=local_nodes,
            local_elements=local_elements,
            global_to_local=global_to_local,
        )

    def extract_all_patches(
        self,
        nodes: Dict[int, Tuple[float, float, float]],
        elements: Dict[int, Dict[str, Any]],
        selected_nodes: Optional[List[int]] = None,
    ) -> Dict[int, Patch]:
        """
        Mesh'teki tüm (veya seçilen) düğümler için 1-Ring Patch çıkarır.

        Args:
            nodes:          Global node koordinat sözlüğü.
            elements:       Global element sözlüğü.
            selected_nodes: İşlenecek düğüm ID listesi. None ise tüm düğümler.

        Returns:
            {center_global_id: Patch}
        """
        # --- ADIM A: Ters Haritalamayı bir kere oluştur ---
        self._node_to_elements = self._build_reverse_map(elements)

        target_nids = selected_nodes if selected_nodes is not None else list(nodes.keys())
        patches: Dict[int, Patch] = {}
        skipped = 0

        for nid in target_nids:
            if nid not in nodes:
                self.logger.warning(f"Node {nid} koordinat sözlüğünde bulunamadı, atlanıyor")
                skipped += 1
                continue

            patch = self.extract_patch(nid, nodes, elements)
            if patch is not None:
                patches[nid] = patch
            else:
                skipped += 1

        self.logger.info(
            f"Patch çıkarma tamamlandı | "
            f"toplam={len(target_nids)} | başarılı={len(patches)} | atlanan={skipped}"
        )
        return patches,skipped


# ====================================================================== #
#  Topoloji Analiz Sınıfı (WL-Hash)
# ====================================================================== #

class PatchAnalyzer:
    """
    Koparılmış Patch nesnelerinin topolojik parmak izini (hash) hesaplar
    ve aynı topolojiye sahip yamaları gruplar.

    Weisfeiler-Lehman (WL) Graph Hashing algoritmasını kullanarak,
    düğüm ID'lerinden bağımsız bir şekilde grafın iskeletine özel
    benzersiz bir kimlik metni üretir.

    Algoritma Adımları:
        1) Patch'in local_elements verisinden yönsüz networkx.Graph oluşturulur.
        2) Merkez düğüm (Lokal ID 0) role="center", diğerleri role="neighbor" ile işaretlenir.
        3) nx.weisfeiler_lehman_graph_hash(G, node_attr="role") ile hash hesaplanır.
        4) Tüm patch'ler hash değerlerine göre gruplanır.
    """

    def __init__(self, file_name: Optional[str] = None) -> None:
        self.logger = get_context_logger(name="patch_analyzer", file_name=file_name)

    # ------------------------------------------------------------------ #
    #  ADIM 2: Graf (Graph) Oluşturma
    # ------------------------------------------------------------------ #

    @staticmethod
    def _build_graph(patch: Patch) -> nx.Graph:
        """
        Patch'in local_elements verisinden yönsüz bir networkx grafı oluşturur.

        Her elemanın (tria/quad) köşeleri arasında kenarlar eklenir:
          - tria [a, b, c]    -> kenarlar: (a,b), (b,c), (c,a)
          - quad [a, b, c, d] -> kenarlar: (a,b), (b,c), (c,d), (d,a)

        Args:
            patch: İşlenecek Patch nesnesi.

        Returns:
            Düğüm ve kenarları eklenmiş nx.Graph.
        """
        G = nx.Graph()

        # Önce tüm düğümleri ekle (izole düğüm kalmaması için)
        for local_nid in patch.local_nodes:
            G.add_node(local_nid)

        # Eleman kenarlarını ekle
        for edata in patch.local_elements.values():
            nids = edata["nodes"]
            num_nodes = len(nids)
            for i in range(num_nodes):
                G.add_edge(nids[i], nids[(i + 1) % num_nodes])

        return G

    # ------------------------------------------------------------------ #
    #  ADIM 3: Merkez Düğüm İşaretleme
    # ------------------------------------------------------------------ #

    @staticmethod
    def _label_roles(G: nx.Graph) -> None:
        """
        Grafın her düğümüne rol özelliği atar.
          - Lokal ID 0 (merkez düğüm) -> role = "center"
          - Diğer tüm düğümler        -> role = "neighbor"

        Bu etiketleme WL-Hash'in merkezi tanıyabilmesi için kritiktir.

        Args:
            G: İşlenecek networkx grafı (in-place güncellenir).
        """
        for nid in G.nodes():
            G.nodes[nid]["role"] = "center" if nid == 0 else "neighbor"

    # ------------------------------------------------------------------ #
    #  ADIM 4: WL-Hash Hesaplama
    # ------------------------------------------------------------------ #

    def compute_hash(self, patch: Patch) -> str:
        """
        Tek bir Patch için WL-Hash hesaplar ve patch.topology_hash'e kaydeder.

        Args:
            patch: Hash'i hesaplanacak Patch nesnesi.

        Returns:
            Hesaplanan WL-Hash string'i.
        """
        # ADIM 2: Graf oluştur
        G = self._build_graph(patch)

        # ADIM 3: Merkez düğümü işaretle
        self._label_roles(G)

        # ADIM 4: WL-Hash hesapla (node_attr="role" ile merkez bilgisi dahil edilir)
        wl_hash = nx.weisfeiler_lehman_graph_hash(G, node_attr="role")

        # Sonucu Patch nesnesine kaydet
        patch.topology_hash = wl_hash

        return wl_hash

    def compute_all_hashes(self, patches: Dict[int, Patch]) -> None:
        """
        Birden fazla Patch için toplu WL-Hash hesaplar.

        Args:
            patches: {center_global_id: Patch} sözlüğü.
        """
        for nid, patch in patches.items():
            self.compute_hash(patch)

        unique_count = len({p.topology_hash for p in patches.values()})
        self.logger.info(
            f"Hash hesaplama tamamlandı | "
            f"toplam={len(patches)} patch | benzersiz={unique_count} topoloji"
        )

    # ------------------------------------------------------------------ #
    #  ADIM 5: Gruplama (Clustering)
    # ------------------------------------------------------------------ #

    def group_by_topology(self, patches: Dict[int, Patch]) -> Dict[str, List[int]]:
        """
        Tüm patch'leri topolojik hash değerlerine göre gruplar.

        Hash'ler henüz hesaplanmamışsa otomatik olarak hesaplanır.

        Args:
            patches: {center_global_id: Patch} sözlüğü.

        Returns:
            {wl_hash_string: [center_global_id_1, center_global_id_2, ...]}
        """
        # Hash'i boş olan patch varsa önce hesapla
        needs_hashing = any(p.topology_hash == "" for p in patches.values())
        if needs_hashing:
            self.compute_all_hashes(patches)

        groups: Dict[str, List[int]] = defaultdict(list)
        for nid, patch in patches.items():
            groups[patch.topology_hash].append(nid)

        self.logger.info(
            f"Topolojik gruplama tamamlandı | "
            f"{len(patches)} patch -> {len(groups)} benzersiz grup"
        )

        return dict(groups)


# # ====================================================================== #
# #  Örnek Kullanım (Dummy Veri + Isomorphism Kanıtı)
# # ====================================================================== #

# if __name__ == "__main__":

#     # ==================================================================
#     # SENARYO 1: Temel patch çıkarma ve hash hesaplama
#     # ==================================================================
#     # Dummy mesh: 7 node, 6 üçgen  (kusursuz iç düğüm = node 1)
#     #
#     #         5
#     #        / \
#     #       3---5
#     #      / \ / \
#     #     6---1---2
#     #      \ / \ /
#     #       7---8
#     #        \ /
#     #         (alt)
#     #
#     # Node 1 etrafında 6 üçgen -> klasik 1-ring

#     dummy_nodes: Dict[int, Tuple[float, float, float]] = {
#         1: (10.0, 10.0, 0.0),
#         2: (12.0, 10.0, 0.0),
#         3: (11.0, 12.0, 0.0),
#         5: ( 9.0, 12.0, 0.0),
#         6: ( 8.0, 10.0, 0.0),
#         7: ( 9.0,  8.0, 0.0),
#         8: (11.0,  8.0, 0.0),
#     }

#     dummy_elements: Dict[int, Dict[str, Any]] = {
#         101: {"type": "tria", "nodes": [1, 2, 3]},
#         102: {"type": "tria", "nodes": [1, 3, 5]},
#         103: {"type": "tria", "nodes": [1, 5, 6]},
#         104: {"type": "tria", "nodes": [1, 6, 7]},
#         105: {"type": "tria", "nodes": [1, 7, 8]},
#         106: {"type": "tria", "nodes": [1, 8, 2]},
#     }

#     extractor = PatchExtractor(file_name="demo")
#     analyzer = PatchAnalyzer(file_name="demo")

#     patch = extractor.extract_patch(center_nid=1, nodes=dummy_nodes, elements=dummy_elements)

#     if patch is not None:
#         h = analyzer.compute_hash(patch)
#         print("=" * 60)
#         print("SENARYO 1: Temel Patch Çıkarma")
#         print("=" * 60)
#         print(f"Merkez Global ID  : {patch.center_global_id}")
#         print(f"Topology WL-Hash  : {patch.topology_hash}")
#         print(f"Lokal Node Sayısı : {len(patch.local_nodes)}")
#         print(f"Lokal Elem Sayısı : {len(patch.local_elements)}")
#         print("-" * 60)
#         print("Lokal Node'lar (merkez = 0,0,0):")
#         for lid, coords in sorted(patch.local_nodes.items()):
#             print(f"  Lokal {lid:>2d} -> ({coords[0]:>7.3f}, {coords[1]:>7.3f}, {coords[2]:>7.3f})")
#         print("-" * 60)
#         print("Lokal Elementler:")
#         for eid, edata in patch.local_elements.items():
#             print(f"  Elem {eid}: {edata['type']} -> {edata['nodes']}")
#         print()

#     # ==================================================================
#     # SENARYO 2: Graph Isomorphism Kanıtı
#     # ==================================================================
#     # Aynı topoloji (merkez + 6 üçgen), tamamen farklı ID'ler ve
#     # farklı numaralandırma sırası. WL-Hash'ler aynı çıkmalıdır.
#     print("=" * 60)
#     print("SENARYO 2: Graph Isomorphism Kanıtı")
#     print("=" * 60)

#     # ----- Mesh A: Düşük ID'ler, saat yönü sırası -----
#     nodes_a: Dict[int, Tuple[float, float, float]] = {
#         10: (0.0, 0.0, 0.0),    # merkez
#         11: (1.0, 0.0, 0.0),
#         12: (0.5, 0.87, 0.0),
#         13: (-0.5, 0.87, 0.0),
#         14: (-1.0, 0.0, 0.0),
#         15: (-0.5, -0.87, 0.0),
#         16: (0.5, -0.87, 0.0),
#     }
#     elems_a: Dict[int, Dict[str, Any]] = {
#         1: {"type": "tria", "nodes": [10, 11, 12]},
#         2: {"type": "tria", "nodes": [10, 12, 13]},
#         3: {"type": "tria", "nodes": [10, 13, 14]},
#         4: {"type": "tria", "nodes": [10, 14, 15]},
#         5: {"type": "tria", "nodes": [10, 15, 16]},
#         6: {"type": "tria", "nodes": [10, 16, 11]},
#     }

#     # ----- Mesh B: Yüksek ID'ler, karışık sıra -----
#     nodes_b: Dict[int, Tuple[float, float, float]] = {
#         5042: (100.0, 200.0, 50.0),   # merkez (tamamen farklı konum)
#         8011: (101.0, 200.0, 50.0),
#         7733: (100.5, 200.87, 50.0),
#         9120: ( 99.5, 200.87, 50.0),
#         6001: ( 99.0, 200.0, 50.0),
#         6500: ( 99.5, 199.13, 50.0),
#         7200: (100.5, 199.13, 50.0),
#     }
#     # Elemanlar kasıtlı olarak karışık sırada tanımlandı
#     elems_b: Dict[int, Dict[str, Any]] = {
#         901: {"type": "tria", "nodes": [5042, 7733, 9120]},
#         902: {"type": "tria", "nodes": [5042, 6500, 7200]},
#         903: {"type": "tria", "nodes": [5042, 8011, 7733]},
#         904: {"type": "tria", "nodes": [5042, 9120, 6001]},
#         905: {"type": "tria", "nodes": [5042, 7200, 8011]},
#         906: {"type": "tria", "nodes": [5042, 6001, 6500]},
#     }

#     ext_a = PatchExtractor(file_name="mesh_A")
#     ext_b = PatchExtractor(file_name="mesh_B")

#     patch_a = ext_a.extract_patch(center_nid=10, nodes=nodes_a, elements=elems_a)
#     patch_b = ext_b.extract_patch(center_nid=5042, nodes=nodes_b, elements=elems_b)

#     if patch_a and patch_b:
#         hash_a = analyzer.compute_hash(patch_a)
#         hash_b = analyzer.compute_hash(patch_b)

#         print(f"Mesh A  -> Merkez: 10   | Hash: {hash_a}")
#         print(f"Mesh B  -> Merkez: 5042 | Hash: {hash_b}")
#         print("-" * 60)

#         if hash_a == hash_b:
#             print("SONUC: AYNI HASH -> Topolojik olarak ESIT (Isomorphic)")
#         else:
#             print("SONUC: FARKLI HASH -> Topolojik olarak FARKLI")
#         print()

#     # ==================================================================
#     # SENARYO 3: Toplu çıkarım + Gruplama
#     # ==================================================================
#     print("=" * 60)
#     print("SENARYO 3: Toplu Patch Çıkarma ve Topolojik Gruplama")
#     print("=" * 60)

#     all_patches = extractor.extract_all_patches(dummy_nodes, dummy_elements)
#     groups = analyzer.group_by_topology(all_patches)

#     print(f"\nToplam patch: {len(all_patches)} | Benzersiz topoloji: {len(groups)}")
#     for wl_hash, node_ids in groups.items():
#         print(f"  Hash: {wl_hash[:16]}... | {len(node_ids)} patch | Node'lar: {node_ids}")
