import numpy as np
from utils.logger_utils import get_logger, get_context_logger
from typing import Dict, Tuple, Any, List, Optional


class CanonicalOrderProcessor:
    """
    Mesh node'larını ve elementlerini canonical order (standartlaştırılmış sıra) algoritmasına göre sıralar.
    
    Özellikler:
    1. Node'lar: Z -> Y -> X uzaysal sırasına göre dizilir.
    2. Elementler: Komşuluk (Topology) esas alınarak, bir uçtan başlayıp
       BFS (Breadth-First Search) ile gezilerek sıralanır.
       Bu, AR (Auto-Regressive) modellerin "sıradaki parça hemen yanda olmalı"
       varsayımı için kritiktir.
    """


    def __init__(self, file_name: Optional[str] = None):
        self.file_name = file_name
        self.logger = get_context_logger(name="canonical_order", file_name=file_name)


    def  _calculate_node_canonical_key(self, node_id: int, coords: Tuple[float, float, float]) -> Tuple[float, float, float]:
        """
        Node sıralama anahtarı: Z (Yükseklik) -> Y (Derinlik) -> X (Genişlik).
        """
        x,y,z = coords 
        # Yuvarlama, kayan nokta hatalarını önler
        return (round(z,5), round(y,5), round(x,5))
    
    def _calculate_element_canonical_key(self, element_id: int, element_data: Dict[str, Any], nodes: Dict[int, Tuple[float, float, float]]) -> Tuple[float, float, float]:
        """
        Element sıralama anahtarı: Element merkezinin (Centroid) koordinatları.
        """
        element_nodes = element_data.get("nodes", [])

        if not element_nodes:
            return (float('inf'),float('inf'),float('inf'))
        
        total_x = total_y =total_z = 0.0 
        valid_node_count = 0

        for node_id in element_nodes:
            if node_id in nodes:
                x,y,z = nodes[node_id]
                total_x += x
                total_y += y
                total_z += z
                valid_node_count += 1
        if valid_node_count == 0:
            return (float('inf'),float('inf'),float('inf'))

        cx = total_x / valid_node_count
        cy = total_y / valid_node_count
        cz = total_z / valid_node_count

        return (round(cz,5), round(cy,5), round(cx,5))
    

    def _build_element_adjacency_graph(self, elements: Dict[int, Dict[str, Any]]) -> Dict[int, List[int]]:
        """
        Elementler arası komşuluk grafiği oluşturur.
        İki element en az 2 node paylaşıyorsa (edge sharing) komşu sayılır.
        """
        # Node -> Element Listesi haritası (Hızlandırma için ters index)

        node_to_elements = {}

        for eid, data in elements.items():
            for nid in data.get("nodes", []):
                if nid not in node_to_elements:
                    node_to_elements[nid] = []
                node_to_elements[nid].append(eid)

        adjacency = {elem_id: set() for elem_id in elements.keys()}


        # Her element için komşuları bul
        for eid, data in elements.items():
            nodes = set(data.get("nodes", []))
            neighbor_candidates = set()

            # Bu elementin node'larını kullanan diğer elementleri aday yap
            for nid in nodes:
                for candidate_id in node_to_elements[nid]:
                    neighbor_candidates.add(candidate_id)

            # Adayları kontrol et (En az 2 ortak node)
            for cand_id in neighbor_candidates:
                cand_nodes = set(elements[cand_id].get("nodes", []))

                if len(nodes.intersection(cand_nodes)) >= 2:
                    adjacency[eid].add(cand_id)
                    adjacency[cand_id].add(eid)
        
        # Set'i listeye çevir
        return {k: list(v) for k, v in adjacency.items()}
    

    def _sort_elements_with_adjacency(self, elements: Dict[int, Dict[str, Any]], nodes: Dict[int, Tuple[float, float, float]]) -> Dict[int, Dict[str, Any]]:
        """
        BFS (Breadth-First Search) kullanarak elementleri topolojik olarak sıralar.
        """

        self.logger.info(f"Element'ler topolojik (komşuluk) sırasına diziliyor... ({len(elements)} adet)")

        if not elements:
            self.logger.warning(f"Element listesi boş")
            return elements
        

        adjacency = self._build_element_adjacency_graph(elements)
        visited = set()
        ordered_element_ids = []

        # Tüm elementler ziyaret edilene kadar dön (Bağlantısız adalar için)
        all_element_ids = set(elements.keys())

        while len(visited) < len(elements):
            # Henüz ziyaret edilmemişler arasından en "altta" (min Z, Y, X) olanı seç

            remaining = list(all_element_ids - visited)
            start_element = min(
                remaining,
                key=lambda eid: self._calculate_element_canonical_key(eid, elements[eid], nodes)
            )

            # BFS Başlat
            queue = [start_element]
            visited.add(start_element)


            while queue:
                current_elem = queue.pop(0)
                ordered_element_ids.append(current_elem)

                # komsulari al
                neighbors = adjacency.get(current_elem, [])
                unvisited_neighbors = [n for n in neighbors if n not in visited]

                # Komşuları da kendi içinde uzaysal sıraya diz (Deterministik olması için)
                unvisited_neighbors.sort(
                    key=lambda eid: self._calculate_element_canonical_key(eid, elements[eid], nodes)
                )

                for neighbor in unvisited_neighbors:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append(neighbor)


        # Sıralanmış dictionary oluştur
        canonical_elements = {elem_id: elements[elem_id] for elem_id in ordered_element_ids}
        return canonical_elements

    def _sort_nodes_canonical(self, nodes: Dict[int, Tuple[float, float, float]]) -> Dict[int, Tuple[float,float,float]]:
        """
        Node'ları uzaysal (Z-Y-X) sıraya dizer.
        """
        sorted_nodes = sorted(
            nodes.items(),
            key=lambda item: self._calculate_node_canonical_key(item[0], item[1])
        )

        return {node_id: coords for node_id, coords in sorted_nodes}
    
    def process_canonical_order(self, nodes: Dict[int, Tuple[float, float, float]], elements: Dict[int, Dict[str, Any]], use_adjacency: bool = True) -> Dict[str, Any]:
        """
        Ana işlem fonksiyonu.
        """
        self.logger.info("Canonical order işlemi başlatılıyor...")
        
        # 1. Node'ları Sırala
        canonical_nodes = self._sort_nodes_canonical(nodes)

        # 2. Elementleri Sırala (BFS ile)
        if use_adjacency:
            canonical_elements = self._sort_elements_with_adjacency(elements, canonical_nodes)
        else:
            # Fallback: Sadece uzaysal sıralama
            sorted_elements = sorted(
                elements.items(),
                key= lambda item: self._calculate_element_canonical_key(item[0], item[1], canonical_nodes)
            )
            canonical_elements = {k: v for k,v in sorted_elements}

        return {
            "nodes": canonical_nodes,
            "elements": canonical_elements
        }

def apply_canonical_order(nodes: Dict[int, Tuple[float, float, float]], elements: Dict[int, Dict[str, Any]], use_adjacency: bool = True, file_name: Optional[str] = None) -> Dict[str, Dict]:
    """
    Bu fonksiyon dışarıdan çağrılan ana fonksiyondur.
    Veriyi sıralar VE ID'leri 0'dan başlayarak yeniden numaralandırır (Re-indexing).
    
    Adımlar:
    1. Node'ları ve Elementleri uzaysal düzene (canonical key) göre sıralar.
    2. Yeni sıralamaya göre ID'leri 0, 1, 2... olacak şekilde günceller.
    3. Elementlerin içindeki node referanslarını da yeni ID'lere çevirir.
    
    Args:
        nodes: Orijinal node verisi.
        elements: Orijinal element verisi.
        use_adjacency: BFS kullanılsın mı? (Varsayılan: True).
        file_name: Loglama için dosya adı.
        
    Returns:
        Dict: {'nodes': {0: (x,y,z)...}, 'elements': {0: {'type':..., 'nodes': [0, 1...]}}}
    """
    logger = get_context_logger(name="canonical_order", file_name=file_name)

    processor = CanonicalOrderProcessor(file_name=file_name)
    # 1. Sıralama (ID'ler hala eski)

    sorted_data = processor.process_canonical_order(nodes, elements, use_adjacency)
    sorted_nodes = sorted_data['nodes']
    sorted_elements = sorted_data['elements']

    # 2. Node Re-indexing
    new_nodes_dict = {}
    old_to_new_node_map = {}


    for new_id, (old_id, coords) in enumerate(sorted_nodes.items()):
        old_to_new_node_map[old_id] = new_id
        new_nodes_dict[new_id] = coords

    # 3. Element Re-indexing
    new_elements_dict = {}
    for new_elem_id, (old_elem_id, elem_data) in enumerate(sorted_elements.items()):
        # Elementin içindeki node ID'lerini de güncelle
        new_node_indices = []
        for old_nid in elem_data['nodes']:
            if old_nid in old_to_new_node_map:
                new_node_indices.append(old_to_new_node_map[old_nid])
            else:
                # Nadir durum: Element silinmiş bir node'a referans veriyorsa
                processor.logger.warning("Element silinmiş bir node'a referans veriyor")
                pass
        new_elements_dict[new_elem_id] = {
            "type": elem_data["type"],
            "nodes": new_node_indices,
            "original_id": old_elem_id # Debug için eski ID'yi sakla
        }
    processor.logger.info(f"Tamamlandı. Node: {len(new_nodes_dict)}, Element: {len(new_elements_dict)}")
    return {'nodes': new_nodes_dict, 'elements': new_elements_dict}
