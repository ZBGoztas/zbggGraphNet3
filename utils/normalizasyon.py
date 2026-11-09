import os, sys
import csv
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Optional
import time
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.config_utils import load_config
from utils.logger_utils import get_logger, get_context_logger
from utils.parser_utils import ParserNas


class NormalizationManager:
    """
    Normalizasyon parametrelerini hesaplar, saklar ve uygular.
    CSV tabanlı parametre yönetiminden sorumludur.
    """

    SCALE_FACTOR = 0.95


    def __init__(self, csv_path: str):
        self.csv_path = csv_path
        self.logger = get_context_logger(
            name="NormalizationManager",
            log_file_name="fileNormalisayonLog"
        )
        self._params: Dict[str, Tuple[np.ndarray, float]] = {}
        self._load_params()

    # ------------------------------------------------------------------ #
    #  CSV Parametre Yönetimi
    # ------------------------------------------------------------------ #
    def _load_params(self) -> None:
        """CSV dosyasından normalizasyon parametrelerini belleğe yükler."""
        if not os.path.exists(self.csv_path):
            self.logger.info(f"csv dosya dizin hatasi | {self.csv_path}")
            return
        try:
            with open(self.csv_path, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                next(reader, None)  # header satırını atla
                for row in reader:
                    if len(row) == 5:
                        file_name, cx, cy, cz, scale = row
                        center = np.array([float(cx), float(cy), float(cz)])
                        self._params[file_name] = (center, float(scale))

        except Exception as e:
            self.logger.warning(f"Normalizasyon parametreleri yüklenemedi | {e}")   

    def _save_params(self, file_name: str, center: np.ndarray, scale: float) -> None:
        """Yeni bir dosyanın normalizasyon parametrelerini CSV'ye ekler."""
        # Zaten kayıtlıysa tekrar yazma
        if self._record_exists_in_csv(file_name):
            return

        try:
            file_exists = os.path.exists(self.csv_path)
            os.makedirs(os.path.dirname(self.csv_path), exist_ok=True)

            with open(self.csv_path, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                if not file_exists:
                    writer.writerow(['file_name', 'center_x', 'center_y', 'center_z', 'scale_max_radius'])
                writer.writerow([file_name, center[0], center[1], center[2], scale])

        except Exception as e:
            self.logger.warning(f"Parametre CSV'ye yazılamadı: {file_name} | {e}")

    def _record_exists_in_csv(self, file_name: str) -> bool:
        """CSV dosyasında bu dosya adının kaydının olup olmadığını kontrol eder."""
        if not os.path.exists(self.csv_path):
            return False
        try:
            with open(self.csv_path, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                next(reader, None)  # header atla
                return any(row[0] == file_name for row in reader if row)
        except Exception as e:
            self.logger.warning(f"CSV okuma hatası: {e}")
            return False
        

    # ------------------------------------------------------------------ #
    #  Public API
    # ------------------------------------------------------------------ #

    def get_params(self, file_name: str) -> Optional[Tuple[np.ndarray, float]]:
        """Bellekten parametre döndürür. Yoksa None."""
        return self._params.get(file_name)

    def has_params(self, file_name: str) -> bool:
        return file_name in self._params

    def apply(self, nodes: dict, center: np.ndarray, scale: float) -> Dict:
        """
        Node koordinatlarını normalize eder.
        Formül: 0.95 * ((coords - center) / scale)
        """
        if not nodes or scale < 1e-10:
            return nodes

        nids = list(nodes.keys())
        coords = np.array([nodes[nid] for nid in nids])
        normalized = self.SCALE_FACTOR * ((coords - center) / scale)
        return dict(zip(nids, map(tuple, normalized)))

    def compute_and_apply(self, file_name: str, nodes: dict) -> Optional[Dict]:
        """
        Parametreleri hesaplar, CSV'ye kaydeder ve normalizasyonu uygular.
        RAW dosyalar için kullanılır.
        """
        if not nodes:
            return None

        coords  = np.array(list(nodes.values()))
        center  = np.mean(coords, axis=0)
        distances  = np.linalg.norm(coords - center, axis=1)
        max_radius = np.max(distances)

        if max_radius < 1e-10:
            return nodes

        # Parametre bellekte yoksa kaydet
        if file_name not in self._params:
            self._params[file_name] = (center, max_radius)
            self._save_params(file_name, center, max_radius)

        return self.apply(nodes, center, max_radius)


# ====================================================================== #


class NasFileProcessor:
    """
    NASTRAN .nas dosyalarını okur, normalize eder ve yazar.
    RAW ve REFINED pipeline'larını yönetir.
    """

    FAILED_LOG = Path("failed_files.txt")

    def __init__(self, config: dict):
        self.config      = config
        self.parser      = ParserNas()
        self.normalizer  = NormalizationManager(config.get('normalized_parameter_csv_dir'))
        self.logger      = get_context_logger(
            name="NasFileProcessor",
            log_file_name="fileNormalisayonLog"
        )
        logs_dir = config.get('logs_dir')
        if logs_dir:
            self.FAILED_LOG = Path(logs_dir) / "failed_files.txt"

    # ------------------------------------------------------------------ #
    #  Yardımcı Metodlar
    # ------------------------------------------------------------------ #

    def _log_failed(self, filename: str, reason: str = "") -> None:
        """Başarısız dosyaları failed_files.txt'ye kaydeder."""
        with open(self.FAILED_LOG, 'a', encoding='utf-8') as f:
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
            f.write(f"[{timestamp}] {filename} | {reason}\n")

    def _write_nas_file(self, output_path: Path, nodes: dict, elements: dict, filename: str) -> bool:
        """Normalize edilmiş mesh verisini NASTRAN formatında yazar."""
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write("$ NASTRAN Bulk Data File (Normalized & Spatially Sorted)\n")
                f.write("BEGIN BULK\n")

                for node_id in sorted(nodes.keys()):
                    x, y, z = nodes[node_id]
                    f.write(f"GRID,{node_id},,{x:.7f},{y:.7f},{z:.7f}\n")

                for elem_id in sorted(elements.keys()):
                    elem      = elements[elem_id]
                    nodes_str = ",".join(map(str, elem["nodes"]))
                    card      = "CTRIA3" if elem["type"] == "tria" else "CQUAD4"
                    f.write(f"{card},{elem_id},1,{nodes_str}\n")

                f.write("ENDDATA\n")
            return True

        except Exception as e:
            self.logger.error(f"Yazma hatası [{filename}]: {e}", exc_info=True)
            return False

    def _parse_file(self, nas_file: Path) -> Optional[dict]:
        """Dosyayı parse eder. Hata olursa None döner."""
        try:
            return self.parser.parse_file(nas_file)
        except Exception as e:
            self.logger.error(f"Parse hatası: {nas_file.name} | {e}")
            self._log_failed(nas_file.name, f"Parse Hatası: {e}")
            return None

    # ------------------------------------------------------------------ #
    #  Pipeline Metodları
    # ------------------------------------------------------------------ #

    def process_raw(self, nas_file: Path) -> None:
        """Tek bir RAW dosyasını işler: parse → normalize → yaz."""
        logger = get_context_logger(
            name="NasFileProcessor",
            log_file_name="fileNormalisayonLog",
            file_name=nas_file.name
        )
        logger.info(f"[RAW] -> {nas_file.name}")

        mesh_data = self._parse_file(nas_file)
        if mesh_data is None:
            return

        if not mesh_data.get("nodes"):
            logger.warning("Mesh nodes boş")
            self._log_failed(nas_file.name, "Mesh nodes boş")
            return

        normalized_nodes = self.normalizer.compute_and_apply(
            file_name=nas_file.name,
            nodes=mesh_data['nodes']
        )

        if normalized_nodes is None:
            logger.warning("Normalizasyon başarısız")
            self._log_failed(nas_file.name, "Normalizasyon başarısız (Node None)")
            return

        output_path = Path(self.config.get('raw_output_dir')) / nas_file.name
        success = self._write_nas_file(output_path, normalized_nodes, mesh_data['elements'], nas_file.name)

        if not success:
            self._log_failed(nas_file.name, "Yazma başarısız")

    def process_refined(self, nas_file: Path) -> None:
        """Tek bir REFINED dosyasını işler: parse → RAW parametresiyle normalize → yaz."""
        logger = get_context_logger(
            name="NasFileProcessor",
            log_file_name="fileNormalisayonLog",
            file_name=nas_file.name
        )
        logger.info(f"[REFINED] -> {nas_file.name}")

        if not self.normalizer.has_params(nas_file.name):
            logger.warning(f"Normalizasyon parametresi bulunamadı: {nas_file.name}")
            self._log_failed(nas_file.name, "Normalizasyon parametreleri bulunamadı")
            return

        mesh_data = self._parse_file(nas_file)
        if mesh_data is None:
            return

        if not mesh_data.get('nodes'):
            logger.warning("Refined mesh nodes boş")
            self._log_failed(nas_file.name, "Refined Mesh nodes boş")
            return

        center, scale = self.normalizer.get_params(nas_file.name)

        try:
            normalized_nodes = self.normalizer.apply(mesh_data['nodes'], center, scale)
        except Exception as e:
            logger.error(f"Refined normalizasyon hatası: {e}")
            self._log_failed(nas_file.name, f"Refined Normalizasyon Hatası: {e}")
            return

        output_path = Path(self.config.get('refined_output_dir')) / nas_file.name
        success = self._write_nas_file(output_path, normalized_nodes, mesh_data['elements'], nas_file.name)

        if not success:
            self._log_failed(nas_file.name, "Refined Yazma Başarısız")

    def run(self) -> None:
        """Tüm pipeline'ı çalıştırır: RAW → REFINED."""
        os.makedirs(self.config.get('raw_output_dir'), exist_ok=True)
        os.makedirs(self.config.get('refined_output_dir'), exist_ok=True)

        # --- RAW ---
        raw_files = list(Path(self.config.get('raw_dir')).glob("*.nas"))
        print(f"\nRAW DOSYALAR: {len(raw_files)}\n")
        for nas_file in raw_files:
            self.process_raw(nas_file)

        # --- REFINED ---
        refined_files = list(Path(self.config.get('refined_dir')).glob("*.nas"))
        print(f"\nREFINED DOSYALAR: {len(refined_files)}\n")
        for nas_file in refined_files:
            self.process_refined(nas_file)


# ====================================================================== #

def main():
    config = load_config()
    processor = NasFileProcessor(config)
    processor.run()


if __name__ == "__main__":
    main()