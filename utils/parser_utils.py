from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from typing import Dict, Tuple, Any, List, Optional
from utils.logger_utils import get_logger, get_context_logger


class ParserNas():
    """
    Basit NASTRAN .nas parser:
    - GRID      -> node (id: (x,y,z))
    - CTRIA3    -> element tria
    - CQUAD4    -> element quad
    - Satır devamı '+' desteği
    - Bozuk bilimsel gösterim düzeltmesi (1.234-05 -> 1.234E-05)
    """
    def __init__(self) -> None:
        self.logger = get_logger(name="parser")

    @staticmethod
    def _fix_scientific_token(tok: str) -> str:
        """
        Bozuk bilimsel gösterimi düzeltir (örn: 1.234-05 -> 1.234E-05, 1.234+05 -> 1.234E+05).
        NASTRAN dosyalarında sıkça karşılaşılan hatalı bilimsel notasyonu standart hale getirir.
        """
        s = tok.strip()
        if not s or 'e' in s.lower():
            return s

        # Pozitif üs: 1.234+05
        plus_idx = s.rfind('+')
        if plus_idx > 0 and s[plus_idx - 1].isdigit():
            return s[:plus_idx] + 'E+' + s[plus_idx + 1:]

        # Negatif üs: 1.234-05 veya -2.626-4
        dash_idx = s.rfind('-')
        if dash_idx > 0 and s[dash_idx - 1].isdigit():
            return s[:dash_idx] + 'E' + s[dash_idx:]

        return s
    def safe_float(self, s:str) ->Optional[float]:
        """
        String değeri güvenli bir şekilde float'a dönüştürür.
        Hatalı değerler için None döner ve hata loglar. Bilimsel notasyon düzeltmesi yapar.
        """

        try:
            return float(self._fix_scientific_token(s))
        
        except Exception as e:
            self.logger.debug(f"safe_float başarısız: '{s}' | Hata: {e}")
            return None
        
    
    @staticmethod
    def _is_comment_or_emty(line: str) -> bool:
        """
        Satırın yorum satırı ('$' ile başlayan) veya boş satır olup olmadığını kontrol eder.
        Parser'da bu satırların atlanması için kullanılır.
        """
        ls = line.strip()
        return (not ls) or ls.startswith("$")
    
    @staticmethod
    def _split_card(line: str) -> List[str]:
        """
        NASTRAN kart satırını parçalara ayırır.
        Free-field virgül veya whitespace ayrımını destekler.
        """
        return line.strip().split(",")
    
    def _normalize_cards(self, lines: List[str]) -> List[str]:
        """
        NASTRAN dosyasındaki satır devamlarını normalleştirir.
        '+' ile devam eden satırları birleştirir ve yorum/boş satırları filtreler.
        """
        merged: List[str] = []
        buf = ""

        for raw in lines:
            if self._is_comment_or_emty(raw):
                continue
            line = raw.rstrip("\n")
            if line.startswith("+"):
                buf += line[1:].strip()
            else:
                if buf:
                    merged.append(buf)
                buf = line

        if buf:
            merged.append(buf)
        return merged
    
    def parse_file(self, nas_path: Path) -> Dict[str, Any]:
        """
        NASTRAN (.nas) dosyasını parse eder ve mesh verilerini çıkarır.
        GRID kartlarından node'ları, CTRIA3/CQUAD4 kartlarından elementleri okur.

        Dönüş:
            {
              "file_name": str,
              "nodes": Dict[int, Tuple[float,float,float]],
              "elements": Dict[int, Dict[str, Any]]
            }
        """
        nas_path = Path(nas_path)
        adapter = get_context_logger("parser", file_name=nas_path.name)

        nodes : Dict[int, Tuple[float, float, float]] = {}
        elements: Dict[int, Dict[str, Any]] = {}


        try: 
            adapter.info("NASTRAN dosyası okunuyor...")
            text = nas_path.read_text(encoding="utf-8", errors="ignore")
            lines = text.splitlines()
            cards =  self._normalize_cards(lines)

            for ln in cards:
                toks = self._split_card(ln)
                if not toks:
                    continue


                key = toks[0].upper()

                if key == "GRID":
                    # GRID, nid, , x, y, z
                    try:
                        nid = int(toks[1])
                        x = self.safe_float(toks[3])
                        y = self.safe_float(toks[4])
                        z = self.safe_float(toks[5])
                        if None in (x, y, z):
                            raise ValueError("GRID koordinat parse edilemedi")
                        nodes[nid] = (float(x), float(y), float(z))
                    except Exception as e:
                        adapter.warning(f"GRID kartı parse edilemedi | ln='{ln}' | Hata: {e}")
                elif key in ("CTRIA3", "CQUAD4"):
                    # CTRIA3,eid, pid, n1, n2, n3
                    # CQUAD4,eid, pid, n1, n2, n3, n4
                    try:
                        eid = int(toks[1])
                        e_type = "tria" if key == "CTRIA3" else "quad"
                        if e_type == "tria":
                            n1, n2, n3 = int(toks[3]), int(toks[4]), int(toks[5])
                            elements[eid] = {"type": "tria", "nodes": [n1, n2, n3]}
                        else:
                            n1, n2, n3, n4 = int(toks[3]), int(toks[4]), int(toks[5]), int(toks[6])
                            elements[eid] = {"type": "quad", "nodes": [n1, n2, n3, n4]}
                    except Exception as e:
                        adapter.warning(f"{key} kartı parse edilemedi | ln='{ln}' | Hata: {e}")
                elif key in ("BEGIN BULK","ENDDATA","PSHELL","MAT1"):
                    pass
                else:
                    adapter.debug(f"Atlanıyor: {key} | ln='{ln}'")
                    continue
            adapter.info(f"Parse tamam | node={len(nodes)} | element={len(elements)}")
            return {
                "file_name": nas_path.name,
                "nodes": nodes,
                "elements": elements
            }
        except Exception as e:
            adapter.warning(f"parse_file() | Hata: {e}")
            return {
                "file_name": nas_path.name,
                "nodes": {},
                "elements": {}
            }


    