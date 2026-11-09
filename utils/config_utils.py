import yaml 
from pathlib import Path
from typing import Union

ROOT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = ROOT_DIR / "config.yaml"

def load_config(config_path: Union[str,Path] = DEFAULT_CONFIG_PATH) -> dict:
    """
    Config.yaml dosyasını yükler ve içindeki göreceli dosya yollarını
    mutlak (absolute) yollara çevirir.
    
    Örnek: './Data' -> 'C:/Users/.../Data'
    
    Args:
        config_path (str/Path): Yüklenecek YAML dosyasının yolu.
    Returns:
        dict: İşlenmiş konfigürasyon sözlüğü.
    """
    config_path = Path(config_path).resolve()
    with open(config_path, "r" ,encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Config içindeki 'dir' veya 'output' geçen anahtarları bulup
    # Path objesine çevirerek tam yol (full path) haline getirir.
    for k, v in config.items():
        if isinstance(v,str) and ("dir" in k or "output" in k):
            config[k] = str(Path(v).resolve())
    
    return config