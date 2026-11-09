import logging
from pathlib import Path
from logging.handlers import RotatingFileHandler
from typing import Optional, Literal, Union, Dict, Any
from utils.config_utils import load_config

_config = None

def _get_config():
    global _config
    if _config is None:
        _config = load_config()
    return _config

# Varsayılanlar ve formatlar
DEFAULT_EXTRAS: Dict[str, Any] = {"file_name": "-"}
CONSOLE_FMT = "%(levelname)s | %(file_name)s | %(message)s"
FILE_FMT = "%(asctime)s | %(name)s | %(levelname)s | %(file_name)s | %(message)s"
DEFAULT_LOG_FILE = "general.log"


class SafeExtraFormatter(logging.Formatter):
    """
    Formatter string'inde bulunan extra alanlar (örn. %(file_name)s)
    record içinde yoksa varsayılanlarla doldurur.
    """

    def __init__(
        self,
        fmt: Optional[str] = None,
        datefmt: Optional[str] = None,
        style: Literal["%", "{", "$"] = "%",
        default_extras: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        SafeExtraFormatter başlatıcı.
        
        Args:
            default_extras: Log kaydında bulunmayan anahtarlar için varsayılan değerler.
        """
        super().__init__(fmt=fmt, datefmt=datefmt, style=style)
        self.default_extras = default_extras or {}

    def format(self, record: logging.LogRecord) -> str:
        """
        Log kaydını formatlar, eksik extra alanları doldurur.
        """
        for k, v in self.default_extras.items():
            if not hasattr(record, k):
                setattr(record, k, v)
        return super().format(record)


def _resolve_log_dir(log_dir: Optional[Union[str, Path]]) -> Path:
    """
    Config'ten veya parametreden log klasörünü güvenli biçimde belirler.
    """
    if log_dir is None:
        raw = _get_config().get("logs_dir")
        if not raw:
            raw = "logs"
        log_dir = raw
    p = Path(log_dir)
    p.mkdir(parents=True, exist_ok=True)
    return p


def get_logger(
    name: str,
    log_dir: Optional[Union[str, Path]] = None,
    console_lvl: int = logging.INFO,
    file_lvl: Optional[int] = logging.WARNING,
    log_file_name: str = DEFAULT_LOG_FILE,
) -> logging.Logger:
    """
    Ana Logger oluşturucu.
    
    İşlevleri:
    1. Konsola renkli ve basit formatta log basar (INFO seviyesi).
    2. Dosyaya tarih ve detay içeren formatta log basar (WARNING seviyesi).
    3. Log dosyasının boyutu 5MB'ı geçerse yedekleyip yenisini açar (RotatingFileHandler).
    
    Args:
        name: Logger adı.
        log_dir: Logların kaydedileceği dizin.
        console_lvl: Konsol log seviyesi (Varsayılan: INFO).
        file_lvl: Dosya log seviyesi (Varsayılan: WARNING).
        log_file_name: Log dosya adı.
    Returns:
        logging.Logger: Yapılandırılmış logger nesnesi.
    """
    target_dir = _resolve_log_dir(log_dir)
    log_path = target_dir / log_file_name

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    # Yinelenen handler'ları önlemek için temizle
    if logger.hasHandlers():
        logger.handlers.clear()

    # Konsol handler
    ch = logging.StreamHandler()
    ch.setLevel(console_lvl)
    ch.setFormatter(
        SafeExtraFormatter(CONSOLE_FMT, default_extras=DEFAULT_EXTRAS)
    )
    logger.addHandler(ch)

    # Dosya handler (opsiyonel)
    if file_lvl is not None:
        fh = RotatingFileHandler(
            log_path,
            maxBytes=5 * 1024 * 1024,  # 5MB
            backupCount=3,
            encoding="utf-8",
        )
        fh.setLevel(file_lvl)
        fh.setFormatter(
            SafeExtraFormatter(FILE_FMT, default_extras=DEFAULT_EXTRAS)
        )
        logger.addHandler(fh)

    return logger


def get_context_logger(
    name: str,
    file_name: Optional[str] = None,
    *,
    log_dir: Optional[Union[str, Path]] = None,
    console_lvl: int = logging.INFO,
    file_lvl: Optional[int] = logging.WARNING,
    log_file_name: str = DEFAULT_LOG_FILE,

) -> logging.LoggerAdapter:
    """
    file_name gibi bağlam bilgilerini otomatik ekleyen LoggerAdapter döndürür.
    """
    base = get_logger(
        name,
        log_dir=log_dir,
        console_lvl=console_lvl,
        file_lvl=file_lvl,
        log_file_name=log_file_name
    )
    return logging.LoggerAdapter(base, {"file_name": file_name or DEFAULT_EXTRAS["file_name"]})