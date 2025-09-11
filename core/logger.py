# core/logger.py
import logging, os
from logging.handlers import RotatingFileHandler
from pathlib import Path

LOG_DIR = Path(__file__).resolve().parents[1] / "logs"
LOG_DIR.mkdir(exist_ok=True)
LOG_FILE = LOG_DIR / "app.log"

_logger = None

def get_logger(name="gv"):
    global _logger
    if _logger:
        return _logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    # evita duplicati
    if not logger.handlers:
        fh = RotatingFileHandler(LOG_FILE, maxBytes=300_000, backupCount=3, encoding="utf-8")
        fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    _logger = logger
    return logger
