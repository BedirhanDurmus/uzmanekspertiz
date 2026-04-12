"""Proje kökü ve CSV yolları — veriler `data/` altında; eski kök düzeni için geriye dönük."""
from __future__ import annotations

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"


def data_csv(filename: str) -> Path:
    p = DATA_DIR / filename
    if p.exists():
        return p
    return BASE_DIR / filename
