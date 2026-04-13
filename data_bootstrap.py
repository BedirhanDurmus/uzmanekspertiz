"""Canlı ortamda gitignore'daki Arabam CSV'si yoksa ARABAM_CSV_URL ile indirme."""
from __future__ import annotations

import logging
import os
import urllib.error
import urllib.request
from pathlib import Path

from paths import DATA_DIR, data_csv

log = logging.getLogger(__name__)

ARABAM_CSV_FILENAME = "arabam.com-otomobil-veri-seti-csv.csv"
_arabam_fetch_attempted = False


def arabam_csv_path() -> Path:
    return data_csv(ARABAM_CSV_FILENAME)


def cars_csv_path() -> Path:
    return data_csv("cars.csv")


def ensure_arabam_csv_from_url() -> None:
    """Dosya yoksa ve ARABAM_CSV_URL tanımlıysa data/ altına indirir (Render vb.)."""
    global _arabam_fetch_attempted
    if arabam_csv_path().exists():
        return
    if _arabam_fetch_attempted:
        return
    url = (os.environ.get("ARABAM_CSV_URL") or "").strip()
    if not url:
        return
    _arabam_fetch_attempted = True
    dest = DATA_DIR / ARABAM_CSV_FILENAME
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + ".part")
    try:
        log.warning("Arabam CSV indiriliyor (ilk cold start uzun surebilir): %s", url[:80] + ("..." if len(url) > 80 else ""))
        req = urllib.request.Request(url, headers={"User-Agent": "arabam-proje/1.0 (data bootstrap)"})
        with urllib.request.urlopen(req, timeout=600) as resp, open(part, "wb") as out:
            while True:
                chunk = resp.read(8 * 1024 * 1024)
                if not chunk:
                    break
                out.write(chunk)
        sz = part.stat().st_size
        if sz < 10_000:
            part.unlink(missing_ok=True)
            log.error("Indirilen Arabam CSV cok kucuk (%s bayt); URL veya icerik hatali olabilir.", sz)
            return
        part.replace(dest)
        log.warning("Arabam CSV indirildi: %s bayt -> %s", sz, dest)
    except (urllib.error.URLError, OSError, ValueError) as e:
        part.unlink(missing_ok=True)
        log.error("Arabam CSV indirilemedi: %s", e)
        print(f"[data_bootstrap] Arabam CSV indirilemedi: {e}", flush=True)
