"""Streamlit ile aynı: ortak form girdileri -> Arabam pipeline satiri."""
from __future__ import annotations

from datetime import datetime

from arabam_preprocess import CATEGORICAL_FEATURES as AR_CAT
from car_preprocess import CATEGORICAL_FEATURES as CP_CAT


def pick_category(val: str, options: list[str]) -> str | None:
    if not options:
        return None
    v = (val or "").strip()
    if not v:
        return None
    if v in options:
        return v
    lower_map = {o.lower(): o for o in options}
    if v.lower() in lower_map:
        return lower_map[v.lower()]
    for o in options:
        ol, vl = o.lower(), v.lower()
        if vl in ol or ol in vl:
            return o
    return None


def _default_cat_from_meta(meta_ar: dict, col: str) -> str:
    opts = meta_ar.get(col) or []
    return opts[0] if opts else "Belirtilmemiş"


def common_row_to_arabam_pipeline_row(common: dict, meta_ar: dict) -> dict:
    cy = datetime.now().year
    yil = int(common["yil"])
    km = float(common["km_num"])
    boyali = float(common.get("boyali_sayi", 0) or 0)
    degisen = float(common.get("degisen_sayi", 0) or 0)
    belirsiz = 1.0
    por = 13.0 - boyali - degisen - belirsiz
    if por < 0:
        por = 0.0
        belirsiz = max(0.0, 13.0 - boyali - degisen)

    arac_yasi = float(max(0, cy - yil))
    km_per_year = km / max(arac_yasi, 1.0)

    def map_to_arabam_cat(col: str) -> str:
        opts = meta_ar.get(col) or []
        raw = str(common.get(col, ""))
        p = pick_category(raw, opts)
        return p if p is not None else _default_cat_from_meta(meta_ar, col)

    num_row = {
        "yil": float(yil),
        "km_num": km,
        "motor_hacmi_num": float(common["motor_hacmi_num"]),
        "motor_gucu_num": float(common["motor_gucu_num"]),
        "ort_yakit_tuketimi_num": float(common.get("ort_yakit_tuketimi_num", 7.0)),
        "yakit_deposu_num": float(common.get("yakit_deposu_num", 50.0)),
        "panel_orijinal_n": por,
        "panel_boyali_n": boyali,
        "panel_degisen_n": degisen,
        "panel_belirsiz_n": belirsiz,
        "arac_yasi": arac_yasi,
        "km_per_year": km_per_year,
    }
    cat_row: dict[str, str] = {}
    for col in AR_CAT:
        if col in ("seri", "model"):
            raw = str(common.get(col, "") or "").strip()
            opts = meta_ar.get(col) or []
            if raw:
                p = pick_category(raw, opts)
                cat_row[col] = (
                    p
                    if p is not None
                    else (raw if raw in opts else _default_cat_from_meta(meta_ar, col))
                )
            else:
                cat_row[col] = _default_cat_from_meta(meta_ar, col)
        elif col in CP_CAT:
            cat_row[col] = map_to_arabam_cat(col)
        else:
            cat_row[col] = _default_cat_from_meta(meta_ar, col)
    return {**num_row, **cat_row}
