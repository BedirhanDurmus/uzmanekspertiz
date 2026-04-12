"""arabam.com CSV için parse, panel özetleri ve modelleme DataFrame'i."""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

# 13 gövde paneli (ham sütun adları)
PANEL_COLUMNS = [
    "sag_arka_camurluk",
    "arka_kaput",
    "sol_arka_camurluk",
    "sag_arka_kapi",
    "sag_on_kapi",
    "tavan",
    "sol_arka_kapi",
    "sol_on_kapi",
    "sag_on_camurluk",
    "motor_kaputu",
    "sol_on_camurluk",
    "on_tampon",
    "arka_tampon",
]

NUMERIC_FEATURES = [
    "yil",
    "km_num",
    "motor_hacmi_num",
    "motor_gucu_num",
    "ort_yakit_tuketimi_num",
    "yakit_deposu_num",
    "panel_orijinal_n",
    "panel_boyali_n",
    "panel_degisen_n",
    "panel_belirsiz_n",
    "arac_yasi",
    "km_per_year",
]

CATEGORICAL_FEATURES = [
    "marka",
    "seri",
    "model",
    "sehir",
    "ilce",
    "yakit_tipi",
    "vites_tipi",
    "renk",
    "arac_durumu",
    "kasa_tipi",
    "agir_hasarli",
    "kimden",
    "cekis",
    "takasa_uygun",
    "boya_degisen",
]

TARGET = "fiyat_num"

_YEAR_MIN = 1980


def _norm_panel_val(val) -> str:
    if pd.isna(val):
        return "belirsiz"
    s = str(val).strip().lower()
    if "orijinal" in s or "orjinal" in s:
        return "orijinal"
    # arabam.com CSV: "Boyanmış", "Lokal Boyanmış" ("boyalı" metni yok; "boyan" yeterli)
    if "boyal" in s or "boyan" in s:
        return "boyali"
    # "Değişmiş" / "Değişen"
    if "değişen" in s or "degisen" in s or "değişmiş" in s or "degismis" in s:
        return "degisen"
    if "belirtilmemiş" in s or s in ("-", "", "nan"):
        return "belirsiz"
    return "belirsiz"


def parse_fiyat(val):
    if pd.isna(val):
        return np.nan
    if isinstance(val, (int, float)) and not isinstance(val, bool):
        if np.isfinite(val):
            return int(round(float(val)))
        return np.nan
    s = str(val).replace(".", "").replace("TL", "").strip()
    try:
        return int(s)
    except ValueError:
        return np.nan


def parse_km(val):
    if pd.isna(val):
        return np.nan
    if isinstance(val, (int, float)) and not isinstance(val, bool):
        if np.isfinite(val):
            return int(round(float(val)))
        return np.nan
    s = str(val).replace(".", "").replace("km", "").strip()
    try:
        return int(s)
    except ValueError:
        return np.nan


def parse_motor_hacmi(val):
    if pd.isna(val):
        return np.nan
    s = str(val).strip()
    if s in ("-", "", "nan"):
        return np.nan
    nums = re.findall(r"\d+", s)
    if len(nums) >= 2:
        return (int(nums[0]) + int(nums[1])) / 2
    if len(nums) == 1:
        return float(nums[0])
    return np.nan


def parse_motor_gucu(val):
    if pd.isna(val):
        return np.nan
    s = str(val).strip()
    if s in ("-", "", "nan"):
        return np.nan
    nums = re.findall(r"\d+", s)
    if len(nums) >= 2:
        return (int(nums[0]) + int(nums[1])) / 2
    if len(nums) == 1:
        return float(nums[0])
    return np.nan


def parse_yakit_tuketim(val):
    if pd.isna(val):
        return np.nan
    s = str(val).replace(",", ".").replace("lt", "").strip()
    if s in ("-", "", "nan"):
        return np.nan
    try:
        return float(s)
    except ValueError:
        return np.nan


def parse_yakit_deposu(val):
    if pd.isna(val):
        return np.nan
    s = str(val).replace("lt", "").strip()
    if s in ("-", "", "nan"):
        return np.nan
    try:
        return int(float(s))
    except ValueError:
        return np.nan


def _count_panels_row(row: pd.Series) -> tuple[int, int, int, int]:
    o, b, d, u = 0, 0, 0, 0
    for col in PANEL_COLUMNS:
        if col not in row.index:
            continue
        cat = _norm_panel_val(row[col])
        if cat == "orijinal":
            o += 1
        elif cat == "boyali":
            b += 1
        elif cat == "degisen":
            d += 1
        else:
            u += 1
    return o, b, d, u


def enrich_arabam_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["fiyat_num"] = df["fiyat"].apply(parse_fiyat) if "fiyat" in df.columns else np.nan

    if "km" in df.columns:
        df["km_num"] = df["km"].apply(parse_km)
    else:
        df["km_num"] = np.nan

    df["yil"] = pd.to_numeric(df.get("yil"), errors="coerce")

    if "motor_hacmi" in df.columns:
        df["motor_hacmi_num"] = df["motor_hacmi"].apply(parse_motor_hacmi)
    else:
        df["motor_hacmi_num"] = np.nan

    if "motor_gucu" in df.columns:
        df["motor_gucu_num"] = df["motor_gucu"].apply(parse_motor_gucu)
    else:
        df["motor_gucu_num"] = np.nan

    ot = "ort_yakit_tuketimi"
    if ot in df.columns:
        df["ort_yakit_tuketimi_num"] = df[ot].apply(parse_yakit_tuketim)
    else:
        df["ort_yakit_tuketimi_num"] = np.nan

    if "yakit_deposu" in df.columns:
        df["yakit_deposu_num"] = df["yakit_deposu"].apply(parse_yakit_deposu)
    else:
        df["yakit_deposu_num"] = np.nan

    counts = df.apply(_count_panels_row, axis=1, result_type="expand")
    counts.columns = [
        "panel_orijinal_n",
        "panel_boyali_n",
        "panel_degisen_n",
        "panel_belirsiz_n",
    ]
    df = pd.concat([df, counts], axis=1)

    current_year = pd.Timestamp.now().year
    if "yil" in df.columns:
        df["arac_yasi"] = current_year - df["yil"]
        df["arac_yasi"] = df["arac_yasi"].clip(lower=0)
    else:
        df["arac_yasi"] = np.nan

    if "arac_yasi" in df.columns and "km_num" in df.columns:
        df["km_per_year"] = df["km_num"] / df["arac_yasi"].clip(lower=1)
    else:
        df["km_per_year"] = np.nan

    for col in CATEGORICAL_FEATURES:
        if col not in df.columns:
            df[col] = np.nan
        else:
            df[col] = df[col].replace("-", np.nan)

    return df


def _winsorize_target(dfm: pd.DataFrame) -> pd.DataFrame:
    dfm = dfm.dropna(subset=[TARGET])
    dfm = dfm[dfm[TARGET] > 0]
    q_low = dfm[TARGET].quantile(0.01)
    q_high = dfm[TARGET].quantile(0.99)
    return dfm[(dfm[TARGET] >= q_low) & (dfm[TARGET] <= q_high)].copy()


def _peer_price_cap(
    dfm: pd.DataFrame,
    quantile: float = 0.97,
    mode: str = "drop",
) -> pd.DataFrame:
    """Cap or drop rows whose price exceeds peer-group (marka+seri+yil) quantile.

    Parameters
    ----------
    quantile : float
        Upper quantile threshold within each peer group.
    mode : str
        "drop" removes rows above cap; "clip" replaces target with cap value.
    """
    group_cols = ["marka", "seri", "yil"]
    available = [c for c in group_cols if c in dfm.columns]
    if not available or TARGET not in dfm.columns:
        return dfm

    caps = dfm.groupby(available)[TARGET].quantile(quantile)
    caps.name = "_peer_cap"
    dfm = dfm.join(caps, on=available, how="left")

    if mode == "drop":
        dfm = dfm[dfm[TARGET] <= dfm["_peer_cap"]].copy()
    elif mode == "clip":
        mask = dfm[TARGET] > dfm["_peer_cap"]
        dfm.loc[mask, TARGET] = dfm.loc[mask, "_peer_cap"]

    dfm = dfm.drop(columns=["_peer_cap"])
    return dfm


def build_modeling_frame(
    df: pd.DataFrame,
    peer_cap_quantile: float | None = 0.97,
    peer_cap_mode: str = "drop",
) -> pd.DataFrame:
    keep_cols = list(
        dict.fromkeys(NUMERIC_FEATURES + CATEGORICAL_FEATURES + [TARGET, "listing_id"])
    )
    keep_cols = [c for c in keep_cols if c in df.columns]
    dfm = df[keep_cols].copy()

    if "listing_id" in dfm.columns:
        dfm = dfm.drop_duplicates(subset=["listing_id"], keep="last")
        dfm = dfm.drop(columns=["listing_id"])

    dfm = _winsorize_target(dfm)

    if "yil" in dfm.columns:
        y_max = pd.Timestamp.now().year + 1
        dfm = dfm[(dfm["yil"] >= _YEAR_MIN) & (dfm["yil"] <= y_max)]

    if peer_cap_quantile is not None:
        dfm = _peer_price_cap(dfm, quantile=peer_cap_quantile, mode=peer_cap_mode)

    return dfm


def load_prepared_arabam_frame(
    csv_path: str | Path,
    peer_cap_quantile: float | None = 0.97,
    peer_cap_mode: str = "drop",
) -> pd.DataFrame:
    df = pd.read_csv(csv_path, encoding="utf-8", low_memory=False)
    df = enrich_arabam_dataframe(df)
    return build_modeling_frame(
        df,
        peer_cap_quantile=peer_cap_quantile,
        peer_cap_mode=peer_cap_mode,
    )


def category_options(dfm: pd.DataFrame) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for col in CATEGORICAL_FEATURES:
        if col not in dfm.columns:
            out[col] = []
            continue
        vals = sorted(dfm[col].dropna().astype(str).unique().tolist())
        out[col] = vals
    return out
