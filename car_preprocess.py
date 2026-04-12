"""cars.csv için notebook ile aynı parse ve dfm oluşturma mantığı."""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

NUMERIC_FEATURES = [
    "yil",
    "km_num",
    "motor_hacmi_num",
    "motor_gucu_num",
    "boyali_sayi",
    "degisen_sayi",
    "tramer_num",
]
CATEGORICAL_FEATURES = [
    "marka",
    "vites_tipi",
    "yakit_tipi",
    "kasa_tipi",
    "cekis",
    "kimden",
]
TARGET = "fiyat_num"


def parse_fiyat(val):
    if pd.isna(val):
        return np.nan
    s = str(val).replace(".", "").replace("TL", "").strip()
    try:
        return int(s)
    except ValueError:
        return np.nan


def parse_km(val):
    if pd.isna(val):
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
    nums = re.findall(r"\d+", s)
    if len(nums) == 2:
        return (int(nums[0]) + int(nums[1])) / 2
    if len(nums) == 1:
        return float(nums[0])
    return np.nan


def parse_motor_gucu(val):
    if pd.isna(val):
        return np.nan
    s = str(val).strip()
    nums = re.findall(r"\d+", s)
    if len(nums) == 2:
        return (int(nums[0]) + int(nums[1])) / 2
    if len(nums) == 1:
        return float(nums[0])
    return np.nan


def parse_yakit_tuketim(val):
    if pd.isna(val):
        return np.nan
    s = str(val).replace(",", ".").replace("lt", "").strip()
    try:
        return float(s)
    except ValueError:
        return np.nan


def parse_yakit_deposu(val):
    if pd.isna(val):
        return np.nan
    s = str(val).replace("lt", "").strip()
    try:
        return int(s)
    except ValueError:
        return np.nan


def parse_boya_degisen(val):
    boyali, degisen = 0, 0
    if pd.isna(val):
        return np.nan, np.nan
    s = str(val).lower().strip()
    if "tamamı orjinal" in s:
        return 0, 0
    if "tamamı boyalı" in s:
        return 10, 0
    if "belirtilmemiş" in s:
        return np.nan, np.nan
    m_boyali = re.search(r"(\d+)\s*boyalı", s)
    m_degisen = re.search(r"(\d+)\s*değişen", s)
    if m_boyali:
        boyali = int(m_boyali.group(1))
    if m_degisen:
        degisen = int(m_degisen.group(1))
    return boyali, degisen


def enrich_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["fiyat_num"] = df["fiyat"].apply(parse_fiyat)
    df["km_num"] = df["kilometre"].apply(parse_km)
    df["motor_hacmi_num"] = df["motor_hacmi"].apply(parse_motor_hacmi)
    df["motor_gucu_num"] = df["motor_gucu"].apply(parse_motor_gucu)
    df["yakit_tuketim_num"] = df["ortalama_yakit_tuketimi"].apply(parse_yakit_tuketim)
    df["yakit_deposu_num"] = df["yakit_deposu"].apply(parse_yakit_deposu)
    boya_parsed = df["boya_degisen"].apply(parse_boya_degisen)
    df["boyali_sayi"] = boya_parsed.apply(lambda x: x[0])
    df["degisen_sayi"] = boya_parsed.apply(lambda x: x[1])
    df["tramer_num"] = df["tramer"].fillna(0)
    return df


def build_modeling_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Notebook ile aynı filtrelerle modelleme DataFrame'i."""
    keep_cols = NUMERIC_FEATURES + CATEGORICAL_FEATURES + [TARGET]
    dfm = df[keep_cols].copy()
    dfm = dfm.dropna(subset=[TARGET])
    dfm = dfm[dfm[TARGET] > 0]
    q_low = dfm[TARGET].quantile(0.01)
    q_high = dfm[TARGET].quantile(0.99)
    dfm = dfm[(dfm[TARGET] >= q_low) & (dfm[TARGET] <= q_high)]
    for col in ("cekis", "kasa_tipi"):
        dfm[col] = dfm[col].replace("-", np.nan)
    return dfm


def load_prepared_frame(csv_path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path, encoding="utf-8")
    df = enrich_dataframe(df)
    return build_modeling_frame(df)


def category_options(dfm: pd.DataFrame) -> dict[str, list[str]]:
    out = {}
    for col in CATEGORICAL_FEATURES:
        vals = sorted(dfm[col].dropna().astype(str).unique().tolist())
        out[col] = vals
    return out
