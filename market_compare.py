"""2025 (cars.csv / sahibinden) vs 2026 (Arabam) piyasa karşılaştırma yardımcıları."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import car_preprocess as cp


def load_2025_frame(csv_path: str | Path) -> pd.DataFrame:
    return cp.load_prepared_frame(csv_path)


def load_2025_frame_wide(csv_path: str | Path) -> pd.DataFrame:
    """Modelleme dar çerçevesi yerine zenginleştirilmiş satırlar (marka+seri, şehir kıyasları için)."""
    df = pd.read_csv(csv_path, encoding="utf-8", low_memory=False)
    df = cp.enrich_dataframe(df)
    df = df[df["fiyat_num"].notna() & (df["fiyat_num"] > 0)].copy()
    ql, qh = df["fiyat_num"].quantile([0.01, 0.99])
    df = df[(df["fiyat_num"] >= ql) & (df["fiyat_num"] <= qh)]
    if "konum" in df.columns:
        df["sehir_parsed"] = df["konum"].astype(str).str.split(",").str[-1].str.strip()
    return df


def load_2026_frame(csv_path: str | Path, **kwargs) -> pd.DataFrame:
    from arabam_preprocess import load_prepared_arabam_frame

    return load_prepared_arabam_frame(csv_path, **kwargs)


def median_compare_by_column(
    df_2025: pd.DataFrame,
    df_2026: pd.DataFrame,
    col: str,
    min_n: int = 30,
) -> pd.DataFrame:
    common = set(df_2025[col].dropna().astype(str).unique()) & set(
        df_2026[col].dropna().astype(str).unique()
    )
    sub25 = df_2025[df_2025[col].astype(str).isin(common)]
    sub26 = df_2026[df_2026[col].astype(str).isin(common)]
    g25 = sub25.groupby(col, observed=False)["fiyat_num"].median()
    g26 = sub26.groupby(col, observed=False)["fiyat_num"].median()
    out = pd.DataFrame({"2025": g25, "2026": g26}).dropna()
    c25 = sub25.groupby(col, observed=False).size()
    c26 = sub26.groupby(col, observed=False).size()
    ok = (c25.reindex(out.index).fillna(0) >= min_n) & (c26.reindex(out.index).fillna(0) >= min_n)
    out = out.loc[ok]
    out["Değişim_%"] = ((out["2026"] / out["2025"]) - 1) * 100
    return out.sort_values("Değişim_%", ascending=False)


def price_band_shares(df: pd.DataFrame) -> pd.Series:
    bins = [0, 250_000, 500_000, 1_000_000, 2_000_000, np.inf]
    labels = ["≤250k", "250k–500k", "500k–1M", "1M–2M", ">2M"]
    x = pd.cut(df["fiyat_num"], bins=bins, labels=labels)
    return x.value_counts(normalize=True).reindex(labels).fillna(0) * 100


def cars_2025_with_city(cars_csv: str | Path) -> pd.DataFrame:
    raw = pd.read_csv(cars_csv, encoding="utf-8", low_memory=False)
    raw = cp.enrich_dataframe(raw)
    raw = raw[raw["fiyat_num"].notna() & (raw["fiyat_num"] > 0)].copy()
    ql, qh = raw["fiyat_num"].quantile([0.01, 0.99])
    raw = raw[(raw["fiyat_num"] >= ql) & (raw["fiyat_num"] <= qh)]
    raw["sehir_parsed"] = raw["konum"].astype(str).str.split(",").str[-1].str.strip()
    return raw


def city_median_compare(
    df_2025_city: pd.DataFrame,
    df_2026: pd.DataFrame,
    min_n: int = 40,
) -> pd.DataFrame:
    common = sorted(
        set(df_2025_city["sehir_parsed"].dropna().unique())
        & set(df_2026["sehir"].dropna().astype(str).unique())
    )
    g25 = df_2025_city[df_2025_city["sehir_parsed"].isin(common)].groupby("sehir_parsed")[
        "fiyat_num"
    ].median()
    g26 = df_2026[df_2026["sehir"].astype(str).isin(common)].groupby(df_2026["sehir"].astype(str))[
        "fiyat_num"
    ].median()
    out = pd.DataFrame({"2025": g25, "2026": g26}).dropna()
    c25 = (
        df_2025_city[df_2025_city["sehir_parsed"].isin(out.index)]
        .groupby("sehir_parsed")
        .size()
    )
    c26 = df_2026[df_2026["sehir"].astype(str).isin(out.index)].groupby(df_2026["sehir"].astype(str)).size()
    ok = (c25.reindex(out.index).fillna(0) >= min_n) & (c26.reindex(out.index).fillna(0) >= min_n)
    out = out.loc[ok]
    out["Değişim_%"] = ((out["2026"] / out["2025"]) - 1) * 100
    return out.sort_values("Değişim_%", ascending=False)


def marka_seri_median_compare(
    df_2025_raw: pd.DataFrame,
    df_2026: pd.DataFrame,
    min_n: int = 25,
) -> pd.DataFrame:
    r = df_2025_raw.copy()
    r["ms"] = r["marka"].astype(str) + " · " + r["seri"].astype(str)
    d6 = df_2026.assign(ms=lambda d: d["marka"].astype(str) + " · " + d["seri"].astype(str))
    common = set(r["ms"].unique()) & set(d6["ms"].unique())
    g25 = r[r["ms"].isin(common)].groupby("ms")["fiyat_num"].median()
    g26 = d6[d6["ms"].isin(common)].groupby("ms")["fiyat_num"].median()
    out = pd.DataFrame({"2025": g25, "2026": g26}).dropna()
    c25 = r[r["ms"].isin(out.index)].groupby("ms").size()
    c26 = d6[d6["ms"].isin(out.index)].groupby("ms").size()
    ok = (c25.reindex(out.index).fillna(0) >= min_n) & (c26.reindex(out.index).fillna(0) >= min_n)
    out = out.loc[ok]
    out["Değişim_%"] = ((out["2026"] / out["2025"]) - 1) * 100
    return out.sort_values("Değişim_%", ascending=False)
