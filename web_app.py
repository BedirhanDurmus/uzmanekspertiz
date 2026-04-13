"""
İkinci el araç fiyat tahmini — Flask web arayüzü.

Çalıştır: python web_app.py
Tarayıcı: http://localhost:5000
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request

import market_compare as mc
from arabam_preprocess import (
    CATEGORICAL_FEATURES as AR_CAT,
    NUMERIC_FEATURES as AR_NUM,
    TARGET,
    load_prepared_arabam_frame,
)
from car_preprocess import CATEGORICAL_FEATURES as CP_CAT, NUMERIC_FEATURES as CP_NUM
from hierarchy_utils import branch_payload_for_marka_seri, load_marka_seri_model_hierarchy, seri_for_marka
from predict_bridge import common_row_to_arabam_pipeline_row, pick_category
from paths import BASE_DIR, data_csv

app = Flask(__name__, template_folder="templates", static_folder="static")
app.config["TEMPLATES_AUTO_RELOAD"] = True
app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 0

if not (BASE_DIR / "static" / "splash-bg.png").is_file():
    import warnings

    warnings.warn(
        "static/splash-bg.png bulunamadi; splash arka plani canlida gorunmez. Dosyayi repoya ekleyip deploy edin.",
        UserWarning,
        stacklevel=1,
    )

# Arayüz sürümü (tarayıcıda kontrol için)
WEB_UI_BUILD = "2026-04-13-v21"

CSV_ARABAM = data_csv("arabam.com-otomobil-veri-seti-csv.csv")
CSV_CARS = data_csv("cars.csv")
ARTIFACTS_DIR = BASE_DIR / "artifacts_arabam"
MODEL_PATH = ARTIFACTS_DIR / "best_pipe.joblib"
META_PATH = ARTIFACTS_DIR / "categories.json"
METRICS_PATH = ARTIFACTS_DIR / "metrics.json"

ARTIFACTS_CARS = BASE_DIR / "artifacts"
MODEL_CARS_PATH = ARTIFACTS_CARS / "best_pipe.joblib"
META_CARS_PATH = ARTIFACTS_CARS / "categories.json"
METRICS_CARS_PATH = ARTIFACTS_CARS / "metrics.json"

_pipe = None
_meta = None
_metrics = None
_dfm = None
_cars_pipe = None
_meta_cars = None
_metrics_cars = None
_hierarchy_df = None
_market_pair = None


def get_pipe():
    global _pipe
    if _pipe is None and MODEL_PATH.exists():
        _pipe = joblib.load(MODEL_PATH)
    return _pipe


def get_meta():
    global _meta
    if _meta is None and META_PATH.exists():
        with open(META_PATH, encoding="utf-8") as f:
            _meta = json.load(f)
    return _meta


def get_metrics():
    global _metrics
    if _metrics is None and METRICS_PATH.exists():
        with open(METRICS_PATH, encoding="utf-8") as f:
            _metrics = json.load(f)
    return _metrics


def get_dfm():
    global _dfm
    if _dfm is None and CSV_ARABAM.exists():
        _dfm = load_prepared_arabam_frame(CSV_ARABAM)
    return _dfm


def get_cars_pipe():
    global _cars_pipe
    if _cars_pipe is None and MODEL_CARS_PATH.exists():
        _cars_pipe = joblib.load(MODEL_CARS_PATH)
    return _cars_pipe


def get_meta_cars():
    global _meta_cars
    if _meta_cars is None and META_CARS_PATH.exists():
        with open(META_CARS_PATH, encoding="utf-8") as f:
            _meta_cars = json.load(f)
    return _meta_cars


def get_metrics_cars():
    global _metrics_cars
    if _metrics_cars is None and METRICS_CARS_PATH.exists():
        with open(METRICS_CARS_PATH, encoding="utf-8") as f:
            _metrics_cars = json.load(f)
    return _metrics_cars


def get_hierarchy_df():
    global _hierarchy_df
    if _hierarchy_df is None:
        _hierarchy_df = load_marka_seri_model_hierarchy(CSV_ARABAM, CSV_CARS)
    return _hierarchy_df


def get_market_pair():
    """(df_2025, df_2026) veya None — referans örneklem ile güncel örneklem kıyası."""
    global _market_pair
    if _market_pair is None:
        if not CSV_CARS.exists() or not CSV_ARABAM.exists():
            _market_pair = False
        else:
            try:
                df_2025 = mc.load_2025_frame(CSV_CARS)
                df_2026 = mc.load_2026_frame(CSV_ARABAM, peer_cap_quantile=0.97, peer_cap_mode="drop")
                _market_pair = (df_2025, df_2026)
            except Exception:
                _market_pair = False
    if _market_pair is False:
        return None
    return _market_pair


def market_implied_ref_price(pred_2026: float, marka: str, df_2025: pd.DataFrame, df_2026: pd.DataFrame) -> tuple[float, str]:
    m25_g = float(df_2025["fiyat_num"].median())
    m26_g = float(df_2026["fiyat_num"].median())
    if m26_g <= 0 or pd.isna(m26_g):
        return pred_2026, "oran uygulanamadi"
    r_global = m25_g / m26_g
    ma = str(marka).strip()
    if ma:
        sub25 = df_2025[df_2025["marka"].astype(str) == ma]
        sub26 = df_2026[df_2026["marka"].astype(str) == ma]
        if len(sub25) >= 30 and len(sub26) >= 30:
            m25b = float(sub25["fiyat_num"].median())
            m26b = float(sub26["fiyat_num"].median())
            if m26b > 0 and not pd.isna(m26b):
                return float(pred_2026 * (m25b / m26b)), "marka_medyan_orani"
    return float(pred_2026 * r_global), "genel_medyan_orani"


@app.after_request
def _no_cache_index(response):
    if request.endpoint == "index":
        response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
        response.headers["Pragma"] = "no-cache"
    return response


@app.route("/")
def index():
    meta = get_meta() or {}
    metrics = get_metrics() or {}
    tpl = BASE_DIR / "templates" / "index.html"
    tpl_mtime = int(tpl.stat().st_mtime) if tpl.exists() else 0
    return render_template(
        "index.html",
        meta=meta,
        metrics=metrics,
        build_id=WEB_UI_BUILD,
        template_mtime=tpl_mtime,
    )


@app.route("/api/meta/arabam")
@app.route("/api/meta/categories")
def api_meta_arabam():
    """Kategoriler (Jinja embed bozuksa veya bos ise tarayici buradan tamamlar)."""
    m = get_meta()
    return jsonify(m if m else {})


@app.route("/api/dashboard")
def api_dashboard():
    dfm = get_dfm()
    if dfm is None:
        return jsonify({"error": "Veri yüklenemedi"}), 500

    stats = {
        "total_listings": int(len(dfm)),
        "median_price": int(dfm[TARGET].median()),
        "mean_price": int(dfm[TARGET].mean()),
        "min_price": int(dfm[TARGET].min()),
        "max_price": int(dfm[TARGET].max()),
        "median_km": int(dfm["km_num"].median()) if "km_num" in dfm.columns else 0,
        "median_year": int(dfm["yil"].median()) if "yil" in dfm.columns else 0,
        "unique_brands": int(dfm["marka"].nunique()) if "marka" in dfm.columns else 0,
    }
    return jsonify(stats)


@app.route("/api/eda/price_distribution")
def api_price_dist():
    dfm = get_dfm()
    if dfm is None:
        return jsonify({"error": "Veri yüklenemedi"}), 500

    prices = dfm[TARGET].dropna()
    bins = [0, 250_000, 500_000, 750_000, 1_000_000, 1_500_000, 2_000_000, 3_000_000, float("inf")]
    labels = ["0-250K", "250K-500K", "500K-750K", "750K-1M", "1M-1.5M", "1.5M-2M", "2M-3M", "3M+"]
    cats = pd.cut(prices, bins=bins, labels=labels)
    counts = cats.value_counts().reindex(labels).fillna(0).astype(int).to_dict()
    return jsonify({"labels": labels, "values": list(counts.values())})


@app.route("/api/eda/km_distribution")
def api_km_dist():
    dfm = get_dfm()
    if dfm is None:
        return jsonify({"error": "Veri yüklenemedi"}), 500

    km = dfm["km_num"].dropna()
    bins = [0, 25_000, 50_000, 100_000, 150_000, 200_000, 300_000, float("inf")]
    labels = ["0-25K", "25K-50K", "50K-100K", "100K-150K", "150K-200K", "200K-300K", "300K+"]
    cats = pd.cut(km, bins=bins, labels=labels)
    counts = cats.value_counts().reindex(labels).fillna(0).astype(int).to_dict()
    return jsonify({"labels": labels, "values": list(counts.values())})


@app.route("/api/eda/year_distribution")
def api_year_dist():
    dfm = get_dfm()
    if dfm is None:
        return jsonify({"error": "Veri yüklenemedi"}), 500

    years = dfm["yil"].dropna().astype(int)
    vc = years.value_counts().sort_index()
    recent = vc[vc.index >= 2005]
    return jsonify({
        "labels": recent.index.tolist(),
        "values": recent.values.tolist(),
    })


@app.route("/api/eda/brand_median")
def api_brand_median():
    dfm = get_dfm()
    if dfm is None:
        return jsonify({"error": "Veri yüklenemedi"}), 500

    brand_stats = dfm.groupby("marka").agg(
        median_price=(TARGET, "median"),
        count=(TARGET, "count"),
    ).reset_index()
    brand_stats = brand_stats[brand_stats["count"] >= 100]
    brand_stats = brand_stats.sort_values("median_price", ascending=False).head(25)
    return jsonify({
        "labels": brand_stats["marka"].tolist(),
        "values": brand_stats["median_price"].astype(int).tolist(),
        "counts": brand_stats["count"].astype(int).tolist(),
    })


@app.route("/api/eda/fuel_type")
def api_fuel_type():
    dfm = get_dfm()
    if dfm is None:
        return jsonify({"error": "Veri yüklenemedi"}), 500

    fuel = dfm["yakit_tipi"].dropna().value_counts()
    return jsonify({
        "labels": fuel.index.tolist(),
        "values": fuel.values.tolist(),
    })


@app.route("/api/eda/transmission")
def api_transmission():
    dfm = get_dfm()
    if dfm is None:
        return jsonify({"error": "Veri yüklenemedi"}), 500

    vites = dfm["vites_tipi"].dropna().value_counts()
    return jsonify({
        "labels": vites.index.tolist(),
        "values": vites.values.tolist(),
    })


@app.route("/api/eda/body_type")
def api_body_type():
    dfm = get_dfm()
    if dfm is None:
        return jsonify({"error": "Veri yüklenemedi"}), 500

    kasa = dfm.groupby("kasa_tipi")[TARGET].median().sort_values(ascending=False)
    kasa = kasa.dropna().head(12)
    return jsonify({
        "labels": kasa.index.tolist(),
        "values": kasa.astype(int).tolist(),
    })


@app.route("/api/eda/correlation")
def api_correlation():
    dfm = get_dfm()
    if dfm is None:
        return jsonify({"error": "Veri yüklenemedi"}), 500

    num_cols = [c for c in AR_NUM if c in dfm.columns] + [TARGET]
    corr = dfm[num_cols].corr()
    target_corr = corr[TARGET].drop(TARGET).sort_values(ascending=False)
    labels: list[str] = []
    values: list[float] = []
    for idx, val in target_corr.items():
        if pd.notna(val) and np.isfinite(float(val)):
            labels.append(str(idx))
            values.append(round(float(val), 4))
    return jsonify({"labels": labels, "values": values})


@app.route("/api/eda/price_vs_year")
def api_price_vs_year():
    dfm = get_dfm()
    if dfm is None:
        return jsonify({"error": "Veri yüklenemedi"}), 500

    yearly = dfm[dfm["yil"] >= 2005].groupby("yil")[TARGET].agg(["median", "mean"]).reset_index()
    return jsonify({
        "labels": yearly["yil"].astype(int).tolist(),
        "median": yearly["median"].astype(int).tolist(),
        "mean": yearly["mean"].astype(int).tolist(),
    })


@app.route("/api/eda/price_vs_km")
def api_price_vs_km():
    dfm = get_dfm()
    if dfm is None:
        return jsonify({"error": "Veri yüklenemedi"}), 500

    sample = dfm[["km_num", TARGET]].dropna()
    if len(sample) > 3000:
        sample = sample.sample(3000, random_state=42)
    return jsonify({
        "km": sample["km_num"].astype(int).tolist(),
        "price": sample[TARGET].astype(int).tolist(),
    })


@app.route("/api/eda/city_top")
def api_city_top():
    dfm = get_dfm()
    if dfm is None:
        return jsonify({"error": "Veri yüklenemedi"}), 500

    city = dfm.groupby("sehir").agg(
        median_price=(TARGET, "median"),
        count=(TARGET, "count"),
    ).reset_index()
    city = city[city["count"] >= 200]
    city = city.sort_values("median_price", ascending=False).head(20)
    return jsonify({
        "labels": city["sehir"].tolist(),
        "values": city["median_price"].astype(int).tolist(),
        "counts": city["count"].astype(int).tolist(),
    })


def _panel_orijinal_block(dfm: pd.DataFrame) -> dict:
    """0–13 orijinal panel adedi vs medyan fiyat (tek grafik)."""
    full_idx = list(range(14))
    col = "panel_orijinal_n"
    if col not in dfm.columns:
        return {"labels": [], "values": [], "counts": [], "subtitle": ""}
    grp = dfm.groupby(col, observed=False)[TARGET].median().sort_index()
    grp = grp[grp.index <= 13]
    cnt = dfm.groupby(col, observed=False)[TARGET].count()
    cnt = cnt[cnt.index <= 13].reindex(full_idx).fillna(0).astype(int)
    g2 = grp.reindex(full_idx)
    return {
        "labels": full_idx,
        "values": [None if pd.isna(x) else int(round(float(x))) for x in g2.tolist()],
        "counts": cnt.tolist(),
        "subtitle": "Orijinal panel adedi (0–13) — medyan fiyat",
    }


@app.route("/api/eda/panel_impact")
def api_panel_impact():
    dfm = get_dfm()
    if dfm is None:
        return jsonify({"error": "Veri yüklenemedi"}), 500
    return jsonify({"panel_orijinal_n": _panel_orijinal_block(dfm)})


def _km_band_label(k: float) -> str:
    if pd.isna(k) or not np.isfinite(float(k)):
        return ""
    v = float(k)
    if v < 50_000:
        return "0-50 bin km"
    if v < 100_000:
        return "50-100 bin km"
    if v < 150_000:
        return "100-150 bin km"
    if v < 200_000:
        return "150-200 bin km"
    if v < 300_000:
        return "200-300 bin km"
    return "300 bin km+"


_KM_BAND_ORDER = [
    "0-50 bin km",
    "50-100 bin km",
    "100-150 bin km",
    "150-200 bin km",
    "200-300 bin km",
    "300 bin km+",
]


def _yil_cohort(y: float) -> str:
    if pd.isna(y) or not np.isfinite(float(y)):
        return ""
    yi = int(float(y))
    if yi < 1990:
        return ""
    if yi >= 2023:
        return "2023+"
    if yi >= 2019:
        return "2019-2022"
    if yi >= 2015:
        return "2015-2018"
    if yi >= 2011:
        return "2011-2014"
    return "1990-2010"


_YIL_COHORT_ORDER = ["2023+", "2019-2022", "2015-2018", "2011-2014", "1990-2010"]


@app.route("/api/eda/buyer_guides")
def api_buyer_guides():
    """Alıcı rehberi: orijinal panel + km / yaş kuşağı / renk / çekiş kırılımları."""
    dfm = get_dfm()
    if dfm is None:
        return jsonify({"error": "Veri yüklenemedi"}), 500

    out: dict = {"panel_orijinal": _panel_orijinal_block(dfm)}

    if "km_num" in dfm.columns:
        tkm = dfm[["km_num", TARGET]].dropna()
        tkm = tkm[tkm["km_num"] >= 0]
        tkm = tkm.assign(_b=tkm["km_num"].map(_km_band_label))
        tkm = tkm[tkm["_b"] != ""]
        gk = tkm.groupby("_b", observed=False)[TARGET].agg(["median", "count"])
        labs_k = [x for x in _KM_BAND_ORDER if x in gk.index and int(gk.loc[x, "count"]) >= 40]
        if not labs_k:
            labs_k = [x for x in _KM_BAND_ORDER if x in gk.index]
        out["km_bands"] = {
            "labels": labs_k,
            "median": [int(round(float(gk.loc[x, "median"]))) for x in labs_k],
            "counts": [int(gk.loc[x, "count"]) for x in labs_k],
        }
    else:
        out["km_bands"] = {"labels": [], "median": [], "counts": []}

    if "yil" in dfm.columns:
        ty = dfm[["yil", TARGET]].dropna()
        ty = ty.assign(_c=ty["yil"].map(_yil_cohort))
        ty = ty[ty["_c"] != ""]
        gy = ty.groupby("_c", observed=False)[TARGET].agg(["median", "count"])
        labs_y = [x for x in _YIL_COHORT_ORDER if x in gy.index and int(gy.loc[x, "count"]) >= 40]
        if not labs_y:
            labs_y = [x for x in _YIL_COHORT_ORDER if x in gy.index]
        out["yil_cohort"] = {
            "labels": labs_y,
            "median": [int(round(float(gy.loc[x, "median"]))) for x in labs_y],
            "counts": [int(gy.loc[x, "count"]) for x in labs_y],
        }
    else:
        out["yil_cohort"] = {"labels": [], "median": [], "counts": []}

    if "renk" in dfm.columns:
        tr = dfm.groupby("renk", observed=False)[TARGET].agg(["median", "count"]).reset_index()
        tr = tr[tr["count"] >= 120].sort_values("count", ascending=False).head(10)
        out["renk_top"] = {
            "labels": tr["renk"].astype(str).tolist(),
            "median": tr["median"].astype(int).tolist(),
            "counts": tr["count"].astype(int).tolist(),
        }
    else:
        out["renk_top"] = {"labels": [], "median": [], "counts": []}

    if "cekis" in dfm.columns:
        tc = dfm[["cekis", TARGET]].dropna()
        tc = tc[tc["cekis"].astype(str).str.strip() != ""]
        gc = tc.groupby("cekis", observed=False)[TARGET].agg(["median", "count"])
        gc = gc[gc["count"] >= 80].sort_values("median", ascending=False).head(8)
        out["cekis"] = {
            "labels": gc.index.astype(str).tolist(),
            "median": gc["median"].astype(int).tolist(),
            "counts": gc["count"].astype(int).tolist(),
        }
    else:
        out["cekis"] = {"labels": [], "median": [], "counts": []}

    return jsonify(out)


@app.route("/api/eda/seller_type")
def api_seller_type():
    dfm = get_dfm()
    if dfm is None:
        return jsonify({"error": "Veri yüklenemedi"}), 500

    kimden = dfm.groupby("kimden").agg(
        median_price=(TARGET, "median"),
        count=(TARGET, "count"),
    ).reset_index()
    kimden = kimden.sort_values("median_price", ascending=False)
    return jsonify({
        "labels": kimden["kimden"].tolist(),
        "median": kimden["median_price"].astype(int).tolist(),
        "counts": kimden["count"].astype(int).tolist(),
    })


@app.route("/api/eda/age_vs_price")
def api_age_vs_price():
    dfm = get_dfm()
    if dfm is None:
        return jsonify({"error": "Veri yüklenemedi"}), 500

    age = dfm[dfm["arac_yasi"] <= 30].groupby("arac_yasi")[TARGET].median().sort_index()
    return jsonify({
        "labels": age.index.astype(int).tolist(),
        "values": age.astype(int).tolist(),
    })


def _serialize_median_compare(tab: pd.DataFrame, k: int = 14) -> dict:
    if tab is None or len(tab) == 0:
        return {"labels": [], "m25": [], "m26": [], "delta_pct": []}
    top = tab.head(k)
    return {
        "labels": [str(x) for x in top.index.tolist()],
        "m25": [int(round(float(v))) for v in top["2025"].tolist()],
        "m26": [int(round(float(v))) for v in top["2026"].tolist()],
        "delta_pct": [round(float(v), 2) for v in top["Değişim_%"].tolist()],
    }


@app.route("/api/eda/cross_market")
def api_eda_cross_market():
    """Referans örneklem (~Ağu 2025) ile güncel örneklem (Mart 2026) — EDA için tek paket."""
    try:
        pair = get_market_pair()
        if pair is None:
            return jsonify(
                {
                    "ok": False,
                    "error": "Referans veya güncel veri seti bulunamadı. data/ klasöründe gerekli dosyaların yer aldığından emin olun.",
                }
            )

        df25, df26 = pair
        m25 = float(df25[TARGET].median())
        m26 = float(df26[TARGET].median())
        d_pct = ((m26 / m25) - 1.0) * 100.0 if m25 > 0 else 0.0

        out: dict = {
            "ok": True,
            "ref_label": "Referans örneklem (~Ağustos 2025)",
            "cur_label": "Güncel örneklem (Mart 2026)",
            "overview": {
                "n_ref": int(len(df25)),
                "n_cur": int(len(df26)),
                "median_ref_tl": int(round(m25)),
                "median_cur_tl": int(round(m26)),
                "delta_pct_median": round(d_pct, 2),
                "median_km_ref": int(round(float(df25["km_num"].median()))) if "km_num" in df25.columns else None,
                "median_km_cur": int(round(float(df26["km_num"].median()))) if "km_num" in df26.columns else None,
                "median_yil_ref": int(round(float(df25["yil"].median()))) if "yil" in df25.columns else None,
                "median_yil_cur": int(round(float(df26["yil"].median()))) if "yil" in df26.columns else None,
            },
        }

        try:
            bands25 = mc.price_band_shares(df25)
            bands26 = mc.price_band_shares(df26)
            labels = [str(x) for x in bands25.index.tolist()]
            b26 = bands26.reindex(bands25.index).fillna(0)
            out["price_bands"] = {
                "labels": labels,
                "ref_pct": [round(float(bands25[l]), 2) for l in bands25.index],
                "cur_pct": [round(float(b26.loc[l]), 2) for l in bands25.index],
            }
        except Exception:
            out["price_bands"] = {"labels": [], "ref_pct": [], "cur_pct": []}

        for col, key, mn in [
            ("yakit_tipi", "yakit", 30),
            ("kasa_tipi", "kasa", 25),
            ("vites_tipi", "vites", 25),
            ("marka", "marka", 80),
        ]:
            try:
                if col not in df25.columns or col not in df26.columns:
                    out[key] = {"labels": [], "m25": [], "m26": [], "delta_pct": []}
                    continue
                tab = mc.median_compare_by_column(df25, df26, col, min_n=mn)
                lim = 16 if col == "marka" else 12
                out[key] = _serialize_median_compare(tab, lim)
            except Exception:
                out[key] = {"labels": [], "m25": [], "m26": [], "delta_pct": []}

        try:
            df25w = mc.load_2025_frame_wide(CSV_CARS)
            ms = mc.marka_seri_median_compare(df25w, df26, min_n=22)
            out["marka_seri"] = _serialize_median_compare(ms, 18)
            cmc = mc.city_median_compare(df25w, df26, min_n=40)
            out["cities"] = _serialize_median_compare(cmc, 14)
        except Exception:
            out["marka_seri"] = {"labels": [], "m25": [], "m26": [], "delta_pct": []}
            out["cities"] = {"labels": [], "m25": [], "m26": [], "delta_pct": []}

        if "tramer_num" in df25.columns:
            try:
                out["overview"]["median_tramer_ref_tl"] = int(round(float(df25["tramer_num"].median())))
            except Exception:
                out["overview"]["median_tramer_ref_tl"] = None
        else:
            out["overview"]["median_tramer_ref_tl"] = None

        return jsonify(out)
    except Exception as ex:
        return jsonify({"ok": False, "error": str(ex)}), 200


def _cars_row_from_common(common: dict, meta_cars: dict) -> dict:
    row: dict = {}
    for col in CP_NUM:
        if col == "tramer_num":
            row[col] = float(common.get("tramer_num", 0) or 0)
        elif col == "yil":
            row[col] = int(common["yil"])
        else:
            row[col] = float(common[col])
    for col in CP_CAT:
        opts = meta_cars.get(col) or []
        v = str(common.get(col, "") or "").strip()
        p = pick_category(v, opts)
        row[col] = p if p is not None else (opts[0] if opts else "")
    return row


@app.route("/api/hierarchy/marka")
def api_hierarchy_marka():
    meta = get_meta()
    if meta and meta.get("marka"):
        return jsonify({"markalar": meta["marka"]})
    h = get_hierarchy_df()
    if h.empty:
        return jsonify({"markalar": []})
    return jsonify({"markalar": sorted(h["marka"].dropna().unique().tolist())})


@app.route("/api/hierarchy/seri")
def api_hierarchy_seri():
    marka = request.args.get("marka", "").strip()
    h = get_hierarchy_df()
    return jsonify({"seriler": seri_for_marka(h, marka)})


@app.route("/api/hierarchy/branch")
def api_hierarchy_branch():
    marka = request.args.get("marka", "").strip()
    seri = request.args.get("seri", "").strip()
    h = get_hierarchy_df()
    return jsonify(branch_payload_for_marka_seri(h, marka, seri))


@app.route("/api/period_compare/summary")
def api_period_compare_summary():
    pair = get_market_pair()
    if pair is None:
        return jsonify({"ok": False, "error": "Referans veya güncel veri seti eksik."}), 400
    df25, df26 = pair
    m25 = float(df25["fiyat_num"].median())
    m26 = float(df26["fiyat_num"].median())
    d_pct = ((m26 / m25) - 1.0) * 100.0 if m25 else 0.0
    return jsonify(
        {
            "ok": True,
            "ref_label": "Referans örneklem (Ağustos 2025) — yaklaşık 6 ay önceki piyasa",
            "current_label": "Güncel örneklem (Mart 2026)",
            "n_ref": int(len(df25)),
            "n_current": int(len(df26)),
            "median_ref_tl": int(round(m25)),
            "median_current_tl": int(round(m26)),
            "median_change_pct": round(d_pct, 2),
        }
    )


@app.route("/api/period_compare/brands")
def api_period_compare_brands():
    pair = get_market_pair()
    if pair is None:
        return jsonify({"ok": False}), 400
    df25, df26 = pair
    common_b = sorted(
        set(df25["marka"].dropna().astype(str).unique()) & set(df26["marka"].dropna().astype(str).unique())
    )
    g25 = df25[df25["marka"].isin(common_b)].groupby("marka")["fiyat_num"].median()
    g26 = df26[df26["marka"].isin(common_b)].groupby("marka")["fiyat_num"].median()
    brand = pd.DataFrame({"ref": g25, "current": g26}).dropna()
    brand["delta_pct"] = ((brand["current"] / brand["ref"]) - 1.0) * 100.0
    brand = brand.sort_values("delta_pct", ascending=False)
    top = brand.head(20)
    return jsonify(
        {
            "ok": True,
            "labels": top.index.tolist(),
            "median_ref": top["ref"].astype(int).tolist(),
            "median_current": top["current"].astype(int).tolist(),
            "delta_pct": top["delta_pct"].round(2).tolist(),
        }
    )


@app.route("/api/predict", methods=["POST"])
def api_predict():
    pipe = get_pipe()
    meta = get_meta()
    metrics = get_metrics()

    if pipe is None or meta is None:
        return jsonify({"error": "Model yüklenemedi. Yerel eğitim çıktılarını oluşturup uygulamayı yeniden başlatın."}), 500

    data = request.json or {}

    yil = int(data.get("yil", 2020))
    km = float(data.get("km_num", 100000))
    motor_hacmi = float(data.get("motor_hacmi_num", 1600))
    motor_gucu = float(data.get("motor_gucu_num", 120))
    boyali = float(data.get("panel_boyali_n", 0))
    degisen = float(data.get("panel_degisen_n", 0))
    tramer = float(data.get("tramer_num", 0) or 0)

    common = {
        "yil": yil,
        "km_num": km,
        "motor_hacmi_num": motor_hacmi,
        "motor_gucu_num": motor_gucu,
        "boyali_sayi": boyali,
        "degisen_sayi": degisen,
        "tramer_num": tramer,
        "marka": str(data.get("marka", "") or "").strip(),
        "seri": str(data.get("seri", "") or "").strip(),
        "model": str(data.get("model", "") or "").strip(),
        "vites_tipi": str(data.get("vites_tipi", "") or ""),
        "yakit_tipi": str(data.get("yakit_tipi", "") or ""),
        "kasa_tipi": str(data.get("kasa_tipi", "") or ""),
        "cekis": str(data.get("cekis", "") or ""),
        "kimden": str(data.get("kimden", "") or ""),
        "renk": str(data.get("renk", "") or ""),
        "arac_durumu": str(data.get("arac_durumu", "") or ""),
        "agir_hasarli": str(data.get("agir_hasarli", "") or ""),
        "takasa_uygun": str(data.get("takasa_uygun", "") or ""),
        "boya_degisen": str(data.get("boya_degisen", "") or ""),
        "ort_yakit_tuketimi_num": float(data.get("ort_yakit_tuketimi_num", 7.0)),
        "yakit_deposu_num": float(data.get("yakit_deposu_num", 50.0)),
    }

    row_ar = common_row_to_arabam_pipeline_row(common, meta)
    X = pd.DataFrame([row_ar])[AR_NUM + AR_CAT]
    pred_2026 = float(pipe.predict(X)[0])

    result: dict = {
        "prediction": int(round(pred_2026)),
        "prediction_label": "Güncel tahmin (Mart 2026 örneklem modeli)",
        "tramer_num": int(round(tramer)),
        "arabam_model_uses_tramer": False,
        "cars_ref_model_uses_tramer": True,
    }

    pred_ref = None
    ref_source = None
    meta_cars = get_meta_cars()
    cars_pipe = get_cars_pipe()
    if cars_pipe is not None and meta_cars is not None:
        try:
            row_cars = _cars_row_from_common(common, meta_cars)
            Xc = pd.DataFrame([row_cars])[CP_NUM + CP_CAT]
            pred_ref = float(cars_pipe.predict(Xc)[0])
            ref_source = "cars_model_aug2025"
        except Exception:
            pred_ref = None

    loaded = get_market_pair()
    ratio_kind = None
    if pred_ref is None and loaded is not None:
        df25, df26 = loaded
        pred_ref, ratio_kind = market_implied_ref_price(pred_2026, common["marka"], df25, df26)
        ref_source = ratio_kind

    if pred_ref is not None:
        result["prediction_ref_period"] = int(round(pred_ref))
        if ref_source == "cars_model_aug2025":
            result["ref_period_label"] = "Yaklaşık 6 ay önce (referans model, Ağustos 2025; tramer dahil)"
        elif ref_source == "marka_medyan_orani":
            result["ref_period_label"] = "Yaklaşık eski dönem (piyasa medyan oranı, marka bazlı)"
        else:
            result["ref_period_label"] = "Yaklaşık eski dönem (piyasa medyan oranı, genel)"
        delta_pct = ((pred_2026 / pred_ref) - 1.0) * 100.0 if pred_ref else 0.0
        result["delta_pct_vs_ref"] = round(delta_pct, 2)
        result["delta_tl_vs_ref"] = int(round(pred_2026 - pred_ref))
        result["value_change_word"] = "yükseldi" if pred_2026 >= pred_ref else "düştü"
    else:
        result["prediction_ref_period"] = None
        result["ref_period_label"] = None
        result["ref_note"] = "Referans tahmini için referans veri seti ve eğitilmiş referans modeli gerekir."

    if metrics:
        mae = float(metrics["mae_tl"])
        rmse = float(metrics.get("rmse_tl", mae * 1.5))
        result["mae"] = int(round(mae))
        result["rmse"] = int(round(rmse))
        result["low_mae"] = int(round(max(0, pred_2026 - mae)))
        result["high_mae"] = int(round(pred_2026 + mae))
        result["low_rmse"] = int(round(max(0, pred_2026 - rmse)))
        result["high_rmse"] = int(round(pred_2026 + rmse))

    return jsonify(result)


@app.route("/api/model_comparison")
def api_model_comparison():
    """Notebook'taki model karşılaştırma tablosu (sabit değerler — notebook çıktısı)."""
    models = [
        {"name": "DummyRegressor (mean)", "mae": 292_871, "rmse": 404_399, "r2": 0.0, "mape": 60.1, "cv_mae": None},
        {"name": "Ridge", "mae": 187_073, "rmse": 272_651, "r2": 0.545, "mape": 33.5, "cv_mae": None},
        {"name": "ElasticNet", "mae": 190_811, "rmse": 276_223, "r2": 0.533, "mape": 34.6, "cv_mae": None},
        {"name": "RandomForest", "mae": 72_405, "rmse": 115_965, "r2": 0.918, "mape": 11.3, "cv_mae": None},
        {"name": "HistGradientBoosting", "mae": 66_489, "rmse": 108_206, "r2": 0.928, "mape": 10.1, "cv_mae": 67_728},
        {"name": "XGBoost", "mae": 64_877, "rmse": 105_788, "r2": 0.932, "mape": 9.8, "cv_mae": 66_102},
        {"name": "XGBoost (tuned)", "mae": 62_100, "rmse": 102_600, "r2": 0.936, "mape": 9.4, "cv_mae": 64_500},
    ]
    return jsonify(models)


if __name__ == "__main__":
    print("Veri on-yukleniyor (ilk baslatmada ~30s surebilir)...")
    get_dfm()
    get_pipe()
    get_meta()
    get_metrics()
    tpl = BASE_DIR / "templates" / "index.html"
    print(f"WEB_UI_BUILD={WEB_UI_BUILD}")
    print(f"web_app.py: {Path(__file__).resolve()}")
    print(f"Sablon: {tpl.resolve()} (var mi: {tpl.exists()})")
    print("Hazir! http://localhost:5000 — tarayicida Ctrl+F5 ile sert yenileyin.")
    app.run(debug=False, host="0.0.0.0", port=5000)
