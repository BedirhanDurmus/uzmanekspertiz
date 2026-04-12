"""
İkinci el araç fiyat tahmini — Streamlit arayüzü (bilgilendirme / karşılaştırmalı analiz).

Türkiye piyasası örneklem verileri üzerinde eğitilmiş ML modelleri; resmi ekspertiz veya ilan platformu onayı değildir.
Ticari markalar yalnızca veri kaynağını tanımlar (kenar çubukta yasal metin).

- cars: `python train_model.py`
- Arabam: `python train_model_arabam.py`
- Çalıştırma: `streamlit run streamlit_app.py`
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import market_compare as mc
import pandas as pd
import streamlit as st

from arabam_preprocess import CATEGORICAL_FEATURES as AR_CAT
from arabam_preprocess import NUMERIC_FEATURES as AR_NUM
from car_preprocess import CATEGORICAL_FEATURES, NUMERIC_FEATURES
from paths import BASE_DIR, data_csv

CSV_ARABAM_DATA = data_csv("arabam.com-otomobil-veri-seti-csv.csv")
CSV_CARS_DATA = data_csv("cars.csv")

ARTIFACTS_CARS = BASE_DIR / "artifacts"
MODEL_CARS = ARTIFACTS_CARS / "best_pipe.joblib"
META_CARS = ARTIFACTS_CARS / "categories.json"
METRICS_CARS = ARTIFACTS_CARS / "metrics.json"

ARTIFACTS_ARABAM = BASE_DIR / "artifacts_arabam"
MODEL_ARABAM = ARTIFACTS_ARABAM / "best_pipe.joblib"
META_ARABAM = ARTIFACTS_ARABAM / "categories.json"
METRICS_ARABAM = ARTIFACTS_ARABAM / "metrics.json"

# Yasal / KVKK: deploy öncesi veri sorumlusu ve aydınlatma metnini kendi ortamınıza göre tamamlayın.
LEGAL_INTRO = (
    "**Türkiye ikinci el otomobil piyasasına** ilişkin, **2025–2026** dönemlerini kapsayan örneklem veriler "
    "üzerinde **dikkatle hazırlanmış** istatistiksel modeller ve **yapay zeka / makine öğrenmesi** yöntemleriyle "
    "çalışan **karşılaştırmalı fiyat tahmin asistanı**. Sonuçlar **bilgilendirme ve akademik karşılaştırma** içindir; "
    "**resmi ekspertiz raporu, bağlayıcı değerleme veya ilan platformu onayı değildir.**"
)

LEGAL_SIDEBAR_SHORT = (
    "Bu arayüz **eğitim ve karşılaştırmalı analiz** amaçlıdır. Tahminler **otomatik model çıktısıdır**; "
    "ekspertiz veya hukuki değer biçme yerine geçmez. Ticari markalar yalnızca veri kaynağını tanımlar; "
    "**ortaklık / sponsorluk / onay yoktur.**"
)

LEGAL_KVKK_FULL = """
**Ticari markalar ve veri kaynakları**  
Platform veya şirket adları (ör. ilan siteleri) yalnızca **hangi örneklem dosyasından** beslenildiğini göstermek içindir. Bu kurumlarla **herhangi bir ticari ilişki, ortaklık, sponsorluk, resmi iş birliği veya ürün tanıtımı** yoktur; markalar ilgili hak sahiplerine aittir.

**Tahminlerin niteliği**  
Gösterilen tutarlar **makine öğrenmesi ile üretilmiş istatistiksel tahminlerdir**. **Bağlayıcı satış teklifi, resmi ekspertiz, sigorta veya hukuki değerleme** olarak kullanılamaz. Gerçek araç fiyatı; ekspertiz, piyasa koşulları ve sözleşmeye bağlıdır.

**Kişisel veri (KVKK)**  
Uygulama, kullanıcı tarafından girilen araç bilgilerini **yalnızca o oturumda tahmin üretmek** için kullanır; kalıcı bir veri tabanına **kayıt taahhüdü yoktur** (deploy ortamınızda sunucu günlükleri, barındırıcı veya analitik araçları için ayrıca aydınlatma metni ve gerekirse açık rıza süreçleri tanımlanmalıdır).

**Veri setleri**  
Kullanılan örneklem dosyaları **toplu ve anonim istatistik** amaçlı işlenmiş veri setleridir; üçüncü kişilere ait kişisel veri işleme iddiası **bu prototip kapsamında** yer almaz.

**Sorumluluk reddi**  
Bu yazılım “olduğu gibi” sunulur; **doğruluk ve güncellik garantisi verilmez**. Üretim ortamında kullanım öncesi **hukuk danışmanlığı** ile metin, çerez ve KVKK uyumu gözden geçirilmelidir.
"""


def render_sidebar_legal() -> None:
    with st.sidebar:
        st.markdown("##### Hakkında")
        st.markdown(LEGAL_SIDEBAR_SHORT)
        with st.expander("Yasal uyarı ve KVKK", expanded=False):
            st.markdown(LEGAL_KVKK_FULL)
        st.caption("Deploy öncesi: veri sorumlusu, aydınlatma metni ve barındırıcı logları için hukuk kontrolü önerilir.")


def fmt_tl(x: float) -> str:
    return f"{int(round(float(x))):,}".replace(",", ".") + " TL"


def fmt_tl_cell(x) -> str:
    if pd.isna(x):
        return "—"
    return fmt_tl(float(x))


@st.cache_data(show_spinner="Piyasa verileri yükleniyor…")
def load_market_pair():
    if not CSV_CARS_DATA.exists() or not CSV_ARABAM_DATA.exists():
        return None
    df_2025 = mc.load_2025_frame(CSV_CARS_DATA)
    df_2026 = mc.load_2026_frame(CSV_ARABAM_DATA, peer_cap_quantile=0.97, peer_cap_mode="drop")
    raw_city = mc.cars_2025_with_city(CSV_CARS_DATA)
    return (df_2025, df_2026, raw_city)


@st.cache_data(show_spinner="Marka / seri / model listesi yükleniyor…")
def load_marka_seri_model_hierarchy() -> pd.DataFrame:
    """Arabam + cars ham CSV’den marka→seri→model benzersiz satırları (Streamlit seçimleri için)."""
    cols = ["marka", "seri", "model"]
    dfs: list[pd.DataFrame] = []
    for p in (CSV_ARABAM_DATA, CSV_CARS_DATA):
        if p.exists():
            try:
                dfs.append(pd.read_csv(p, usecols=cols, encoding="utf-8", low_memory=False))
            except (ValueError, OSError):
                continue
    if not dfs:
        return pd.DataFrame(columns=cols)
    df = pd.concat(dfs, ignore_index=True)
    for c in cols:
        df[c] = df[c].fillna("").astype(str).str.strip()
    df = df[df["marka"] != ""]
    return df.drop_duplicates()


def _hierarchy_seri_for_marka(h: pd.DataFrame, marka: str) -> list[str]:
    if h.empty or not str(marka).strip():
        return []
    sub = h[h["marka"] == str(marka).strip()]
    return sorted({s for s in sub["seri"] if s})


def _hierarchy_models_for(h: pd.DataFrame, marka: str, seri: str) -> list[str]:
    if h.empty or not str(marka).strip():
        return []
    m = str(marka).strip()
    s = str(seri).strip()
    if s:
        sub = h[(h["marka"] == m) & (h["seri"] == s)]
        out = sorted({x for x in sub["model"] if x})
        if out:
            return out
    sub_m = h[h["marka"] == m]
    return sorted({x for x in sub_m["model"] if x})


def split_model_motor_paket(model_str: str) -> tuple[str, str]:
    """Ham `model` metnini motor/versiyon + paket (donanım) olarak ayırır (son boşluktan)."""
    s = (model_str or "").strip()
    if not s:
        return "", ""
    parts = s.rsplit(" ", 1)
    if len(parts) == 1:
        return "", parts[0]
    return parts[0].strip(), parts[1].strip()


def _motor_prefix_to_paket_map(models: list[str]) -> dict[str, list[str]]:
    """prefix -> bu motorda görülen paket sonekleri (ham model satırına uyum için sıralı benzersiz)."""
    buckets: dict[str, list[str]] = {}
    for mod in models:
        pre, suf = split_model_motor_paket(mod)
        buckets.setdefault(pre, [])
        if suf and suf not in buckets[pre]:
            buckets[pre].append(suf)
    for pre in buckets:
        buckets[pre] = sorted(buckets[pre], key=str.lower)
    return buckets


def _resolve_model_from_motor_paket(motor_key: str, paket: str, all_models: list[str]) -> str:
    """Seçilen motor + paket için veri setindeki tam `model` dizesini bul."""
    composed = f"{motor_key} {paket}".strip() if motor_key else paket
    if composed in all_models:
        return composed
    for m in all_models:
        if m.replace("  ", " ").strip() == composed.replace("  ", " ").strip():
            return m
    return composed


_MOTOR_SINGLE = "— (tek parça, paket adı)"


def cascading_marka_seri_model_inputs(
    key_prefix: str,
    marka_options: list[str],
    hierarchy: pd.DataFrame,
) -> tuple[str, str, str]:
    """Marka → seri → (motor/versiyon → paket) veya tek liste model — ham verideki `model` sütununa uyum."""
    if marka_options:
        marka = st.selectbox("Marka", options=marka_options, index=0, key=f"{key_prefix}_marka")
    else:
        marka = st.text_input("Marka", value="", key=f"{key_prefix}_marka")

    seri_opts = _hierarchy_seri_for_marka(hierarchy, marka)
    if seri_opts:
        seri = st.selectbox("Seri", options=seri_opts, index=0, key=f"{key_prefix}_seri")
    else:
        seri = st.text_input("Seri (listedeki marka için kayıt yoksa)", value="", key=f"{key_prefix}_seri")

    mod_opts = _hierarchy_models_for(hierarchy, marka, seri)
    if not mod_opts:
        model = st.text_input("Model (seri/marka için kayıt yoksa)", value="", key=f"{key_prefix}_model")
        return str(marka).strip(), str(seri).strip(), str(model).strip()

    if len(mod_opts) == 1:
        model = mod_opts[0]
        st.caption(f"**Model:** `{model}` *(bu seri için tek kayıt)*")
        return str(marka).strip(), str(seri).strip(), str(model).strip()

    buckets = _motor_prefix_to_paket_map(mod_opts)
    non_empty_prefixes = sorted([p for p in buckets if p], key=str.lower)
    has_single_token = "" in buckets and buckets[""]

    motor_labels: list[str] = []
    motor_to_key: dict[str, str] = {}
    for p in non_empty_prefixes:
        label = f"{p} …"
        motor_labels.append(label)
        motor_to_key[label] = p
    if has_single_token:
        motor_labels.append(_MOTOR_SINGLE)
        motor_to_key[_MOTOR_SINGLE] = ""

    if not motor_labels:
        model = st.selectbox("Model", options=mod_opts, index=0, key=f"{key_prefix}_model")
        return str(marka).strip(), str(seri).strip(), str(model).strip()

    motor_label = st.selectbox(
        "Motor / versiyon",
        options=motor_labels,
        index=0,
        help="İlan metnindeki motor hacmi + tip (ör. 1.6 TDi BlueMotion). Sonraki adımda donanım paketi seçilir.",
        key=f"{key_prefix}_motor",
    )
    mkey = motor_to_key[motor_label]
    paket_opts = buckets.get(mkey, [])
    if not paket_opts:
        model = st.selectbox("Model", options=mod_opts, index=0, key=f"{key_prefix}_model")
        return str(marka).strip(), str(seri).strip(), str(model).strip()

    paket = st.selectbox(
        "Paket / donanım",
        options=paket_opts,
        index=0,
        help="Örn. Comfortline, Trendline, Dream, Elegance — veri setinde bu motorla birlikte geçen seçenekler.",
        key=f"{key_prefix}_paket",
    )
    model = _resolve_model_from_motor_paket(mkey, paket, mod_opts)
    return str(marka).strip(), str(seri).strip(), str(model).strip()


def _fmt_degisim_pct(x: float) -> str:
    return f"{float(x):+.1f}%"


def _display_compare_df(df: pd.DataFrame, index_label: str) -> pd.DataFrame:
    d = df.reset_index()
    d = d.rename(columns={d.columns[0]: index_label})
    d["2025"] = d["2025"].map(fmt_tl_cell)
    d["2026"] = d["2026"].map(fmt_tl_cell)
    d["Değişim_%"] = d["Değişim_%"].map(_fmt_degisim_pct)
    return d


def _display_brand_table(brand: pd.DataFrame) -> pd.DataFrame:
    d = brand.reset_index().rename(columns={"marka": "Marka"})
    d["2025"] = d["2025"].map(fmt_tl_cell)
    d["2026"] = d["2026"].map(fmt_tl_cell)
    d["Δ%"] = d["Δ%"].map(_fmt_degisim_pct)
    return d


@st.cache_resource
def load_joblib(path: str):
    p = Path(path)
    if not p.exists():
        return None
    return joblib.load(p)


@st.cache_data
def load_json_meta(path: str):
    p = Path(path)
    if not p.exists():
        return None
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def load_metrics_uncached(path: Path):
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _pick_category(val: str, options: list[str]) -> str | None:
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
    """cars.csv ile aynı ortak girdiler → Arabam modelinin beklediği tüm sayısal/kategorik sütunlar.

    Ortak olmayan Arabam alanları: ortalama tüketim / depo varsayılanı, panel kalanı = 13 − boyalı − değişen − belirsiz,
    ek kategoriler meta’daki ilk seçenekle doldurulur; seçilen etiketler Arabam sözlüğüne `_pick_category` ile eşlenir.
    """
    from datetime import datetime

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
        p = _pick_category(raw, opts)
        return p if p is not None else _default_cat_from_meta(meta_ar, col)

    num_row = {
        "yil": float(yil),
        "km_num": km,
        "motor_hacmi_num": float(common["motor_hacmi_num"]),
        "motor_gucu_num": float(common["motor_gucu_num"]),
        "ort_yakit_tuketimi_num": 7.0,
        "yakit_deposu_num": 50.0,
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
                p = _pick_category(raw, opts)
                cat_row[col] = (
                    p
                    if p is not None
                    else (raw if raw in opts else _default_cat_from_meta(meta_ar, col))
                )
            else:
                cat_row[col] = _default_cat_from_meta(meta_ar, col)
        elif col in CATEGORICAL_FEATURES:
            cat_row[col] = map_to_arabam_cat(col)
        else:
            cat_row[col] = _default_cat_from_meta(meta_ar, col)
    return {**num_row, **cat_row}


def market_implied_2025_price(
    pred_2026: float, marka: str, loaded: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
) -> tuple[float, str]:
    """2026 tahminini, iki veri setindeki medyan oranıyla 2025 eşdeğerine çevir."""
    df_2025, df_2026, _ = loaded
    m25_g = float(df_2025["fiyat_num"].median())
    m26_g = float(df_2026["fiyat_num"].median())
    if m26_g <= 0 or pd.isna(m26_g):
        return pred_2026, "oran uygulanamadı"
    r_global = m25_g / m26_g
    ma = str(marka).strip()
    if ma:
        sub25 = df_2025[df_2025["marka"].astype(str) == ma]
        sub26 = df_2026[df_2026["marka"].astype(str) == ma]
        if len(sub25) >= 30 and len(sub26) >= 30:
            m25b = float(sub25["fiyat_num"].median())
            m26b = float(sub26["fiyat_num"].median())
            if m26b > 0 and not pd.isna(m26b):
                return float(pred_2026 * (m25b / m26b)), f"**{ma}** markası medyan fiyat oranı (iki örneklem)"
    return float(pred_2026 * r_global), "Tüm ilanlar genel medyan oranı (iki örneklem)"


def render_cars_tab():
    pipe = load_joblib(str(MODEL_CARS))
    meta = load_json_meta(str(META_CARS))
    metrics = load_metrics_uncached(METRICS_CARS)

    if pipe is None or meta is None:
        st.error(
            "cars.csv modeli bulunamadı. Proje klasöründe:\n\n`python train_model.py`"
        )
        st.code(f"cd {BASE_DIR}\npython train_model.py", language="bash")
        return

    st.success("Model yüklendi — **Ağustos 2025** `cars.csv` referans örneklemi.")
    if metrics:
        st.caption(
            f"Doğrulama MAE ≈ **{fmt_tl(metrics['mae_tl'])}** — tahmin bandı buna göre ± gösterilir."
        )

    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("Sayısal bilgiler")
        yil = st.number_input("Model yılı", min_value=1980, max_value=2026, value=2019, step=1, key="c_yil")
        km = st.number_input(
            "Kilometre", min_value=0, max_value=2_000_000, value=85_000, step=1_000, key="c_km"
        )
        motor_cc = st.number_input(
            "Motor hacmi (cc)",
            min_value=0.0,
            max_value=8000.0,
            value=1598.0,
            step=1.0,
            key="c_cc",
        )
        motor_hp = st.number_input(
            "Motor gücü (hp)",
            min_value=0.0,
            max_value=800.0,
            value=150.0,
            step=1.0,
            key="c_hp",
        )
        boyali = st.number_input("Boyalı panel sayısı (tahmini)", min_value=0, max_value=20, value=0, key="c_bo")
        degisen = st.number_input("Değişen parça sayısı (tahmini)", min_value=0, max_value=20, value=0, key="c_de")
        tramer = st.number_input(
            "Tramer tutarı (TL, bilinmiyorsa 0)",
            min_value=0.0,
            max_value=50_000_000.0,
            value=0.0,
            step=1000.0,
            key="c_tr",
        )

    with col_right:
        st.subheader("Kategorik bilgiler")
        h_cat = load_marka_seri_model_hierarchy()
        marka, seri, model = cascading_marka_seri_model_inputs("c", meta["marka"], h_cat)
        st.caption(
            "Marka → **seri** → **motor/versiyon** → **paket** (Comfortline, Dream vb.) adımları birleşik ham veri setlerindeki `model` metninden üretilir. "
            "**Bu sekmedeki** tahminde yalnızca **marka** (+ vites/yakıt/kasa/çekiş/kimden) kullanılır; seri/motor/paket doğrudan girmez."
        )
        vites = st.selectbox("Vites", options=meta["vites_tipi"], key="c_vites")
        yakit = st.selectbox("Yakıt", options=meta["yakit_tipi"], key="c_yakit")
        kasa = st.selectbox("Kasa tipi", options=meta["kasa_tipi"], key="c_kasa")
        cekis = st.selectbox("Çekiş", options=meta["cekis"], key="c_cekis")
        kimden = st.selectbox("Kimden", options=meta["kimden"], key="c_kimden")

    if st.button("Fiyat tahmin et (cars)", type="primary", key="c_btn"):
        row = {
            "yil": int(yil),
            "km_num": float(km),
            "motor_hacmi_num": float(motor_cc),
            "motor_gucu_num": float(motor_hp),
            "boyali_sayi": float(boyali),
            "degisen_sayi": float(degisen),
            "tramer_num": float(tramer),
            "marka": marka,
            "vites_tipi": vites,
            "yakit_tipi": yakit,
            "kasa_tipi": kasa,
            "cekis": cekis,
            "kimden": kimden,
        }
        X = pd.DataFrame([row])[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
        pred = float(pipe.predict(X)[0])
        st.metric("Tahmini fiyat (nokta tahmin)", fmt_tl(pred))

        if metrics:
            mae = float(metrics["mae_tl"])
            rmse = float(metrics.get("rmse_tl", mae * 1.5))
            low = max(0.0, pred - mae)
            high = pred + mae
            st.subheader("Yaklaşık fiyat aralığı")
            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown(f"**± MAE**\n\n**{fmt_tl(low)}** — **{fmt_tl(high)}**")
            with col_b:
                low_rm = max(0.0, pred - rmse)
                high_rm = pred + rmse
                st.markdown(f"**± RMSE**\n\n**{fmt_tl(low_rm)}** — **{fmt_tl(high_rm)}**")
        st.info("Sonuç eğitim verisine dayalıdır; ekspertiz ve piyasa farklılık gösterebilir.")


def render_arabam_tab():
    pipe = load_joblib(str(MODEL_ARABAM))
    meta_ar = load_json_meta(str(META_ARABAM))
    meta_cars = load_json_meta(str(META_CARS))
    metrics = load_metrics_uncached(METRICS_ARABAM)

    if pipe is None or meta_ar is None:
        st.error(
            "2026 güncel model bulunamadı. Proje klasöründe:\n\n`python train_model_arabam.py`"
        )
        st.code(f"cd {BASE_DIR}\npython train_model_arabam.py", language="bash")
        return

    st.success("Model yüklendi — **Mart 2026** güncel ilan örneklemi üzerinde eğitilmiş tahmin.")
    if metrics:
        st.caption(
            f"Doğrulama MAE ≈ **{fmt_tl(metrics['mae_tl'])}** — tahmin bandı buna göre ± gösterilir."
        )
    st.caption(
        "Sayısal alanlar ve temel kategoriler **cars** ile aynı. **Motor** ve **paket** seçimleri birleşerek tam **model** dizesini oluşturur (verideki ilanlarla eşleşir); "
        "şehir vb. ek alanlar meta varsayılanıyla tamamlanır."
    )

    def cat_opts(col: str) -> list[str]:
        if meta_cars and meta_cars.get(col):
            return meta_cars[col]
        return meta_ar.get(col) or []

    col_left, col_right = st.columns(2)
    with col_left:
        st.subheader("Sayısal bilgiler (cars ile aynı)")
        yil = st.number_input("Model yılı", min_value=1980, max_value=2026, value=2019, step=1, key="a_yil")
        km = st.number_input(
            "Kilometre", min_value=0, max_value=2_000_000, value=85_000, step=1_000, key="a_km"
        )
        motor_cc = st.number_input(
            "Motor hacmi (cc)",
            min_value=0.0,
            max_value=8000.0,
            value=1598.0,
            step=1.0,
            key="a_cc",
        )
        motor_hp = st.number_input(
            "Motor gücü (hp)",
            min_value=0.0,
            max_value=800.0,
            value=150.0,
            step=1.0,
            key="a_hp",
        )
        boyali = st.number_input("Boyalı panel sayısı (tahmini)", min_value=0, max_value=20, value=0, key="a_bo")
        degisen = st.number_input("Değişen parça sayısı (tahmini)", min_value=0, max_value=20, value=0, key="a_de")
        tramer = st.number_input(
            "Tramer tutarı (TL, bilinmiyorsa 0)",
            min_value=0.0,
            max_value=50_000_000.0,
            value=0.0,
            step=1000.0,
            key="a_tr",
        )

    with col_right:
        st.subheader("Kategorik bilgiler (cars ile aynı + seri/model)")
        h_cat = load_marka_seri_model_hierarchy()
        om = cat_opts("marka")
        marka, seri, model = cascading_marka_seri_model_inputs("a", om, h_cat)
        vites = st.selectbox(
            "Vites", options=cat_opts("vites_tipi") or ["Belirtilmemiş"], key="a_vites"
        )
        yakit = st.selectbox(
            "Yakıt", options=cat_opts("yakit_tipi") or ["Belirtilmemiş"], key="a_yakit"
        )
        kasa = st.selectbox(
            "Kasa tipi", options=cat_opts("kasa_tipi") or ["Belirtilmemiş"], key="a_kasa"
        )
        cekis = st.selectbox(
            "Çekiş", options=cat_opts("cekis") or ["Belirtilmemiş"], key="a_cekis"
        )
        kimden = st.selectbox(
            "Kimden", options=cat_opts("kimden") or ["Belirtilmemiş"], key="a_kimden"
        )

    if st.button("Fiyat tahmin et (2026 güncel model)", type="primary", key="a_btn"):
        common = {
            "yil": int(yil),
            "km_num": float(km),
            "motor_hacmi_num": float(motor_cc),
            "motor_gucu_num": float(motor_hp),
            "boyali_sayi": float(boyali),
            "degisen_sayi": float(degisen),
            "tramer_num": float(tramer),
            "marka": str(marka),
            "seri": str(seri),
            "model": str(model),
            "vites_tipi": str(vites),
            "yakit_tipi": str(yakit),
            "kasa_tipi": str(kasa),
            "cekis": str(cekis),
            "kimden": str(kimden),
        }
        row_ar = common_row_to_arabam_pipeline_row(common, meta_ar)
        X = pd.DataFrame([row_ar])[AR_NUM + AR_CAT]
        pred_2026 = float(pipe.predict(X)[0])

        st.subheader("2025 referansı vs 2026 (güncel model)")
        ref_note = ""
        pred_2025: float | None = None

        pipe_cars = load_joblib(str(MODEL_CARS))
        if pipe_cars is not None and meta_cars is not None:
            try:
                Xc = pd.DataFrame([common])[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
                pred_2025 = float(pipe_cars.predict(Xc)[0])
                ref_note = (
                    "**2025** sütunu: `cars.csv` (Ağustos 2025) referans modeli — marka ve ortak alanlar (seri/model bu modelde yok). "
                    "**2026** güncel örneklem: seçilen **seri/model** dahil; ek kategoriler meta varsayılanı ve panel/yakıt türevleriyle tamamlanır."
                )
            except Exception:
                pred_2025 = None

        loaded_m = load_market_pair()
        if pred_2025 is None and loaded_m is not None:
            pred_2025, ratio_expl = market_implied_2025_price(pred_2026, str(common.get("marka", "")), loaded_m)
            ref_note = (
                "**2025** sütunu: 2026 tahmini, iki örneklemdeki medyan fiyat oranıyla geriye dönük "
                f"ölçeklendi — {ratio_expl}. Model yerine **piyasa oranı** kullanıldı (`cars.csv` modeli yok veya hata)."
            )
        elif pred_2025 is not None and loaded_m is not None:
            ref_note += (
                " Genel piyasa kırılımları için **2026 piyasa kıyası (2025 ref.)** sekmesine bakın."
            )

        c_left, c_right = st.columns(2)
        if pred_2025 is not None:
            delta_pct = ((pred_2026 / pred_2025) - 1) * 100 if pred_2025 else 0.0
            with c_left:
                st.metric(
                    "Ağustos 2025 (referans)",
                    fmt_tl(pred_2025),
                    help="2025 referans modeli veya medyan oranı",
                )
            with c_right:
                st.metric(
                    "Mart 2026 (güncel)",
                    fmt_tl(pred_2026),
                    delta=f"{delta_pct:+.1f}% vs 2025 ref.",
                )
        else:
            st.metric("Mart 2026 (güncel)", fmt_tl(pred_2026))
            st.warning(
                "2025 referans fiyatı üretilemedi: `artifacts/best_pipe.joblib` veya örneklem veri dosyaları eksik. "
                "`python train_model.py` ve veri dosyalarını kontrol edin."
            )

        if ref_note:
            st.caption(ref_note)

        if metrics:
            mae = float(metrics["mae_tl"])
            rmse = float(metrics.get("rmse_tl", mae * 1.5))
            low = max(0.0, pred_2026 - mae)
            high = pred_2026 + mae
            st.subheader("2026 modeli — yaklaşık aralık (±MAE / ±RMSE)")
            c1, c2 = st.columns(2)
            with c1:
                st.markdown(f"**± MAE**\n\n**{fmt_tl(low)}** — **{fmt_tl(high)}**")
            with c2:
                low_rm = max(0.0, pred_2026 - rmse)
                high_rm = pred_2026 + rmse
                st.markdown(f"**± RMSE**\n\n**{fmt_tl(low_rm)}** — **{fmt_tl(high_rm)}**")
        st.info(
            "**2026** tahmini güncel ilan örnekleminde eğitilmiş modele dayanır. **2025** referansı farklı dönem/kaynak "
            "olduğu için mutlak fiyat değil, **göreli düzey** olarak yorumlayın."
        )


def render_market_tab():
    st.subheader("2026 piyasası — 2025 referansıyla kıyas")
    st.markdown(
        "**Odak:** **Mart 2026** güncel ilan örneklemi (model ile uyumlu ön-işleme). "
        "**Referans:** **Ağustos 2025** tarihli örneklem (`cars.csv`) — dönemler arası dağılım ve yön karşılaştırması. "
        "Farklı kaynak ve örneklem nedeniyle rakamlar **resmi endeks değildir**; **yaklaşık düzey** olarak yorumlanmalıdır."
    )
    loaded = load_market_pair()
    if loaded is None:
        st.warning("`cars.csv` veya `arabam.com-otomobil-veri-seti-csv.csv` bulunamadı.")
        return
    df_2025, df_2026, raw_city = loaded

    m25, m26 = df_2025["fiyat_num"].median(), df_2026["fiyat_num"].median()
    d_pct = ((m26 / m25) - 1) * 100 if m25 else 0.0
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("2026 ilan (güncel örneklem)", f"{len(df_2026):,}")
    k2.metric("Medyan fiyat 2026", fmt_tl(m26), delta=f"{d_pct:+.1f}% vs 2025")
    k3.metric("2025 ilan (cars)", f"{len(df_2025):,}")
    k4.metric("Medyan fiyat 2025 (ref.)", fmt_tl(m25))

    st.divider()
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("##### Ortak markalar — 2026 / 2025 medyan değişimi (%)")
        common_b = sorted(
            set(df_2025["marka"].dropna().unique()) & set(df_2026["marka"].dropna().unique())
        )
        g25 = df_2025[df_2025["marka"].isin(common_b)].groupby("marka")["fiyat_num"].median()
        g26 = df_2026[df_2026["marka"].isin(common_b)].groupby("marka")["fiyat_num"].median()
        brand = pd.DataFrame({"2025": g25, "2026": g26}).dropna()
        brand["Δ%"] = ((brand["2026"] / brand["2025"]) - 1) * 100
        brand = brand.sort_values("Δ%", ascending=False)
        top = pd.concat([brand.head(12), brand.tail(12)]).drop_duplicates()
        st.bar_chart(top.sort_values("Δ%")[["Δ%"]].rename(columns={"Δ%": "Değişim %"}))
        st.dataframe(_display_brand_table(brand.head(40)), use_container_width=True, hide_index=True)

    with c2:
        st.markdown("##### Fiyat bandı — ilan payı (%) (önce 2026)")
        bands = pd.DataFrame({"2026": mc.price_band_shares(df_2026), "2025 ref.": mc.price_band_shares(df_2025)})
        st.dataframe(bands.round(1), use_container_width=True)
        st.markdown("##### Yakıt tipi — medyan ve değişim")
        fuel = mc.median_compare_by_column(df_2025, df_2026, "yakit_tipi", min_n=25)
        st.dataframe(_display_compare_df(fuel, "Yakıt"), use_container_width=True, hide_index=True)

    st.divider()
    st.markdown("##### Kategori kırılımları (ortak etiketler, min. örneklem)")
    t1, t2 = st.columns(2)
    with t1:
        for title, col in [
            ("Vites", "vites_tipi"),
            ("Kasa tipi", "kasa_tipi"),
        ]:
            tbl = mc.median_compare_by_column(df_2025, df_2026, col, min_n=25)
            st.caption(title)
            st.dataframe(_display_compare_df(tbl, title), use_container_width=True, hide_index=True)
    with t2:
        for title, col in [
            ("Çekiş", "cekis"),
            ("Kimden", "kimden"),
        ]:
            tbl = mc.median_compare_by_column(df_2025, df_2026, col, min_n=25)
            st.caption(title)
            st.dataframe(_display_compare_df(tbl, title), use_container_width=True, hide_index=True)

    st.divider()
    st.markdown("##### Şehir (referans `konum` son segmenti × güncel `sehir`)")
    city_tbl = mc.city_median_compare(raw_city, df_2026, min_n=40)
    st.dataframe(_display_compare_df(city_tbl.head(35), "Şehir"), use_container_width=True, hide_index=True)

    st.divider()
    st.markdown("##### Marka · seri (yüksek hacimli ortaklar)")
    ms_tbl = mc.marka_seri_median_compare(raw_city, df_2026, min_n=25)
    st.dataframe(
        _display_compare_df(ms_tbl.head(40), "Marka · seri"),
        use_container_width=True,
        hide_index=True,
    )


def main():
    st.set_page_config(page_title="Araç Fiyat Tahmini (2026)", page_icon="🚗", layout="wide")
    render_sidebar_legal()
    st.title("İkinci el araç fiyat tahmini")
    st.markdown(LEGAL_INTRO)
    st.caption(
        "Teknik: makine öğrenmesi pipeline (ör. XGBoost / HGB). Veri kaynakları sekme başlıklarında; **platform onayı veya ticari bağ yoktur** — ayrıntı için kenar çubuk."
    )

    tab_arabam, tab_market, tab_cars = st.tabs(
        ["2026 tahmin (güncel örneklem)", "2025–2026 piyasa kıyası", "2025 referans modeli"]
    )
    with tab_arabam:
        render_arabam_tab()
    with tab_market:
        render_market_tab()
    with tab_cars:
        render_cars_tab()

    st.divider()
    st.markdown(
        "**Özet:** Bu uygulama, **Türkiye araç piyasası** örneklem verileri üzerinde eğitilmiş modellerle **bilgilendirici tahmin** sunar; "
        "**KVKK**, marka kullanımı ve sorumluluk konularında tam metin **kenar çubuktaki** yasal uyarıdadır."
    )


if __name__ == "__main__":
    main()
