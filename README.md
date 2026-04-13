# İkinci el araç fiyat analizi ve tahmini

2025 Ağustos ve 2026 Mart Türkiye Yerel araç ilan veri setleri üzerinde **keşifsel veri analizi (EDA)**, **model karşılaştırması**, **dönemler arası piyasa kıyası** ve **makine öğrenmesi ile fiyat tahmini** sunan Flask tabanlı bir web arayüzü. Marmara Üniversitesi Veri Madenciliği dersi kapsamında geliştirilmiştir.

**Hazırlayan: Bedirhan Durmus**

<p align="center">
  <img src="static/splash-bg.png" alt="Uygulama açılış ekranı — koyu tema ve araç görseli" width="92%" />
  <br />
  <em>Açılış ekranı (splash) — arayüz aynı görsel dilde devam eder.</em>
</p>

---

## Özellikler

| Alan | Açıklama |
|------|----------|
| **Dashboard** | Özet KPI’lar, fiyat dağılımı, yakıt tipi, yıla göre fiyat, marka bazlı medyanlar |
| **EDA** | Korelasyon, scatter, km / yıl / vites / kasa / satıcı tipi dağılımları, şehir ve yaş grafikleri, **alıcı rehberi** (panel, km bandı, model yılı, renk, çekiş) |
| **Model karşılaştırması** | Notebook ile uyumlu metrik tablosu ve MAE / R² grafikleri |
| **Fiyat tahmini** | Eğitilmiş pipeline ile güncel tahmin; varsa referans dönem tahmini ve karşılaştırma |
| **~6 ay piyasa kıyası** | Referans ve güncel örneklem için çapraz özet ve grafikler (veri dosyaları mevcutsa) |

Arayüz metinleri ticari marka veya ham dosya adlarına dayanmayacak şekilde nötrleştirilmiştir; model ve `categories.json` ile uyum için ham kategori değerleri sunucu tarafında korunur.

---

## Teknoloji yığını

- **Python 3.10+** (önerilir)
- **Flask** — REST API + Jinja şablonu
- **pandas**, **numpy**, **scikit-learn**, **joblib**
- **XGBoost** (eğitim ve karşılaştırma için)
- **Chart.js** (CDN) — tarayıcıda grafikler

**Canlı ortam (Render vb.):** `web_app.py` + `gunicorn` — `render.yaml`.

---

## Kurulum

```bash
cd arabam_proje
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Linux / macOS
pip install -r requirements.txt
```

---

## Veri dosyaları

CSV dosyaları öncelikle **`data/`** klasörüne konur; yoksa proje köküne bakılır (`paths.py`).

| Dosya (örnek ad) | Rol |
|------------------|-----|
| Güncel örneklem CSV | Ana EDA ve güncel model eğitimi |
| Referans örneklem CSV | Referans model, dönem kıyası, çapraz piyasa |

Dosya adlarını `data/` içine koyduktan sonra `web_app.py` içindeki sabitlerle veya eğitim scriptlerindeki yollarla uyumlu olduğundan emin olun.

---

## Model eğitimi

**Güncel örneklem modeli** (çıktı: `artifacts_arabam/`):

```bash
python train_model_arabam.py
```

**Referans örneklem modeli** (çıktı: `artifacts/`):

```bash
python train_model.py
```

Oluşması beklenen dosyalar (özet): `best_pipe.joblib`, `categories.json`, `metrics.json`.

---

## Web uygulamasını çalıştırma

```bash
python web_app.py
```

Tarayıcı: **http://127.0.0.1:5000**

- İlk açılışta veri ön-yüklemesi bir süre sürebilir (konsolda bilgi verilir).
- Arayüz sürümü `WEB_UI_BUILD` ile işaretlenir; güncelleme sonrası tarayıcıda **Ctrl+F5** ile sert yenileme önerilir.
- **EDA** sekmesindeki grafikler (alıcı rehberi dahil), sekme ilk kez açıldığında yüklenir; böylece gizli sekmede sıfır boyutlu grafik oluşması engellenir.

### Önemli API uçları (özet)

- `GET /api/dashboard` — özet istatistikler  
- `GET /api/meta/categories` — kategori listeleri (meta)  
- `POST /api/predict` — fiyat tahmini  
- `GET /api/eda/buyer_guides` — alıcı rehberi grafik verisi  
- `GET /api/eda/cross_market` — çapraz piyasa paketi  

---

## Proje yapısı (özet)

```
arabam_proje/
├── web_app.py              # Flask uygulaması
├── templates/
│   └── index.html          # Tek sayfa arayüz + Chart.js
├── static/
│   └── splash-bg.png       # Açılış ekranı görseli
├── data/                   # CSV verileri (tercih edilen konum)
├── artifacts_arabam/       # Güncel model çıktıları
├── artifacts/              # Referans model çıktıları
├── arabam_preprocess.py    # Güncel veri ön işleme
├── car_preprocess.py       # Referans veri ön işleme
├── market_compare.py       # Dönem / çapraz karşılaştırma
├── train_model_arabam.py
├── train_model.py
├── requirements.txt
├── render.yaml
└── README.md
```

---

## Notlar ve sınırlar

- Tahminler **bilgilendirme amaçlıdır**; resmi ekspertiz veya bağlayıcı değerleme yerine geçmez.
- Dönem kıyası ve çapraz analizler, her iki veri seti de erişilebilir olduğunda anlamlıdır.
- Jupyter not defterleri (`notebook_*.ipynb`) ayrı analiz ve raporlama için kullanılabilir; web arayüzü bunlardan bağımsız çalışır.

---

## Lisans ve atıf

Akademik bir ödev / ders projesi olarak kullanım için hazırlanmıştır. Veri setlerinin kullanım koşullarına kendi kurumunuzun etik kuralları çerçevesinde uyun.
