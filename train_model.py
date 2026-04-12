"""
Notebook ile aynı pipeline'ı eğitir ve artifacts/ altına kaydeder.
Arayüz (streamlit_app.py) bu dosyayı çalıştırmadan önce bir kez çalıştırılmalıdır.
"""
from __future__ import annotations

import json

import joblib
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from car_preprocess import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    TARGET,
    category_options,
    load_prepared_frame,
)
from paths import BASE_DIR, data_csv

CSV_PATH = data_csv("cars.csv")
ARTIFACTS_DIR = BASE_DIR / "artifacts"
MODEL_PATH = ARTIFACTS_DIR / "best_pipe.joblib"
META_PATH = ARTIFACTS_DIR / "categories.json"
METRICS_PATH = ARTIFACTS_DIR / "metrics.json"


def build_pipeline():
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUMERIC_FEATURES),
            ("cat", categorical_transformer, CATEGORICAL_FEATURES),
        ]
    )
    return Pipeline(
        [
            ("preprocessor", preprocessor),
            (
                "model",
                HistGradientBoostingRegressor(
                    max_iter=500,
                    max_depth=8,
                    learning_rate=0.05,
                    random_state=42,
                ),
            ),
        ]
    )


def main():
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"Veri bulunamadı: {CSV_PATH}")

    print("Loading data...")
    dfm = load_prepared_frame(CSV_PATH)
    X = dfm[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = dfm[TARGET]

    print(f"Rows: {len(dfm)}")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print("Training model (may take a minute)...")
    pipe = build_pipeline()
    pipe.fit(X_train, y_train)
    y_hat = pipe.predict(X_test)
    mae = float(mean_absolute_error(y_test, y_hat))
    rmse = float(root_mean_squared_error(y_test, y_hat))

    print("Refitting on full data for deployment...")
    pipe.fit(X, y)

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, MODEL_PATH)
    meta = category_options(dfm)
    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    metrics = {
        "mae_tl": mae,
        "rmse_tl": rmse,
        "n_validation": int(len(y_test)),
        "note": "mae_tl: ortalama mutlak hata (dogrulama seti); arayuzde +/- bandi icin kullanilir",
    }
    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print(f"Validation MAE: {mae:,.0f} TL | RMSE: {rmse:,.0f} TL")
    print(f"Saved: {MODEL_PATH}")
    print(f"Saved: {META_PATH}")
    print(f"Saved: {METRICS_PATH}")


if __name__ == "__main__":
    main()
