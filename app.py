"""Aplikasi web Flask: prediksi risiko karyawan resign (Regresi Logistik)."""
import json
import logging
from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, render_template, request

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"

app = Flask(__name__)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------- definisi form
def _skala(label_rendah, label_tinggi):
    return [
        ("1", f"1 - {label_rendah}"),
        ("2", "2 - Cukup"),
        ("3", "3 - Baik"),
        ("4", f"4 - {label_tinggi}"),
    ]

GROUPS = [
    {
        "title": "Profil karyawan",
        "fields": [
            {"name": "Age", "label": "Usia (tahun)", "kind": "number", "min": 18, "max": 65, "default": 35},
            {"name": "MaritalStatus", "label": "Status pernikahan", "kind": "select", "default": "Married",
             "options": [("Single", "Lajang"), ("Married", "Menikah"), ("Divorced", "Cerai")]},
            {"name": "DistanceFromHome", "label": "Jarak rumah ke kantor (km)", "kind": "number", "min": 1, "max": 50, "default": 8},
            {"name": "TotalWorkingYears", "label": "Total pengalaman kerja (tahun)", "kind": "number", "min": 0, "max": 45, "default": 10},
            {"name": "NumCompaniesWorked", "label": "Jumlah perusahaan sebelumnya", "kind": "number", "min": 0, "max": 12, "default": 2},
        ],
    },
    {
        "title": "Pekerjaan",
        "fields": [
            {"name": "MonthlyIncome", "label": "Gaji bulanan (USD)", "kind": "number", "min": 1000, "max": 20000, "default": 5000},
            {"name": "JobLevel", "label": "Level jabatan", "kind": "select", "default": "2",
             "options": [(str(i), f"Level {i}") for i in range(1, 6)]},
            {"name": "StockOptionLevel", "label": "Level opsi saham", "kind": "select", "default": "0",
             "options": [(str(i), f"Level {i}") for i in range(0, 4)]},
            {"name": "OverTime", "label": "Lembur", "kind": "select", "default": "No",
             "options": [("No", "Tidak"), ("Yes", "Ya")]},
            {"name": "BusinessTravel", "label": "Perjalanan dinas", "kind": "select", "default": "Travel_Rarely",
             "options": [("Non-Travel", "Tidak pernah"), ("Travel_Rarely", "Jarang"), ("Travel_Frequently", "Sering")]},
            {"name": "YearsAtCompany", "label": "Lama bekerja di perusahaan (tahun)", "kind": "number", "min": 0, "max": 40, "default": 5},
            {"name": "YearsSinceLastPromotion", "label": "Tahun sejak promosi terakhir", "kind": "number", "min": 0, "max": 15, "default": 1},
        ],
    },
    {
        "title": "Kepuasan",
        "fields": [
            {"name": "JobSatisfaction", "label": "Kepuasan terhadap pekerjaan", "kind": "select", "default": "3",
             "options": _skala("Rendah", "Sangat tinggi")},
            {"name": "EnvironmentSatisfaction", "label": "Kepuasan terhadap lingkungan kerja", "kind": "select", "default": "3",
             "options": _skala("Rendah", "Sangat tinggi")},
            {"name": "WorkLifeBalance", "label": "Keseimbangan kerja dan hidup", "kind": "select", "default": "3",
             "options": _skala("Buruk", "Sangat baik")},
        ],
    },
]
FIELDS = {f["name"]: f for g in GROUPS for f in g["fields"]}

FEATURE_LABELS = {name: f["label"] for name, f in FIELDS.items()}

# ----------------------------------------------------------------- muat model
model = scaler = meta = None
load_error = None
try:
    model = joblib.load(MODEL_DIR / "logreg_model.joblib")
    scaler = joblib.load(MODEL_DIR / "scaler.joblib")
    meta = json.loads((MODEL_DIR / "metadata.json").read_text())
    import sklearn
    if meta.get("sklearn_version") != sklearn.__version__:
        log.warning("Versi scikit-learn berbeda: model dilatih dengan %s, server memakai %s",
                    meta.get("sklearn_version"), sklearn.__version__)
except FileNotFoundError:
    load_error = ("Model belum tersedia. Jalankan analisis_attrition.ipynb sampai bagian "
                  "'Menyimpan Model' agar folder models/ terisi.")
except Exception as exc:  # file rusak atau versi tidak cocok
    load_error = f"Model gagal dimuat: {exc}"


def default_values():
    return {name: str(f["default"]) for name, f in FIELDS.items()}


def parse_form(form):
    """Validasi input; kembalikan (nilai_bersih, pesan_error)."""
    values, errors = {}, {}
    for name, f in FIELDS.items():
        raw = (form.get(name) or "").strip()
        if f["kind"] == "select":
            allowed = {v for v, _ in f["options"]}
            if raw not in allowed:
                errors[name] = "Pilih salah satu opsi."
            else:
                values[name] = raw
            continue
        try:
            num = float(raw)
        except ValueError:
            errors[name] = "Isi dengan angka."
            continue
        if not (f["min"] <= num <= f["max"]):
            errors[name] = f"Isi antara {f['min']} dan {f['max']}."
        else:
            values[name] = raw
    if not errors:
        if float(values["YearsAtCompany"]) > float(values["TotalWorkingYears"]):
            errors["YearsAtCompany"] = "Tidak boleh lebih besar dari total pengalaman kerja."
        if float(values["YearsSinceLastPromotion"]) > float(values["YearsAtCompany"]):
            errors["YearsSinceLastPromotion"] = "Tidak boleh lebih besar dari lama bekerja di perusahaan."
    return values, errors


def build_features(values):
    row = {}
    for name in meta["numeric_cols"]:
        row[name] = float(values[name])
    for name in meta["categorical_cols"]:
        row[name] = values[name]
    df = pd.DataFrame([row])
    X = pd.get_dummies(df, columns=meta["categorical_cols"], dtype=int)
    X = X.reindex(columns=meta["feature_columns"], fill_value=0)
    X[meta["numeric_cols"]] = scaler.transform(X[meta["numeric_cols"]])
    return X


def readable(feature):
    if feature in FEATURE_LABELS:
        return FEATURE_LABELS[feature]
    var, _, val = feature.partition("_")
    label = FEATURE_LABELS.get(var, var)
    opts = dict(FIELDS.get(var, {}).get("options", []))
    return f"{label}: {opts.get(val, val)}"


def predict(values):
    X = build_features(values)
    prob = float(model.predict_proba(X)[0, 1])
    threshold = float(meta["threshold"])

    # Kontribusi tiap fitur terhadap log-odds, dibandingkan dengan karyawan "rata-rata" data latih
    means = pd.Series(meta.get("feature_means", {})).reindex(meta["feature_columns"]).fillna(0.0)
    contrib = model.coef_[0] * (X.iloc[0].values - means.values)
    numeric = set(meta["numeric_cols"])
    pairs = [
        (f, c) for f, c, x in zip(meta["feature_columns"], contrib, X.iloc[0].values)
        if f in numeric or x == 1  # kategori yang tidak dimiliki karyawan diabaikan
    ]
    pairs.sort(key=lambda p: p[1], reverse=True)
    drivers = [{"name": readable(f), "score": round(float(c), 2)} for f, c in pairs if c > 0.05][:3]

    return {
        "probability": prob,
        "percent": round(prob * 100, 1),
        "threshold_percent": round(threshold * 100),
        "at_risk": prob >= threshold,
        "label": "Berisiko Resign" if prob >= threshold else "Tidak Berisiko",
        "drivers": drivers,
    }


@app.route("/", methods=["GET", "POST"])
def index():
    values, errors, result = default_values(), {}, None
    if request.method == "POST":
        values = {name: (request.form.get(name) or "").strip() for name in FIELDS}
        if load_error is None:
            clean, errors = parse_form(request.form)
            if not errors:
                result = predict(clean)
    return render_template(
        "index.html", groups=GROUPS, values=values, errors=errors, result=result,
        load_error=load_error, metrics=(meta or {}).get("metrics_test"),
    )


@app.route("/health")
def health():
    return {"status": "ok" if load_error is None else "model_missing"}


if __name__ == "__main__":
    app.run(debug=True)
