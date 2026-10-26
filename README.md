# TA-03: Prediksi Risiko Karyawan Resign (Regresi Logistik)

Aplikasi web Flask yang memprediksi peluang seorang karyawan resign, berdasarkan dataset
*IBM HR Analytics Employee Attrition & Performance* (Kaggle).

## Struktur proyek

```
analisis_attrition.ipynb   Notebook: EDA, preprocessing, SMOTE, model, evaluasi, simpan model
app.py                     Aplikasi Flask
templates/index.html       Halaman formulir dan hasil prediksi
static/style.css           Tampilan
models/                    Diisi oleh notebook (model, scaler, metadata)
data/                      Letakkan dataset CSV di sini
requirements.txt           Dependensi aplikasi (untuk deploy)
requirements-dev.txt       Dependensi tambahan untuk notebook
render.yaml, Procfile      Konfigurasi deploy
```

## Menjalankan di komputer sendiri

1. Unduh dataset dari Kaggle, simpan sebagai `data/WA_Fn-UseC_-HR-Employee-Attrition.csv`.
2. Siapkan lingkungan:
   ```
   python -m venv .venv
   .venv\Scripts\activate          # Windows  (Mac/Linux: source .venv/bin/activate)
   pip install -r requirements-dev.txt
   ```
3. Buka `analisis_attrition.ipynb` (Jupyter/VS Code) lalu **Run All**. Ini membuat isi folder `models/`.
4. Pin versi scikit-learn di `requirements.txt` sesuai yang dicetak di akhir notebook, mis. `scikit-learn==1.5.2`.
5. Jalankan aplikasi:
   ```
   python app.py
   ```
   lalu buka http://127.0.0.1:5000

## Deploy ke Render

1. Buat repositori GitHub, lalu push seluruh folder ini **termasuk isi `models/`** (hasil langkah 3).
2. Di Render: New > Web Service > pilih repositori. Render membaca `render.yaml`
   (build: `pip install -r requirements.txt`, start: `gunicorn app:app`).
3. Setelah status *Live*, buka URL-nya dan uji satu prediksi. Paket gratis bisa "tidur" saat sepi,
   jadi request pertama mungkin lambat.

## Checklist output TA-03

- [ ] Repositori GitHub (notebook, `app.py`, folder `models/`)
- [ ] URL aplikasi yang sudah di-deploy
- [ ] Laporan DOC dan PDF (pakai template dari dosen)
- [ ] Video YouTube 5-7 menit (demo aplikasi + temuan, seolah melapor ke manajer HR)
