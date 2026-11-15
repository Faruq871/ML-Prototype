# ML Workbench — Panduan Menjalankan (Bahasa Indonesia)

Ini adalah aplikasi Python multi-file, bukan Jupyter Notebook. UI menggunakan Streamlit; mesin ML dibagi menjadi modul-modul terpisah di folder `ml_workbench/`.

## Menjalankan di Windows

Buka PowerShell di folder proyek:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python scripts/generate_demo_data.py
streamlit run app.py
```

Kalau aktivasi virtual environment diblokir PowerShell:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

## Menjalankan di macOS/Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python scripts/generate_demo_data.py
streamlit run app.py
```

## Alur penggunaan

1. Upload dataset CSV/XLSX.
2. Periksa jumlah baris, tipe fitur, missing values, duplikasi, dan cardinality.
3. Pilih task: classification, regression, clustering, dimensionality reduction, atau anomaly detection.
4. Untuk supervised learning, pilih target variable.
5. Tentukan kolom yang perlu dikeluarkan, misalnya ID atau fitur yang berpotensi leakage.
6. Lihat rekomendasi algoritma beserta alasan dan caveat.
7. Pilih salah satu dari 40 algoritma scikit-learn dan jalankan eksperimen.
8. Evaluasi hasil, lalu unduh model, report, atau output CSV.
9. Untuk classification/regression, gunakan tab Prediction untuk memprediksi data baru dengan skema fitur yang sama.
10. Perbandingan model opsional menggunakan cross-validation di bagian training data; test set tidak digunakan untuk menentukan ranking kandidat.

## Modul utama

- `data/`: membaca, memvalidasi, dan memprofilkan dataset.
- `recommendations/`: memberi ranking heuristik yang bisa dijelaskan.
- `algorithms/`: katalog 40 algoritma dan factory pembuat estimator.
- `preprocessing/`: imputasi, scaling, dan encoding di dalam pipeline.
- `training/`: split, training, eksperimen unsupervised, dan perbandingan CV.
- `evaluation/`: metric sesuai task.
- `artifacts/`: penyimpanan model `.joblib` dan report `.json`.
- `ui/`: seluruh komponen antarmuka Streamlit.

Baca [README.md](README.md) untuk detail teknis dan [docs/architecture.md](docs/architecture.md) untuk alur arsitektur serta cara menambahkan modul/algoritma.
