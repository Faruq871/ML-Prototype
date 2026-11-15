# ML Workbench — Tabular Machine Learning Platform

A modular Python application that turns the original **60 Machine Learning Algorithms Guidebook** into an interactive tabular experimentation workflow.

## What this repository contains

- **Streamlit UI** for uploading CSV/XLSX, exploring data, selecting tasks and algorithms, running experiments, and downloading artifacts.
- **Dataset ingestion and profiling** for schema, missing values, duplicates, cardinality, feature types, and basic quality checks.
- **Explainable recommendation engine** that ranks candidate models using transparent, dataset-aware heuristics.
- **40 classical scikit-learn algorithms** from the original guidebook, organized in a central catalog and estimator factory.
- **Preprocessing pipelines** for numerical/categorical values and missing-value imputation, fitted only on training data for supervised models.
- **Training and evaluation** for classification, regression, clustering, dimensionality reduction, and anomaly detection.
- **Cross-validation comparison** for supervised candidate models, performed on the training partition only.
- **Artifact export** for trained models (`.joblib`), reports (`.json`), predictions and transformed outputs (`.csv`).
- **Prediction workflow** for new tabular data using a trained classification/regression model.

## Repository structure

```text
ml-workbench/
├── app.py                         # Streamlit entry point
├── ml_workbench/
│   ├── config.py                  # Paths and app configuration
│   ├── schemas.py                 # Shared dataclasses
│   ├── exceptions.py              # User-facing exceptions
│   ├── data/
│   │   ├── loader.py              # CSV/XLSX ingestion and validation
│   │   └── profiler.py            # Dataset profiling
│   ├── recommendations/
│   │   └── engine.py              # Explainable candidate ranking
│   ├── algorithms/
│   │   ├── catalog.py             # Central 40-algorithm catalog
│   │   └── factory.py             # Estimator construction
│   ├── preprocessing/
│   │   └── pipeline.py            # Numeric/categorical pipelines
│   ├── evaluation/
│   │   └── metrics.py             # Supervised/unsupervised metrics
│   ├── training/
│   │   ├── trainer.py             # End-to-end training execution
│   │   └── comparison.py          # Cross-validation comparison
│   ├── artifacts/
│   │   └── manager.py             # Model/report persistence
│   └── ui/
│       └── dashboard.py           # Streamlit interface
├── scripts/
│   └── generate_demo_data.py      # Create sample CSV
├── docs/
│   ├── architecture.md            # Component boundaries and data flow
│   └── 60_algorithms_reference.md # Original 60-entry scope mapping
├── original/
|   └── 60_ML_ori.ipynb            # original notebook
├── LICENSE                        # MIT License
├── tests/                         # Unit and smoke tests
├── data/samples/                  # Generated demo dataset location
├── artifacts/                     # Saved model/report artifacts
├── requirements.txt
└── pyproject.toml
```

A quick Indonesian setup guide is also available in [README_ID.md](README_ID.md). See [docs/architecture.md](docs/architecture.md) for module boundaries and extension points, and [docs/60_algorithms_reference.md](docs/60_algorithms_reference.md) for the mapping of all 60 guidebook entries.

## Requirements

- Python 3.10 or newer
- pip
- Internet access for the initial package installation

## Install and run on Windows (PowerShell)

```powershell
# 1. Create and activate a virtual environment
py -m venv .venv
.\.venv\Scripts\Activate.ps1

# 2. Install dependencies
python -m pip install --upgrade pip
pip install -r requirements.txt

# 3. Create a demo dataset (optional)
python scripts/generate_demo_data.py

# 4. Launch the application
streamlit run app.py
```

If PowerShell blocks script activation, run this once in the current terminal:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

## Install and run on macOS/Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python scripts/generate_demo_data.py
streamlit run app.py
```

## Workflow inside the application

1. Upload a `.csv` or `.xlsx` dataset (maximum 100 MB by default).
2. Review the dataset overview, data types, missing values, and duplicated rows.
3. Choose the learning task: classification, regression, clustering, dimensionality reduction, or anomaly detection.
4. For classification/regression, select the target variable. Exclude ID, leakage, or irrelevant columns from the feature set.
5. Review candidate recommendations and their rationale/caveats.
6. Choose an algorithm from the compatible catalog.
7. Run training or execution. For supervised tasks, the app uses a held-out test split; preprocessing is fitted within a scikit-learn pipeline on training rows only.
8. Inspect metrics and download the model, JSON report, or CSV output.
9. For a trained classification/regression model, upload new rows with the same feature schema in the Prediction tab.
10. Optionally compare multiple supervised algorithms using cross-validation on the training partition. The held-out test set is not used for ranking.

## Algorithm coverage

The app exposes all 40 classical scikit-learn entries from the source guidebook:

- Regression: IDs 1–4, 10, 12, 14, 16, 18, 20, 22
- Classification: IDs 5–9, 11, 13, 15, 17, 19, 21, 23–31
- Clustering: IDs 32–37
- Dimensionality reduction: IDs 38–39
- Anomaly detection: ID 40

The 20 PyTorch/TensorFlow examples in the guidebook include image and sequential architectures (CNN, RNN, LSTM, GRU, Transformer, token embedding, GAN, etc.). They are intentionally **not registered as tabular algorithms** here: a flat CSV/XLSX table does not automatically provide image tensors, token sequences, or ordered sequence windows. They should be added later as separate data modalities with their own input contracts and training pipelines.

## Recommendation engine: what the score means

The current recommender is a transparent rule-based ranking layer. It considers factors such as row count, feature count/type, missingness, target class balance, and algorithm family. It provides candidate priorities and reasons to test, not a guarantee of accuracy and not a learned meta-model. Final model selection should be based on valid validation/CV results and the problem's metric.

## Experimental safeguards

- For supervised models, rows with a missing target are removed before splitting.
- Classification uses stratified splitting when class counts permit; it rejects cases with fewer than two observations in a class because a trustworthy split cannot be formed.
- Preprocessing is part of a Pipeline, so imputation, encoding, and scaling are fitted on training folds rather than on the full supervised dataset.
- Cross-validation comparison uses only the training portion. Use the held-out test set once for the final selected model evaluation.
- Time-series, panel, grouped-subject, and repeated-entity datasets need time-aware or group-aware splitting. The current prototype uses random splitting and must not be used unchanged for those study designs.
- The prototype does not automatically solve leakage, class imbalance, causal inference, or domain validity. Review the data and experimental design.

## Run tests

```bash
pytest -q
```

## Core modules can also be used without the UI

```python
import pandas as pd
from ml_workbench.recommendations import RecommendationEngine
from ml_workbench.training import train_model

frame = pd.read_csv("your_dataset.csv")
recommendations = RecommendationEngine().recommend(
    frame, task="classification", target="target_column"
)
print(recommendations)

result = train_model(frame, algorithm_id=15, target="target_column")
print(result.metrics)
```

## Important note about saved models

A `.joblib` file uses Python pickle-based serialization. Only load model files from sources you trust. Keep the training environment and scikit-learn version recorded for reproducibility; scikit-learn does not guarantee model compatibility across arbitrary versions.

## Roadmap after the prototype

- Add group-aware and time-aware validation strategies.
- Add explicit feature engineering for date/time columns.
- Add robust imbalance strategies and threshold tuning.
- Add hyperparameter search with compute budgets.
- Add experiment history and model registry metadata.
- Add explainability (permutation importance, partial dependence, SHAP as an optional dependency).
- Add separate image/text/time-series data modalities and deep-learning execution.
- Add authentication, background job queue, resource limits, and deployment hardening before multi-user production use.
