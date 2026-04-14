# 🏦 Loan Approval Predictor — End-to-End ML Project (Phase 1)

A production-structured ML project with a Streamlit frontend.
Clean separation of data, features, training, and serving layers.

---

## Project Structure

```
ml_project/
├── src/
│   ├── config.py          # All paths + params in one place
│   ├── data_loader.py     # Load raw data (generates sample if none)
│   ├── preprocess.py      # Clean, encode, train/val/test split
│   ├── features.py        # Feature engineering + ColumnTransformer
│   ├── train.py           # Fit pipeline, save artifacts
│   ├── evaluate.py        # Metrics, confusion matrix, feature importance
│   ├── predict.py         # Load model, single + batch inference
│   └── utils.py           # Shared helpers
│
├── app/
│   ├── main.py            # Streamlit entry point
│   ├── pages/
│   │   ├── 1_predict.py   # Single prediction UI
│   │   ├── 2_batch.py     # CSV upload → batch predictions
│   │   ├── 3_eda.py       # Interactive data exploration
│   │   └── 4_model_info.py# Metrics + feature importance plots
│   └── components/
│       ├── charts.py      # Reusable Plotly charts
│       └── input_form.py  # Reusable input form
│
├── data/
│   ├── raw/               # Original CSVs (git-ignored)
│   └── processed/         # train/val/test splits (git-ignored)
│
├── models/                # Saved pipeline.joblib + metrics.json
├── reports/               # confusion_matrix.png, feature_importance.png
├── tests/
│   ├── test_preprocess.py
│   └── test_predict.py
│
├── .streamlit/config.toml
├── requirements.txt
├── Makefile
└── README.md
```

---

## Quickstart

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Train the model
```bash
make train
# or: python -m src.train
```
This will:
- Generate synthetic loan data if no CSV exists in `data/raw/`
- Clean and split the data
- Train a Random Forest pipeline
- Save `models/pipeline.joblib` + `models/metrics.json`
- Save plots to `reports/`

### 3. Launch the app
```bash
make app
# or: streamlit run app/main.py
```

### 4. Run tests
```bash
make test
```

---

## Swapping in Your Own Data

1. Drop your CSV into `data/raw/` and update `cfg.raw_data_path` in `src/config.py`
2. Update `cfg.numeric_cols`, `cfg.categorical_cols`, and `cfg.target_col`
3. Update `app/components/input_form.py` with your feature fields
4. Run `make clean && make train`

---

## Deploy to Streamlit Cloud

1. Push this repo to GitHub
   - If `models/` is large, use [Git LFS](https://git-lfs.github.com/)
   - Or run `make train` locally and commit the `models/` folder for small models
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repo
4. Set **Main file path** to `app/main.py`
5. Click **Deploy** — done ✅

Every `git push` auto-redeploys.

---

## Upgrade Path (Phase 2+)

| Phase | What changes |
|-------|-------------|
| **Phase 1** (now) | joblib + Streamlit Cloud |
| **Phase 2** | Add MLflow tracking → replace metrics.json with MLflow runs |
| **Phase 3** | Add DVC for data versioning |
| **Phase 4** | Add Prefect pipeline + drift monitoring → auto-retrain |

---

## Model Config

All parameters live in `src/config.py`. No magic strings in code.

| Param | Default | Description |
|-------|---------|-------------|
| `model_type` | `random_forest` | `random_forest`, `gradient_boost`, `logistic` |
| `n_estimators` | 200 | Number of trees |
| `max_depth` | 8 | Max tree depth |
| `test_size` | 0.20 | Held-out test fraction |
| `val_size` | 0.10 | Validation fraction |
| `random_seed` | 42 | Reproducibility seed |
| `class_weight` | `balanced` | Handles class imbalance |

---

Built with: `scikit-learn` · `pandas` · `streamlit` · `plotly` · `matplotlib`
