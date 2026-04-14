import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import streamlit as st
import pandas as pd
from src.utils import load_metrics, fmt_pct
from src.config import cfg
from app.components.charts import metrics_bar

st.set_page_config(page_title="Model Info", page_icon="📈", layout="wide")
st.title("📈 Model Performance")

metrics = load_metrics()

if metrics is None:
    st.warning("No metrics found. Run `make train` first.")
    st.stop()

# ── Config used ───────────────────────────────────────────────────────────────
with st.expander("Model configuration"):
    st.json({
        "model_type"  : metrics.get("model_type"),
        "n_estimators": metrics.get("n_estimators"),
        "max_depth"   : metrics.get("max_depth"),
    })

# ── Key metrics ───────────────────────────────────────────────────────────────
st.subheader("Key Metrics")
col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Test Accuracy",  fmt_pct(metrics["test_accuracy"]))
col2.metric("Test ROC-AUC",   f"{metrics['test_roc_auc']:.4f}")
col3.metric("Test F1",        f"{metrics['test_f1']:.4f}")
col4.metric("Test Precision", f"{metrics['test_precision']:.4f}")
col5.metric("Test Recall",    f"{metrics['test_recall']:.4f}")

if metrics.get("cv_roc_auc_mean"):
    st.info(f"5-Fold CV ROC-AUC: **{metrics['cv_roc_auc_mean']:.4f}** ± {metrics['cv_roc_auc_std']:.4f}")

# ── Bar chart ─────────────────────────────────────────────────────────────────
st.plotly_chart(metrics_bar(metrics), use_container_width=True)

# ── Val vs Test comparison ────────────────────────────────────────────────────
st.subheader("Validation vs Test")
compare = pd.DataFrame({
    "Metric"    : ["Accuracy", "ROC-AUC", "F1", "Precision", "Recall"],
    "Validation": [
        metrics["val_accuracy"], metrics["val_roc_auc"],
        metrics["val_f1"],       metrics["val_precision"], metrics["val_recall"],
    ],
    "Test"      : [
        metrics["test_accuracy"], metrics["test_roc_auc"],
        metrics["test_f1"],       metrics["test_precision"], metrics["test_recall"],
    ],
})
compare["Validation"] = compare["Validation"].round(4)
compare["Test"]       = compare["Test"].round(4)
st.dataframe(compare, use_container_width=True, hide_index=True)

# ── Classification report ─────────────────────────────────────────────────────
st.subheader("Classification Report")
report = metrics.get("classification_report", {})
if report:
    rows = []
    for label, vals in report.items():
        if isinstance(vals, dict):
            rows.append({
                "Class"    : label,
                "Precision": round(vals["precision"], 3),
                "Recall"   : round(vals["recall"],    3),
                "F1-score" : round(vals["f1-score"],  3),
                "Support"  : int(vals["support"]),
            })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

# ── Plots ─────────────────────────────────────────────────────────────────────
st.subheader("Report Plots")
col_a, col_b = st.columns(2)

with col_a:
    if cfg.feature_imp_path.exists():
        st.image(str(cfg.feature_imp_path), caption="Feature Importances", use_column_width=True)
    else:
        st.info("Feature importance plot not found.")

with col_b:
    if cfg.confusion_path.exists():
        st.image(str(cfg.confusion_path), caption="Confusion Matrix", use_column_width=True)
    else:
        st.info("Confusion matrix plot not found.")
