"""
evaluate.py
───────────
Compute all metrics, save JSON, and generate report plots.
Called from train.py after fitting.
"""

import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report,
)
from sklearn.pipeline import Pipeline

from src.config import cfg


# ── Main evaluator ────────────────────────────────────────────────────────────

def evaluate_and_save(
    pipeline  : Pipeline,
    X_val     : pd.DataFrame,
    y_val     : pd.Series,
    X_test    : pd.DataFrame,
    y_test    : pd.Series,
    cv_mean   : float = None,
    cv_std    : float = None,
) -> dict:
    cfg.ensure_dirs()

    # Ensure targets are binary int (0/1) regardless of upstream encoding
    import numpy as np
    def _to_binary(y):
        try:
            return y.astype(int)
        except (ValueError, TypeError):
            return (y == "Y").astype(int)

    y_val  = _to_binary(y_val)
    y_test = _to_binary(y_test)

    # Predictions
    val_preds  = pipeline.predict(X_val)
    val_probas = pipeline.predict_proba(X_val)[:, 1]
    test_preds  = pipeline.predict(X_test)
    test_probas = pipeline.predict_proba(X_test)[:, 1]

    # Also cast predictions to int
    val_preds  = np.array(val_preds).astype(int)
    test_preds = np.array(test_preds).astype(int)

    # Metrics dict
    metrics = {
        "model_type"      : cfg.model_type,
        "n_estimators"    : cfg.n_estimators,
        "max_depth"       : cfg.max_depth,
        "train_samples"   : int(len(y_val)),   # placeholder

        "cv_roc_auc_mean" : round(float(cv_mean), 4) if cv_mean is not None else None,
        "cv_roc_auc_std"  : round(float(cv_std),  4) if cv_std  is not None else None,

        "val_roc_auc"     : round(roc_auc_score(y_val,  val_probas),  4),
        "val_accuracy"    : round(accuracy_score(y_val, val_preds),   4),
        "val_f1"          : round(f1_score(y_val,       val_preds),   4),
        "val_precision"   : round(precision_score(y_val, val_preds),  4),
        "val_recall"      : round(recall_score(y_val,   val_preds),   4),

        "test_roc_auc"    : round(roc_auc_score(y_test,  test_probas), 4),
        "test_accuracy"   : round(accuracy_score(y_test, test_preds),  4),
        "test_f1"         : round(f1_score(y_test,       test_preds),  4),
        "test_precision"  : round(precision_score(y_test, test_preds), 4),
        "test_recall"     : round(recall_score(y_test,   test_preds),  4),

        "classification_report": classification_report(y_test, test_preds, output_dict=True),
    }

    # Save metrics JSON
    with open(cfg.metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[evaluate] Metrics saved → {cfg.metrics_path}")

    # Plots
    _plot_confusion_matrix(y_test, test_preds)
    _plot_feature_importance(pipeline)

    return metrics


# ── Plots ─────────────────────────────────────────────────────────────────────

def _plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["Rejected (0)", "Approved (1)"])
    ax.set_yticklabels(["Rejected (0)", "Approved (1)"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix — Test Set")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black",
                    fontsize=14, fontweight="bold")
    plt.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(cfg.confusion_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"[evaluate] Confusion matrix → {cfg.confusion_path}")


def _plot_feature_importance(pipeline: Pipeline):
    """Works for tree-based models that expose feature_importances_."""
    model = pipeline.named_steps["model"]
    if not hasattr(model, "feature_importances_"):
        print("[evaluate] Model has no feature_importances_ — skipping plot.")
        return

    importances = model.feature_importances_
    preprocessor = pipeline.named_steps["preprocessor"]
    try:
        feature_names = list(preprocessor.get_feature_names_out())
    except Exception:
        feature_names = [f"feature_{i}" for i in range(len(importances))]

    # Top 15 features
    idx = np.argsort(importances)[-15:]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(
        [feature_names[i] for i in idx],
        importances[idx],
        color="#4a90d9",
    )
    ax.set_xlabel("Importance")
    ax.set_title("Top Feature Importances")
    fig.tight_layout()
    fig.savefig(cfg.feature_imp_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"[evaluate] Feature importance → {cfg.feature_imp_path}")


# ── Standalone evaluation (on saved pipeline) ────────────────────────────────

def evaluate_saved_model():
    """Re-evaluate from saved artifacts without retraining."""
    import joblib
    from src.preprocess import load_splits
    from src.features import get_X_y

    pipeline = joblib.load(cfg.pipeline_path)
    _, val_df, test_df = load_splits()
    X_val,  y_val  = get_X_y(val_df)
    X_test, y_test = get_X_y(test_df)
    return evaluate_and_save(pipeline, X_val, y_val, X_test, y_test)


if __name__ == "__main__":
    metrics = evaluate_saved_model()
    print(json.dumps({k: v for k, v in metrics.items()
                      if k != "classification_report"}, indent=2))
