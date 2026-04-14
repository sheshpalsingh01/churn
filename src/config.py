from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Config:
    # ── Paths ──────────────────────────────────────────────
    raw_data_path   : Path = Path("data/raw/loan_data.csv")
    processed_dir   : Path = Path("data/processed/")
    model_dir       : Path = Path("models/")
    reports_dir     : Path = Path("reports/")

    # Artifact names
    pipeline_name   : str  = "pipeline.joblib"
    model_name      : str  = "model.joblib"
    metrics_name    : str  = "metrics.json"
    feature_imp_plot: str  = "feature_importance.png"
    confusion_plot  : str  = "confusion_matrix.png"

    # ── Data ───────────────────────────────────────────────
    target_col      : str   = "loan_status"
    test_size       : float = 0.2
    val_size        : float = 0.1
    random_seed     : int   = 42

    # Columns by type (update for your dataset)
    numeric_cols    : list = field(default_factory=lambda: [
        "applicant_income", "coapplicant_income",
        "loan_amount", "loan_term", "credit_history"
    ])
    categorical_cols: list = field(default_factory=lambda: [
        "gender", "married", "dependents",
        "education", "self_employed", "property_area"
    ])

    # ── Model ──────────────────────────────────────────────
    model_type      : str  = "random_forest"   # random_forest | xgboost | logistic
    n_estimators    : int  = 200
    max_depth       : int  = 8
    min_samples_leaf: int  = 5
    class_weight    : str  = "balanced"        # handles imbalance

    # ── App ────────────────────────────────────────────────
    app_title       : str  = "Loan Approval Predictor"
    confidence_threshold: float = 0.60

    # ── Helpers ────────────────────────────────────────────
    @property
    def pipeline_path(self) -> Path:
        return self.model_dir / self.pipeline_name

    @property
    def metrics_path(self) -> Path:
        return self.model_dir / self.metrics_name

    @property
    def feature_imp_path(self) -> Path:
        return self.reports_dir / self.feature_imp_plot

    @property
    def confusion_path(self) -> Path:
        return self.reports_dir / self.confusion_plot

    def ensure_dirs(self):
        for d in [self.processed_dir, self.model_dir, self.reports_dir]:
            Path(d).mkdir(parents=True, exist_ok=True)


# Single instance — import cfg everywhere
cfg = Config()
