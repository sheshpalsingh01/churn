"""
app/components/charts.py
─────────────────────────
Reusable Plotly chart functions for all Streamlit pages.
"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


COLORS = {
    "approved"   : "#2ecc71",
    "rejected"   : "#e74c3c",
    "primary"    : "#4a90d9",
    "secondary"  : "#7f8c8d",
    "background" : "#f8f9fa",
}


def probability_gauge(prob_approved: float) -> go.Figure:
    """Circular gauge showing approval probability."""
    fig = go.Figure(go.Indicator(
        mode  = "gauge+number+delta",
        value = round(prob_approved * 100, 1),
        title = {"text": "Approval Probability (%)", "font": {"size": 14}},
        delta = {"reference": 50},
        gauge = {
            "axis"      : {"range": [0, 100], "tickwidth": 1},
            "bar"       : {"color": COLORS["primary"]},
            "bgcolor"   : "white",
            "steps"     : [
                {"range": [0,  40],  "color": "#fde8e8"},
                {"range": [40, 60],  "color": "#fff3cd"},
                {"range": [60, 100], "color": "#d4edda"},
            ],
            "threshold" : {
                "line" : {"color": "black", "width": 3},
                "value": 60,
            },
        },
        number={"suffix": "%"},
    ))
    fig.update_layout(height=280, margin=dict(l=20, r=20, t=40, b=20))
    return fig


def target_distribution(df: pd.DataFrame, target_col: str) -> go.Figure:
    counts = df[target_col].value_counts()
    labels = {0: "Rejected", 1: "Approved"}
    colors = [COLORS["rejected"], COLORS["approved"]]

    fig = go.Figure(go.Pie(
        labels = [labels.get(k, str(k)) for k in counts.index],
        values = counts.values,
        hole   = 0.45,
        marker_colors = colors,
        textinfo = "label+percent",
    ))
    fig.update_layout(
        title="Target Distribution",
        height=320,
        margin=dict(l=20, r=20, t=50, b=20),
        showlegend=True,
    )
    return fig


def numeric_histogram(df: pd.DataFrame, col: str, target_col: str = None) -> go.Figure:
    if target_col and target_col in df.columns:
        fig = px.histogram(
            df, x=col, color=target_col,
            barmode="overlay", opacity=0.7,
            color_discrete_map={0: COLORS["rejected"], 1: COLORS["approved"]},
            labels={target_col: "Status"},
        )
    else:
        fig = px.histogram(df, x=col, nbins=40, color_discrete_sequence=[COLORS["primary"]])

    fig.update_layout(
        title=f"Distribution of {col}",
        height=320,
        xaxis_title=col,
        yaxis_title="Count",
        margin=dict(l=40, r=20, t=50, b=40),
    )
    return fig


def correlation_heatmap(df: pd.DataFrame, numeric_cols: list) -> go.Figure:
    corr = df[numeric_cols].corr()
    fig = go.Figure(go.Heatmap(
        z           = corr.values,
        x           = corr.columns.tolist(),
        y           = corr.index.tolist(),
        colorscale  = "RdBu",
        zmid        = 0,
        text        = np.round(corr.values, 2),
        texttemplate="%{text}",
        hoverongaps = False,
    ))
    fig.update_layout(
        title="Correlation Matrix",
        height=400,
        margin=dict(l=80, r=20, t=50, b=80),
    )
    return fig


def metrics_bar(metrics: dict) -> go.Figure:
    keys = ["test_accuracy", "test_roc_auc", "test_f1", "test_precision", "test_recall"]
    labels = ["Accuracy", "ROC-AUC", "F1", "Precision", "Recall"]
    values = [metrics.get(k, 0) for k in keys]

    fig = go.Figure(go.Bar(
        x             = labels,
        y             = values,
        marker_color  = [COLORS["primary"]] * len(labels),
        text          = [f"{v:.3f}" for v in values],
        textposition  = "outside",
    ))
    fig.update_layout(
        title  = "Model Performance — Test Set",
        yaxis  = {"range": [0, 1.15], "tickformat": ".2f"},
        height = 320,
        margin = dict(l=40, r=20, t=50, b=40),
    )
    return fig
