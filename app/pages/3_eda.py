import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import streamlit as st
import pandas as pd
from src.config import cfg
from src.data_loader import load_raw_data
from app.components.charts import (
    target_distribution,
    numeric_histogram,
    correlation_heatmap,
)

st.set_page_config(page_title="EDA", page_icon="📊", layout="wide")
st.title("📊 Exploratory Data Analysis")


@st.cache_data(show_spinner="Loading data...")
def get_data():
    return load_raw_data()


df = get_data()

# ── Overview ──────────────────────────────────────────────────────────────────
st.subheader("Dataset Overview")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Rows",    df.shape[0])
col2.metric("Columns", df.shape[1])
col3.metric("Missing values", int(df.isnull().sum().sum()))
col4.metric("Approval rate",
            f"{(df[cfg.target_col]=='Y').mean():.1%}" if df[cfg.target_col].dtype == object
            else f"{df[cfg.target_col].mean():.1%}")

with st.expander("Raw data sample"):
    st.dataframe(df.head(20), use_container_width=True)

with st.expander("Data types & null counts"):
    info = pd.DataFrame({
        "dtype"     : df.dtypes,
        "null_count": df.isnull().sum(),
        "null_%"    : (df.isnull().mean() * 100).round(1),
    })
    st.dataframe(info, use_container_width=True)

# ── Target distribution ───────────────────────────────────────────────────────
st.divider()
st.subheader("Target Distribution")
col_a, col_b = st.columns([1, 2])

with col_a:
    target_col_encoded = cfg.target_col
    if df[cfg.target_col].dtype == object:
        temp_df = df.copy()
        temp_df["__target__"] = (df[cfg.target_col] == "Y").astype(int)
        fig = target_distribution(temp_df, "__target__")
    else:
        fig = target_distribution(df, cfg.target_col)
    st.plotly_chart(fig, use_container_width=True)

with col_b:
    dist_table = df[cfg.target_col].value_counts().reset_index()
    dist_table.columns = ["Status", "Count"]
    dist_table["Percentage"] = (dist_table["Count"] / len(df) * 100).round(1)
    st.dataframe(dist_table, use_container_width=True, hide_index=True)

# ── Numeric distributions ─────────────────────────────────────────────────────
st.divider()
st.subheader("Numeric Feature Distributions")

numeric_cols = df.select_dtypes(include="number").columns.tolist()
numeric_cols = [c for c in numeric_cols if c != cfg.target_col]

selected_col = st.selectbox("Select a feature", numeric_cols)

target_for_chart = None
if df[cfg.target_col].dtype == object:
    temp = df.copy()
    temp[cfg.target_col] = (df[cfg.target_col] == "Y").astype(int)
    st.plotly_chart(
        numeric_histogram(temp, selected_col, cfg.target_col),
        use_container_width=True,
    )
else:
    st.plotly_chart(
        numeric_histogram(df, selected_col, cfg.target_col),
        use_container_width=True,
    )

# ── Correlation ───────────────────────────────────────────────────────────────
st.divider()
st.subheader("Correlation Matrix")
if len(numeric_cols) >= 2:
    st.plotly_chart(
        correlation_heatmap(df, numeric_cols),
        use_container_width=True,
    )
else:
    st.info("Not enough numeric columns for correlation matrix.")

# ── Categorical features ──────────────────────────────────────────────────────
st.divider()
st.subheader("Categorical Breakdowns")
cat_cols = df.select_dtypes(include="object").columns.tolist()
cat_cols = [c for c in cat_cols if c != cfg.target_col]

if cat_cols:
    import plotly.express as px
    selected_cat = st.selectbox("Select a categorical feature", cat_cols)

    temp = df.copy()
    if df[cfg.target_col].dtype == object:
        temp[cfg.target_col] = (df[cfg.target_col] == "Y").astype(int)

    grp = temp.groupby(selected_cat)[cfg.target_col].mean().reset_index()
    grp.columns = [selected_cat, "approval_rate"]
    grp["approval_rate"] = grp["approval_rate"].round(3)

    fig = px.bar(
        grp, x=selected_cat, y="approval_rate",
        title=f"Approval rate by {selected_cat}",
        color="approval_rate",
        color_continuous_scale="RdYlGn",
        text=grp["approval_rate"].apply(lambda x: f"{x:.1%}"),
    )
    fig.update_layout(yaxis_tickformat=".0%", height=340)
    st.plotly_chart(fig, use_container_width=True)
