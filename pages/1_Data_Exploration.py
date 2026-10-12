import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))  # so `utils` is importable

import pandas as pd
import plotly.express as px
import streamlit as st

from utils import STRETCH, load_raw, require

st.set_page_config(page_title="Data Exploration", page_icon="📊", layout="wide")
st.title("📊 Data Exploration")

raw = require(load_raw(), "`data/raw_data.csv` not found.")

# ----------------------------------------------------------------------
# Sidebar filters
# ----------------------------------------------------------------------
st.sidebar.header("Filters")
contracts = st.sidebar.multiselect("Contract", raw["Contract"].unique(), default=list(raw["Contract"].unique()))
internet = st.sidebar.multiselect("Internet service", raw["InternetService"].unique(),
                                  default=list(raw["InternetService"].unique()))
t_min, t_max = st.sidebar.slider("Tenure (months)", 0, int(raw["tenure"].max()), (0, int(raw["tenure"].max())))

df = raw[raw["Contract"].isin(contracts) & raw["InternetService"].isin(internet)
         & raw["tenure"].between(t_min, t_max)]

if df.empty:
    st.warning("No customers match these filters.")
    st.stop()

k1, k2, k3, k4 = st.columns(4)
k1.metric("Customers", f"{len(df):,}")
k2.metric("Churn rate", f"{(df['Churn'] == 'Yes').mean():.1%}",
          delta=f"{((df['Churn'] == 'Yes').mean() - (raw['Churn'] == 'Yes').mean()) * 100:+.1f} pts vs all",
          delta_color="inverse")
k3.metric("Avg. monthly charge", f"${df['MonthlyCharges'].mean():,.2f}")
k4.metric("Avg. tenure", f"{df['tenure'].mean():.0f} months")

tab_cat, tab_num, tab_corr, tab_data = st.tabs(
    ["🏷️ Categorical features", "📈 Numeric features", "🔗 Correlations", "🗂️ Data"]
)

# ----------------------------------------------------------------------
with tab_cat:
    cat_cols = [c for c in df.columns if c not in ("customerID", "Churn", "tenure", "MonthlyCharges", "TotalCharges")]
    feature = st.selectbox("Feature", cat_cols, index=cat_cols.index("Contract"))
    g = (df.assign(churned=(df["Churn"] == "Yes"))
           .groupby(feature, as_index=False)
           .agg(churn_rate=("churned", "mean"), customers=("churned", "size")))
    a, b = st.columns(2)
    fig = px.bar(g, x=feature, y="churn_rate", color=feature, text=g["churn_rate"].map("{:.0%}".format),
                 title=f"Churn rate by {feature}")
    fig.update_layout(yaxis_tickformat=".0%", yaxis_title="Churn rate", showlegend=False)
    a.plotly_chart(fig, **STRETCH)
    fig = px.bar(g, x=feature, y="customers", color=feature, title=f"Number of customers by {feature}")
    fig.update_layout(showlegend=False)
    b.plotly_chart(fig, **STRETCH)
    st.caption("Check the right chart before trusting a high rate: small groups give noisy percentages.")

# ----------------------------------------------------------------------
with tab_num:
    num = st.selectbox("Numeric feature", ["tenure", "MonthlyCharges", "TotalCharges"])
    a, b = st.columns(2)
    a.plotly_chart(px.histogram(df, x=num, color="Churn", nbins=40, barmode="overlay", opacity=0.7,
                                title=f"{num} distribution"), **STRETCH)
    b.plotly_chart(px.box(df, x="Churn", y=num, color="Churn", title=f"{num} by churn"), **STRETCH)

    st.markdown("**Churn rate by tenure group**")
    bins = pd.cut(df["tenure"], [-1, 6, 12, 24, 48, 72], labels=["0-6", "7-12", "13-24", "25-48", "49-72"])
    tg = df.assign(group=bins, churned=(df["Churn"] == "Yes")).groupby("group", observed=True, as_index=False)["churned"].mean()
    fig = px.line(tg, x="group", y="churned", markers=True)
    fig.update_layout(yaxis_tickformat=".0%", yaxis_title="Churn rate", xaxis_title="Tenure (months)")
    st.plotly_chart(fig, **STRETCH)

# ----------------------------------------------------------------------
with tab_corr:
    enc = pd.get_dummies(df.drop(columns=["customerID"]), drop_first=True).astype(float)
    corr = enc.corr()["Churn_Yes"].drop("Churn_Yes").dropna()
    top = corr.reindex(corr.abs().sort_values(ascending=False).index).head(15).iloc[::-1]
    fig = px.bar(x=top.values, y=top.index, orientation="h", color=top.values,
                 color_continuous_scale="RdBu_r", color_continuous_midpoint=0,
                 labels={"x": "Correlation with churn", "y": ""}, title="Top 15 features correlated with churn")
    fig.update_coloraxes(showscale=False)
    st.plotly_chart(fig, **STRETCH)
    st.caption("Red = more likely to churn, blue = less likely. Correlation is not causation.")

# ----------------------------------------------------------------------
with tab_data:
    st.dataframe(df.head(200), **STRETCH)
    st.download_button("⬇️ Download filtered data (CSV)", df.to_csv(index=False).encode(), "filtered_customers.csv", "text/csv")