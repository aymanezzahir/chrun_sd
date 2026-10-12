import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))  # so `utils` is importable

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve

from utils import (STRETCH, compute_metrics, feature_columns, feature_effects,
                   final_estimator, load_csv, load_model, predict_scores, require, to_binary)

st.set_page_config(page_title="Model Performance", page_icon="📈", layout="wide")
st.title("📈 Model Performance")

model = require(load_model(), "`model/model.pkl` not found.")
test = require(load_csv("test.csv"), "`data/test.csv` not found.")
columns = require(feature_columns(), "`data/train.csv` not found (needed for the feature list).")

X, y = test[columns], pd.Series(to_binary(test["Churn"]))
pred, score, is_proba = predict_scores(model, X)
m = compute_metrics(y, pred, score)

st.markdown(f"**Model:** `{type(final_estimator(model)).__name__}` · evaluated on "
            f"**{len(test):,}** unseen customers ({int(y.sum())} churners, {y.mean():.0%})")

cols = st.columns(5)
for col, k in zip(cols, ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]):
    col.metric(k if k in ("Accuracy", "ROC-AUC") else f"{k} (churn)", f"{m[k]:.3f}")

# ----------------------------------------------------------------------
tab_main, tab_thr, tab_base, tab_feat = st.tabs(
    ["📊 Curves & confusion matrix", "🎚️ Threshold explorer", "🪤 Why not just accuracy?", "🧬 Features"]
)

with tab_main:
    a, b = st.columns(2)
    cm = confusion_matrix(y, pred, labels=[0, 1])
    fig = px.imshow(cm, text_auto=True, color_continuous_scale="Purples",
                    x=["Predicted: stay", "Predicted: churn"], y=["Actual: stay", "Actual: churn"],
                    title="Confusion matrix")
    fig.update_coloraxes(showscale=False)
    a.plotly_chart(fig, **STRETCH)

    fpr, tpr, _ = roc_curve(y, score)
    fig = go.Figure()
    fig.add_scatter(x=fpr, y=tpr, mode="lines", name="Model")
    fig.add_scatter(x=[0, 1], y=[0, 1], mode="lines", name="Random guess", line=dict(dash="dash"))
    fig.update_layout(title=f"ROC curve (AUC = {m['ROC-AUC']:.3f})",
                      xaxis_title="False positive rate", yaxis_title="True positive rate (recall)")
    b.plotly_chart(fig, **STRETCH)

    st.caption("**Recall** = share of real churners caught. **Precision** = share of churn alerts that were right.")

with tab_thr:
    st.write("A model outputs a score; the **threshold** decides who gets flagged. "
             "Lower it to catch more churners (higher recall) at the cost of more false alarms.")
    lo, hi = float(np.min(score)), float(np.max(score))
    default = 0.5 if is_proba else 0.0
    thr = st.slider("Decision threshold", lo, hi, float(np.clip(default, lo, hi)), step=(hi - lo) / 200)
    p_thr = (score >= thr).astype(int)
    mt = compute_metrics(y, p_thr)

    c = st.columns(4)
    c[0].metric("Precision (churn)", f"{mt['Precision']:.3f}")
    c[1].metric("Recall (churn)", f"{mt['Recall']:.3f}")
    c[2].metric("F1 (churn)", f"{mt['F1']:.3f}")
    c[3].metric("Customers flagged", f"{int(p_thr.sum()):,} of {len(y):,}")

    prec, rec, thresholds = precision_recall_curve(y, score)
    fig = go.Figure()
    fig.add_scatter(x=thresholds, y=prec[:-1], mode="lines", name="Precision")
    fig.add_scatter(x=thresholds, y=rec[:-1], mode="lines", name="Recall")
    fig.add_vline(x=thr, line_dash="dash")
    fig.update_layout(title="Precision and recall vs threshold", xaxis_title="Threshold", yaxis_title="Score")
    st.plotly_chart(fig, **STRETCH)

with tab_base:
    prevalence = y.mean()
    base = {"Accuracy": 1 - prevalence, "Precision": 0.0, "Recall": 0.0, "F1": 0.0}
    table = pd.DataFrame(
        [{"Model": "Lazy baseline: predict 'stay' for everyone", **{k: f"{v:.3f}" for k, v in base.items()}},
         {"Model": "My model", **{k: f"{m[k]:.3f}" for k in base}}]
    )
    st.dataframe(table, hide_index=True, **STRETCH)
    st.markdown(
        f"""
        Only **{prevalence:.0%}** of customers churn, so predicting "nobody leaves" is already
        **{1 - prevalence:.0%} accurate** while catching **0** churners. That is why accuracy alone is
        misleading here, and why this project is judged on **recall, precision and F1 for the churn class**.
        """
    )

with tab_feat:
    eff, kind = feature_effects(model, columns)
    if eff is None:
        st.info("This model type doesn't expose feature importances or coefficients.")
    else:
        top = eff.reindex(eff[kind].abs().sort_values(ascending=False).index).head(15).iloc[::-1]
        st.plotly_chart(px.bar(top, x=kind, y="Feature", orientation="h", title="Top 15 features"), **STRETCH)
        if "Coefficient" in kind:
            st.caption("Coefficients refer to scaled features, so they are comparable across features.")