import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))  # so `utils` is importable

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from utils import (STRETCH, encode, feature_columns, load_csv, load_model,
                   predict_scores, require)

st.set_page_config(page_title="Predict Churn", page_icon="🤖", layout="wide")
st.title("🤖 Predict Customer Churn")
st.write("Describe a customer below. The prediction updates instantly.")

model = require(load_model(), "`model/model.pkl` not found.")
columns = require(feature_columns(), "`data/train.csv` not found (needed for the feature list).")

YES_NO = ["No", "Yes"]

# ----------------------------------------------------------------------
# Inputs
# ----------------------------------------------------------------------
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("👤 Customer")
    gender = st.selectbox("Gender", ["Female", "Male"])
    senior = st.selectbox("Senior citizen", YES_NO)
    partner = st.selectbox("Partner", YES_NO)
    dependents = st.selectbox("Dependents", YES_NO)
    tenure = st.slider("Tenure (months)", 0, 72, 12)

with col2:
    st.subheader("📞 Services")
    phone = st.selectbox("Phone service", ["Yes", "No"])
    lines = st.selectbox("Multiple lines", ["No", "Yes"] if phone == "Yes" else ["No phone service"])
    internet = st.selectbox("Internet service", ["Fiber optic", "DSL", "No"])
    addon_opts = ["No internet service"] if internet == "No" else YES_NO
    security = st.selectbox("Online security", addon_opts)
    backup = st.selectbox("Online backup", addon_opts)
    protection = st.selectbox("Device protection", addon_opts)
    tech = st.selectbox("Tech support", addon_opts)
    tv = st.selectbox("Streaming TV", addon_opts)
    movies = st.selectbox("Streaming movies", addon_opts)

with col3:
    st.subheader("💳 Contract & billing")
    contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    paperless = st.selectbox("Paperless billing", ["Yes", "No"])
    payment = st.selectbox(
        "Payment method",
        ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
    )
    monthly = st.number_input("Monthly charges ($)", 18.0, 120.0, 70.0, step=0.5)
    total = round(tenure * monthly, 2)
    st.caption(f"Total charges estimated as tenure × monthly = **${total:,.2f}**")

inputs = {
    "gender": gender, "SeniorCitizen": int(senior == "Yes"), "Partner": partner,
    "Dependents": dependents, "tenure": tenure, "PhoneService": phone, "MultipleLines": lines,
    "InternetService": internet, "OnlineSecurity": security, "OnlineBackup": backup,
    "DeviceProtection": protection, "TechSupport": tech, "StreamingTV": tv,
    "StreamingMovies": movies, "Contract": contract, "PaperlessBilling": paperless,
    "PaymentMethod": payment, "MonthlyCharges": monthly, "TotalCharges": total,
}


# ----------------------------------------------------------------------
# Scoring
# ----------------------------------------------------------------------
def risk_of(customer):
    """Churn risk in [0, 1] plus the model's class prediction."""
    pred, score, is_proba = predict_scores(model, encode(customer, columns))
    s = float(score[0])
    risk = s if is_proba else 1 / (1 + np.exp(-s))  # sigmoid for margin-based models (e.g. SVC)
    return risk, int(pred[0]), is_proba


@st.cache_data(show_spinner=False)
def test_scores(_model):
    test = load_csv("test.csv")
    if test is None:
        return None
    return predict_scores(_model, test[columns])[1]


try:
    risk, pred, is_proba = risk_of(inputs)
except Exception as e:
    st.error(f"Could not run the model on these inputs: {e}")
    st.stop()

label = "Churn probability" if is_proba else "Risk score"

# ----------------------------------------------------------------------
# Result
# ----------------------------------------------------------------------
st.divider()
left, right = st.columns([1, 1])

with left:
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=risk * 100,
        number={"suffix": "%"},
        title={"text": label},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": "#7c3aed"},
            "steps": [
                {"range": [0, 35], "color": "#bbf7d0"},
                {"range": [35, 60], "color": "#fde68a"},
                {"range": [60, 100], "color": "#fecaca"},
            ],
        },
    ))
    fig.update_layout(height=280, margin=dict(l=20, r=20, t=60, b=10))
    st.plotly_chart(fig, **STRETCH)

with right:
    if pred == 1:
        st.error("### ⚠️ Likely to churn")
    else:
        st.success("### ✅ Likely to stay")

    dist = test_scores(model)
    if dist is not None:
        _, s_now, _ = predict_scores(model, encode(inputs, columns))
        pct = (dist < s_now[0]).mean() * 100
        st.metric("Riskier than…", f"{pct:.0f}% of test customers")
    st.caption(
        "The training data was balanced 50/50, so probabilities run higher than real-world "
        "rates. Use the score to **rank** customers rather than as an exact probability."
        + ("" if is_proba else " This model has no native probabilities, so the score is a rescaled margin.")
    )

# ----------------------------------------------------------------------
# What-if: which retention action lowers the risk most?
# ----------------------------------------------------------------------
st.subheader("🛠️ What if we act?")
scenarios = []
if contract == "Month-to-month":
    scenarios += [("Move to a 1-year contract", {"Contract": "One year"}),
                  ("Move to a 2-year contract", {"Contract": "Two year"})]
if payment == "Electronic check":
    scenarios.append(("Switch to automatic card payment", {"PaymentMethod": "Credit card (automatic)"}))
if internet != "No" and tech == "No":
    scenarios.append(("Add tech support", {"TechSupport": "Yes"}))
if internet != "No" and security == "No":
    scenarios.append(("Add online security", {"OnlineSecurity": "Yes"}))

if not scenarios:
    st.info("This customer already has the main retention levers in place.")
else:
    rows = []
    for name, change in scenarios:
        r_new, _, _ = risk_of({**inputs, **change})
        rows.append({"Action": name, f"New {label.lower()}": f"{r_new:.0%}",
                     "Change": f"{(r_new - risk) * 100:+.1f} pts"})
    st.dataframe(pd.DataFrame(rows), hide_index=True, **STRETCH)
    st.caption("These are model-based what-ifs, not proof of cause and effect: "
               "they show how the model's score reacts, which is a good place to start an A/B test.")