import pandas as pd
import plotly.express as px
import streamlit as st

from utils import (STRETCH, compute_metrics, feature_columns, feature_effects,
                   final_estimator, load_csv, load_model, load_raw, predict_scores)

# ----------------------------------------------------------------------
# CONFIG – edit these to personalise the page
# ----------------------------------------------------------------------
YOUR_NAME = "Your Name"
TAGLINE = "Data Scientist · Machine Learning · Analytics"
GITHUB_URL = "https://github.com/your-username"
LINKEDIN_URL = "https://www.linkedin.com/in/your-username"
REPO_URL = "https://github.com/your-username/churn-prediction"

# Results from your notebook (test set: 1,409 customers, 374 churners).
# "—" = not recorded. Add the other models' numbers from your results table.
BENCHMARK = pd.DataFrame(
    [
        {"Model": "Linear SVC (best F1)", "Accuracy": "—", "Precision": "—", "Recall": "0.663", "F1": "0.618", "ROC-AUC": "0.834"},
        {"Model": "Random Forest", "Accuracy": "0.764", "Precision": "0.54", "Recall": "0.70", "F1": "0.61", "ROC-AUC": "—"},
        {"Model": "SVC (RBF)", "Accuracy": "0.763", "Precision": "0.55", "Recall": "0.62", "F1": "0.58", "ROC-AUC": "—"},
    ]
)

st.set_page_config(page_title="Customer Churn Prediction", page_icon="📉", layout="wide")

st.markdown(
    """
    <style>
    .hero {
        padding: 2.2rem 2rem; border-radius: 18px; margin-bottom: 1.5rem;
        background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 55%, #db2777 100%);
        color: white;
    }
    .hero h1 {margin: 0; font-size: 2.4rem; color: white;}
    .hero p  {margin: .5rem 0 0 0; font-size: 1.1rem; opacity: .92;}
    .card {
        border: 1px solid rgba(128,128,128,.25); border-radius: 14px;
        padding: 1rem 1.2rem; height: 100%;
    }
    .card h4 {margin-top: 0;}
    .pill {
        display: inline-block; padding: .2rem .7rem; margin: .15rem .15rem .15rem 0;
        border-radius: 999px; font-size: .85rem;
        background: rgba(124,58,237,.12); border: 1px solid rgba(124,58,237,.35);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

raw = load_raw()
train = load_csv("train.csv")
test = load_csv("test.csv")
model = load_model()

# ----------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------
with st.sidebar:
    st.success("Select a page above 👆")
    st.markdown("### About me")
    st.markdown(f"**{YOUR_NAME}**  \n{TAGLINE}")
    st.link_button("GitHub", GITHUB_URL, **STRETCH)
    st.link_button("LinkedIn", LINKEDIN_URL, **STRETCH)
    st.link_button("Project repository", REPO_URL, **STRETCH)

# ----------------------------------------------------------------------
# Hero + KPIs
# ----------------------------------------------------------------------
st.markdown(
    """
    <div class="hero">
        <h1>📉 Customer Churn Prediction</h1>
        <p>An end-to-end machine learning project: from raw telecom data to a
        deployed model that flags customers likely to leave, so retention teams
        can act before it's too late.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

if raw is not None:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Customers analysed", f"{len(raw):,}")
    c2.metric("Churn rate", f"{(raw['Churn'] == 'Yes').mean():.1%}")
    c3.metric("Avg. monthly charge", f"${raw['MonthlyCharges'].mean():,.2f}")
    c4.metric("Avg. tenure", f"{raw['tenure'].mean():.0f} months")
else:
    st.warning("`data/raw_data.csv` not found.")

st.divider()

# ----------------------------------------------------------------------
# Navigation
# ----------------------------------------------------------------------
st.subheader("🧭 Explore the project")
n1, n2, n3 = st.columns(3)
with n1:
    st.markdown('<div class="card"><h4>📊 Data Exploration</h4>'
                "Filter customers and explore churn drivers, distributions and correlations.</div>",
                unsafe_allow_html=True)
    st.page_link("pages/1_Data_Exploration.py", label="Open Data Exploration", icon="📊")
with n2:
    st.markdown('<div class="card"><h4>🤖 Predict Churn</h4>'
                "Enter a customer profile, get a churn risk and test retention what-ifs.</div>",
                unsafe_allow_html=True)
    st.page_link("pages/2_Prediction.py", label="Open Prediction", icon="🤖")
with n3:
    st.markdown('<div class="card"><h4>📈 Model Performance</h4>'
                "Metrics, ROC curve, confusion matrix and an interactive threshold explorer.</div>",
                unsafe_allow_html=True)
    st.page_link("pages/3_Model_Performance.py", label="Open Model Performance", icon="📈")

st.divider()

# ----------------------------------------------------------------------
# Tabs
# ----------------------------------------------------------------------
tab_problem, tab_insights, tab_pipeline, tab_results = st.tabs(
    ["🎯 Problem", "🔍 Key insights", "⚙️ Pipeline", "🏆 Results"]
)

with tab_problem:
    left, right = st.columns([3, 2])
    with left:
        st.markdown(
            """
            #### Business problem
            Acquiring a new customer costs far more than keeping an existing one.
            Telecom companies lose a significant share of customers every year, and
            usually find out only after the customer has left.

            #### Goal
            Build a classifier that predicts **which customers are likely to churn**,
            and identify **why**, so the business can target retention offers
            (discounts, contract upgrades, support outreach) where they matter most.

            #### Dataset
            Telco customer data: demographics, subscribed services, contract and
            billing information, with a binary `Churn` label.
            """
        )
    with right:
        if raw is not None:
            st.markdown("#### Data at a glance")
            st.write(f"**Raw dataset:** {raw.shape[0]:,} rows × {raw.shape[1]} columns")
            if train is not None:
                st.write(f"**Preprocessed training set:** {train.shape[0]:,} rows × {train.shape[1]} columns")
            st.write(
                "Raw data is imbalanced (~27% churn). The training set was oversampled to "
                "50/50; the test set keeps the real distribution so scores are realistic."
            )

with tab_insights:
    if raw is None:
        st.info("Load the data to see the insights.")
    else:
        churned = raw.assign(churned=(raw["Churn"] == "Yes"))
        a, b = st.columns(2)

        by_contract = churned.groupby("Contract", as_index=False)["churned"].mean()
        fig1 = px.bar(by_contract, x="Contract", y="churned", color="Contract",
                      text=by_contract["churned"].map("{:.0%}".format),
                      title="Churn rate by contract type")
        fig1.update_layout(yaxis_tickformat=".0%", yaxis_title="Churn rate", showlegend=False)
        a.plotly_chart(fig1, **STRETCH)

        fig2 = px.histogram(raw, x="tenure", color="Churn", nbins=36, barmode="overlay",
                            opacity=0.7, title="Tenure distribution by churn")
        b.plotly_chart(fig2, **STRETCH)

        c, d = st.columns(2)
        by_net = churned.groupby("InternetService", as_index=False)["churned"].mean()
        fig3 = px.bar(by_net, x="InternetService", y="churned", color="InternetService",
                      text=by_net["churned"].map("{:.0%}".format),
                      title="Churn rate by internet service")
        fig3.update_layout(yaxis_tickformat=".0%", yaxis_title="Churn rate", showlegend=False)
        c.plotly_chart(fig3, **STRETCH)

        fig4 = px.box(raw, x="Churn", y="MonthlyCharges", color="Churn",
                      title="Monthly charges vs churn")
        d.plotly_chart(fig4, **STRETCH)

        st.markdown(
            """
            **Takeaways**
            - Month-to-month contracts churn far more than 1- or 2-year contracts.
            - Customers churn most in their **first months**; loyalty grows with tenure.
            - Fiber-optic and higher monthly charges are associated with higher churn.
            """
        )

with tab_pipeline:
    steps = [
        ("1️⃣ Data cleaning", "Fixed `TotalCharges` type, handled missing values, dropped `customerID`."),
        ("2️⃣ Encoding", "One-hot encoded categorical features; kept numeric features."),
        ("3️⃣ Balancing", "Oversampled the minority class in training to fix the 73/27 imbalance."),
        ("4️⃣ Modelling", "Compared 20 scikit-learn classifiers on the same held-out test set."),
        ("5️⃣ Evaluation", "Judged by churn-class recall, F1 and ROC-AUC, not accuracy alone."),
        ("6️⃣ Deployment", "Saved the winner as `model.pkl` and served it with Streamlit."),
    ]
    cols = st.columns(3)
    for i, (title, desc) in enumerate(steps):
        cols[i % 3].markdown(f'<div class="card"><h4>{title}</h4>{desc}</div>', unsafe_allow_html=True)
        cols[i % 3].write("")

    st.markdown("#### Tech stack")
    stack = ["Python", "Pandas", "NumPy", "Scikit-learn", "Matplotlib", "Plotly", "Streamlit", "Joblib", "Git"]
    st.markdown("".join(f'<span class="pill">{s}</span>' for s in stack), unsafe_allow_html=True)

with tab_results:
    live = None
    if model is not None and test is not None:
        try:
            X, y = test[feature_columns()], test["Churn"].astype(int)
            pred, score, _ = predict_scores(model, X)
            live = compute_metrics(y, pred, score)
            live["cm"] = pd.crosstab(y, pred).reindex(index=[0, 1], columns=[0, 1], fill_value=0).values
        except Exception as e:
            st.warning(f"Could not score model.pkl on test.csv: {e}")

    if live:
        st.markdown(f"#### Final model: `{type(final_estimator(model)).__name__}` — scored live on the test set")
        cols = st.columns(5)
        for col, k in zip(cols, ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]):
            v = live[k]
            col.metric(k if k in ("Accuracy", "ROC-AUC") else f"{k} (churn)", "—" if v is None else f"{v:.3f}")

        left, right = st.columns(2)
        fig_cm = px.imshow(live["cm"], text_auto=True, color_continuous_scale="Purples",
                           x=["Predicted: stay", "Predicted: churn"],
                           y=["Actual: stay", "Actual: churn"], title="Confusion matrix")
        fig_cm.update_coloraxes(showscale=False)
        left.plotly_chart(fig_cm, **STRETCH)

        eff, kind = feature_effects(model, feature_columns())
        if eff is not None:
            eff = eff.reindex(eff[kind].abs().sort_values(ascending=False).index).head(10).iloc[::-1]
            right.plotly_chart(px.bar(eff, x=kind, y="Feature", orientation="h", title="Top 10 features"), **STRETCH)
    else:
        st.markdown("#### Final model results")
        st.caption("Add `model/model.pkl` and `data/test.csv` to score the model live here.")

    st.markdown("#### Model comparison (from my notebook)")
    st.dataframe(BENCHMARK, hide_index=True, **STRETCH)

    with st.expander("📝 Honest take: limitations & next steps"):
        st.markdown(
            """
            - The model catches roughly **2 in 3 churners**, but about **half of its churn alerts
              are false alarms** (precision ≈ 0.55). Fine for cheap actions like an email offer,
              risky for expensive ones.
            - Many models land within a few points of each other, so the features, not the
              algorithm, are the bottleneck.
            - **Next:** tune the decision threshold for a target recall, try cost-sensitive
              learning, add feature engineering (e.g. tenure buckets, services count), and use
              SHAP for per-customer explanations.
            """
        )
    st.info("💡 **Business impact:** rank customers by churn risk and focus the retention budget "
            "on the highest-risk, highest-value segments.")

st.divider()
st.caption(f"Built by {YOUR_NAME} · Streamlit · Open to data science opportunities")