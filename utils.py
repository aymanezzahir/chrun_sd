"""Shared helpers for the churn app (data loading, model scoring, encoding)."""
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
MODEL_PATH = ROOT / "model" / "model.pkl"


_ver = tuple(int(x) for x in st.__version__.split(".")[:2])
STRETCH = {"width": "stretch"} if _ver >= (1, 50) else {"use_container_width": True}


def require(obj, message):
    """Show an error and stop the page if something is missing."""
    if obj is None:
        st.error(message)
        st.stop()
    return obj


@st.cache_data
def load_csv(name):
    p = DATA_DIR / name
    if not p.exists():
        return None
    df = pd.read_csv(p)
   
    return df.drop(columns=df.columns[df.columns.str.startswith("Unnamed")])


@st.cache_data
def load_raw():
    df = load_csv("raw_data.csv")
    if df is None:
        return None
    df = df.copy()
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0)
    return df


@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        return None
    try:
        import joblib
        return joblib.load(MODEL_PATH)
    except Exception:
        with open(MODEL_PATH, "rb") as f:
            return pickle.load(f)


@st.cache_data
def feature_columns():
    """The exact model input columns (same order as in train.csv)."""
    train = load_csv("train.csv")
    return None if train is None else [c for c in train.columns if c != "Churn"]


# scrore
def final_estimator(model):
    return model.steps[-1][1] if hasattr(model, "steps") else model


def to_binary(labels):
    """Convert labels to 0/1. Accepts 0/1, True/False or text such as 'Yes'/'No'."""
    arr = np.asarray(labels).ravel()
    if arr.dtype.kind in "OUS":
        return np.isin(np.char.lower(arr.astype(str)), ["yes", "1", "true", "churn"]).astype(int)
    return arr.astype(int)


def predict_scores(model, X):
    """Returns (class predictions, continuous scores, scores_are_probabilities)."""
    pred = to_binary(model.predict(X))
    if hasattr(model, "predict_proba"):
        return pred, model.predict_proba(X)[:, 1], True
    if hasattr(model, "decision_function"):
        return pred, np.ravel(model.decision_function(X)), False
    return pred, pred.astype(float), False


def compute_metrics(y, pred, score=None):
    from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                                 recall_score, roc_auc_score)
    return {
        "Accuracy": accuracy_score(y, pred),
        "Precision": precision_score(y, pred, zero_division=0),
        "Recall": recall_score(y, pred, zero_division=0),
        "F1": f1_score(y, pred, zero_division=0),
        "ROC-AUC": roc_auc_score(y, score) if score is not None else None,
    }


def feature_effects(model, columns):
    """Importances (tree models) or coefficients (linear models) as a DataFrame."""
    est = final_estimator(model)
    if hasattr(est, "feature_importances_"):
        vals, kind = np.ravel(est.feature_importances_), "Importance"
    elif hasattr(est, "coef_"):
        vals, kind = np.ravel(est.coef_), "Coefficient (+ raises churn risk)"
    else:
        return None, None
    if len(vals) != len(columns):
        return None, None
    return pd.DataFrame({"Feature": columns, kind: vals}), kind



def encode(inputs, columns):
    """Turn raw inputs, e.g. {'Contract': 'One year', 'tenure': 5, ...}, into one model row."""
    row = {}
    for c in columns:
        if c in inputs:                       #
            row[c] = inputs[c]
        else:                                 
            base, value = c.split("_", 1)
            row[c] = int(inputs[base] == value)
    return pd.DataFrame([row])[columns]