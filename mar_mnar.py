"""
RQ1 — MAR and MNAR missingness mechanism tests.
Builds on SemSynth's fitted missingness pipelines.
"""

import warnings
import numpy as np
import pandas as pd
from semsynth.missingness import fit_missingness_model
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LinearRegression

warnings.filterwarnings("ignore")


def mar_test(df, exclude_cols=None):
    """
    For each column with missing values, predict whether it is missing
    from all other columns using SemSynth's logistic regression pipeline.
    Returns AUC per column.
    """
    df    = df.drop(columns=exclude_cols or [], errors="ignore")
    model = fit_missingness_model(df)

    results = {}
    for col, m in model.models_.items():
        if m.pipeline_ is None:
            continue
        try:
            y = df[col].isnull().astype(int)
            proba = m.pipeline_.predict_proba(df.drop(columns=[col]))[:, 1]
            results[col] = round(float(roc_auc_score(y, proba)), 4)
        except Exception:
            continue
    return results


def mnar_test(df, exclude_cols=None, threshold=0.2):
    """
    For each numeric column, predict what its value would have been
    in missing rows using linear regression on other numeric columns.
    Returns relative difference between predicted and observed mean.
    """
    df       = df.drop(columns=exclude_cols or [], errors="ignore")
    num_cols = df.select_dtypes(include="number").columns.tolist()
    results  = {}

    for col in num_cols:
        if df[col].isnull().sum() < 5:
            continue
        other   = [c for c in num_cols if c != col]
        present = df[col].notna()
        missing = df[col].isna()
        medians = df[other].median()
        X_p     = df.loc[present, other].fillna(medians)
        X_m     = df.loc[missing,  other].fillna(medians)
        y_p     = df.loc[present, col]
        pred    = LinearRegression().fit(X_p, y_p).predict(X_m)
        rel_diff = abs(pred.mean() - y_p.mean()) / (abs(y_p.mean()) + 1e-9)
        results[col] = {
            "present_mean":   round(float(y_p.mean()), 3),
            "predicted_mean": round(float(pred.mean()), 3),
            "relative_diff":  round(rel_diff, 4),
            "signal":         "strong" if rel_diff > threshold else "weak",
        }
    return results