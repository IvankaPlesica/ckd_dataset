"""
CKD subgroup analysis — dataset-specific definitions.
Uses subgroups.py for dataset profiling and reporting.
"""

import pandas as pd
from subgroups import build_report, save_json, save_html
import os

from ckd_data import load_ckd


df = load_ckd()


# ── eGFR (CKD-EPI 2021, sex-agnostic) ────────────────────────────────────────

def compute_egfr(row):
    sc  = pd.to_numeric(row["sc"],  errors="coerce")
    age = pd.to_numeric(row["age"], errors="coerce")
    if pd.isna(sc) or pd.isna(age) or sc <= 0:
        return float("nan")
    kappa, alpha = 0.8, -0.27
    ratio = sc / kappa
    if ratio < 1:
        gfr = 142 * (ratio ** alpha) * (0.9938 ** age)
    else:
        gfr = 142 * (ratio ** -1.200) * (0.9938 ** age)
    return round(gfr, 1)

df["egfr"] = df.apply(compute_egfr, axis=1)
ckd = df[df["class"] == "ckd"].copy()

ckd["ckd_stage"] = pd.cut(
    ckd["egfr"],
    bins=[0, 15, 60, 999],
    labels=["late", "mid", "early"]
)


# subgroup definitions

# triple comorbidity: htn + dm + cad (cardiometabolic syndrome)
# (1) clinically recognized high cardiovascular risk profile
# (2) rare categorical co-occurrence (n=24, 6% of dataset)
# (3) concentrated in mid/late CKD — 0 early CKD patients
# (4) stress test for synthetic models on rare categorical combinations

SUBGROUPS = {
    "early_ckd":          ckd[ckd["ckd_stage"] == "early"].copy(),
    "mid_ckd":            ckd[ckd["ckd_stage"] == "mid"].copy(),
    "late_ckd":           ckd[ckd["ckd_stage"] == "late"].copy(),
    "triple_comorbidity": df[
        (df["htn"] == "yes") &
        (df["dm"]  == "yes") &
        (df["cad"] == "yes")
    ].copy(),
    "notckd":             df[df["class"] == "notckd"].copy(),
}
LABELS = {
    "early_ckd":          "Early CKD (eGFR ≥ 60, stage 1+2)",
    "mid_ckd":            "Mid CKD (eGFR 15–59, stage 3+4)",
    "late_ckd":           "Late CKD (eGFR < 15, stage 5)",
    "triple_comorbidity": "Triple Comorbidity (htn + dm + cad)",
    "notckd": "notCKD (no chronic kidney disease)",
}

COLORS = {
    "early_ckd":          ("e6f4ea", "1e6e32"),
    "mid_ckd":            ("fff3e0", "854f0b"),
    "late_ckd":           ("fce8e8", "a32d2d"),
    "triple_comorbidity": ("e8f0fe", "1a56a0"),
    "notckd": ("f5f5f5", "444444"),
}

PROFILE_COLS     = ["age", "egfr", "sc", "hemo", "bp", "bu", "bgr", "sod"]
COMORBIDITY_COLS = ["htn", "dm", "cad", "ane", "pe", "appet", "ba"]



# run

if __name__ == "__main__":
    print("Building CKD subgroup report...")

    report = build_report(
        df         = df,
        subgroups  = SUBGROUPS,
        labels     = LABELS,
        profile_cols     = PROFILE_COLS,
        comorbidity_cols = COMORBIDITY_COLS,
        exclude_from_missingness = ["ckd_stage", "egfr"],
        meta = {
            "dataset":        "Chronic Kidney Disease (UCI 336)",
            "ckd_rows":       len(ckd),
            "notckd_rows":    len(df[df["class"] == "notckd"]),
            "staging_method": "eGFR-based (CKD-EPI 2021, sex-agnostic): early=eGFR>=60, mid=15-59, late=<15",
            "limitation":     (
                "Sex column unavailable. eGFR uses sex-agnostic CKD-EPI approximation "
                "(kappa=0.8, alpha=-0.27). Values are approximate. "
                "Unstaged: patients with missing age or sc."
            ),
        }
    )

    os.makedirs("ckd_reports", exist_ok=True)
    save_json(report, "ckd_reports/subgroups.json")
    save_html(report, LABELS, COLORS, "ckd_reports/subgroups.html")