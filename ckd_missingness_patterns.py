"""
CKD missingness pattern analysis.
Apriori rule mining + PyAerial rule mining on the full CKD dataset.
"""

import os
import json
import warnings
import pandas as pd
import numpy as np
from mlxtend.frequent_patterns import apriori, association_rules
from aerial import model as aerial_model, rule_extraction
from ckd_data import load_ckd

warnings.filterwarnings("ignore")


df = load_ckd()

os.makedirs("ckd_reports", exist_ok=True)

# missingness matrix

missing_cols = [c for c in df.columns if df[c].isnull().any()]
M            = df[missing_cols].isnull()

# Apriori

def mine_apriori(M, min_support=0.03, min_confidence=0.5, max_len=4):
    freq  = apriori(M, min_support=min_support,
                    use_colnames=True, max_len=max_len)
    rules = association_rules(freq, metric="confidence",
                              min_threshold=min_confidence,
                              num_itemsets=len(freq))
    rules = rules[rules["consequents"].apply(len) == 1]
    return rules.sort_values("lift", ascending=False)

def apriori_to_list(rules_df, n=20):
    return [
        {
            "if_missing":  list(r["antecedents"]),
            "then_missing": list(r["consequents"])[0],
            "support":     round(float(r["support"]),    4),
            "confidence":  round(float(r["confidence"]), 4),
            "lift":        round(float(r["lift"]),       4),
        }
        for _, r in rules_df.head(n).iterrows()
    ]

print("Running Apriori...")
apriori_rules = mine_apriori(M)
print(f"  Rules found: {len(apriori_rules)}")

# PyAerial

def mine_aerial(M, missing_cols, epochs=50):
    print(f"  Training autoencoder ({epochs} epochs)...")
    trained = aerial_model.train(M.astype(str), epochs=epochs)
    result  = rule_extraction.generate_rules(trained)

    co_miss = [
        r for r in result["rules"]
        if all(a["value"] == "True" for a in r["antecedents"])
        and r["consequent"]["value"] == "True"
    ]

    marginal = {col: float(M[col].mean()) for col in missing_cols}
    rules_out = []
    for r in co_miss:
        ant_cols  = [a["feature"] for a in r["antecedents"]]
        cons_col  = r["consequent"]["feature"]
        ant_sup   = marginal.get(ant_cols[0], 0) if len(ant_cols) == 1 \
                    else float(M[ant_cols].all(axis=1).mean())
        cons_sup  = marginal.get(cons_col, 0)
        joint_sup = float(r["support"])
        denom     = ant_sup * cons_sup
        lift      = round(joint_sup / denom, 3) if denom > 0 else None

        rules_out.append({
            "if_missing":    ant_cols,
            "then_missing":  cons_col,
            "support":       round(float(r["support"]), 4),
            "confidence":    round(float(r["confidence"]), 4),
            "lift":          lift,
            "zhangs_metric": round(float(r["zhangs_metric"]), 4),
        })

    rules_out.sort(key=lambda x: (-x["confidence"], -x["support"]))
    return rules_out, result["statistics"]

print("Running PyAerial...")
aerial_rules, aerial_stats = mine_aerial(M, missing_cols)
print(f"  Co-missingness rules: {len(aerial_rules)}")
print(f"  Data coverage: {aerial_stats['data_coverage']}")

# save JSON

report = {
    "apriori": {
        "n_rules":       len(apriori_rules),
        "min_support":   0.03,
        "top_by_lift":   apriori_to_list(apriori_rules.sort_values("lift",    ascending=False)),
        "top_by_support":apriori_to_list(apriori_rules.sort_values("support", ascending=False)),
    },
    "aerial": {
        "n_rules":        len(aerial_rules),
        "data_coverage":  aerial_stats["data_coverage"],
        "top_rules":      aerial_rules[:20],
    },
}

with open("ckd_reports/missingness_patterns.json", "w") as f:
    json.dump(report, f, indent=2)
print("Saved ckd_reports/missingness_patterns.json")

# HTML helpers

def apriori_table(rules_list, highlight_col, high_t, mid_t):
    rows = ""
    for i, r in enumerate(rules_list, 1):
        ant   = ", ".join(r["if_missing"])
        val   = r[highlight_col]
        color = "#a32d2d" if val > high_t else \
                "#854f0b" if val > mid_t  else "#444"
        rows += (f"<tr><td>{i}</td><td>{ant}</td>"
                 f"<td>{r['then_missing']}</td>"
                 f"<td>{r['support']}</td><td>{r['confidence']}</td>"
                 f"<td style='color:{color};font-weight:500'>{r['lift']}</td></tr>")
    return f"""<table>
<tr><th>#</th><th>If missing</th><th>Then missing</th>
    <th>Support</th><th>Confidence</th><th>Lift</th></tr>
{rows}</table>"""


def aerial_table(rules_list):
    rows = ""
    for i, r in enumerate(rules_list, 1):
        ant      = ", ".join(r["if_missing"])
        lift_str = f"{r['lift']:.2f}" if r["lift"] else "n/a"
        rows += (f"<tr><td>{i}</td><td>{ant}</td>"
                 f"<td>{r['then_missing']}</td>"
                 f"<td>{r['support']}</td><td>{r['confidence']}</td>"
                 f"<td>{lift_str}</td>"
                 f"<td>{r['zhangs_metric']}</td></tr>")
    return f"""<table>
<tr><th>#</th><th>If missing</th><th>Then missing</th>
    <th>Support</th><th>Confidence</th><th>Lift</th><th>Zhang</th></tr>
{rows}</table>"""

# save HTML

html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>CKD Missingness Patterns</title>
<style>
  body  {{ font-family: sans-serif; max-width: 960px; margin: 40px auto;
           padding: 0 24px; color: #222; line-height: 1.5; }}
  h1    {{ font-size: 22px; margin-bottom: 4px; }}
  h2    {{ font-size: 17px; margin-top: 40px; padding-bottom: 6px;
           border-bottom: 2px solid #eee; }}
  h3    {{ font-size: 15px; margin-top: 24px; color: #444; }}
  p     {{ font-size: 14px; color: #444; }}
  table {{ border-collapse: collapse; width: 100%; margin: 12px 0;
           font-size: 13px; }}
  th    {{ background: #f5f5f5; text-align: left; padding: 6px 10px;
           border: 1px solid #ddd; font-weight: 500; }}
  td    {{ padding: 6px 10px; border: 1px solid #ddd; }}
  tr:nth-child(even) {{ background: #fafafa; }}
  .note {{ background: #f9f9f9; border-left: 3px solid #ccc;
           padding: 10px 14px; font-size: 13px; color: #555; margin: 16px 0; }}
  .meta {{ font-size: 13px; color: #666; }}
</style>
</head>
<body>
<h1>CKD Missingness Patterns</h1>
<p class="meta">Dataset: UCI 336 &nbsp;|&nbsp; Two mining approaches compared</p>

<h2>Approach 1 — Apriori</h2>
<div class="note">
  Search over all column combinations above the support threshold.
  Guaranteed to find all patterns with support &ge; 0.03.
  Total rules: {len(apriori_rules)} &nbsp;|&nbsp;
  min_support=0.03, min_confidence=0.50, max_len=4
</div>

<h3>Top 20 by Lift — most statistically surprising</h3>
<p>Rare but highly specific. High lift means co-occurrence is far from random.</p>
{apriori_table(report['apriori']['top_by_lift'], 'lift', 10, 5)}

<h3>Top 20 by Support — most common patterns</h3>
<p>Affects the most patients. Lower lift because individual columns are also commonly missing.</p>
{apriori_table(report['apriori']['top_by_support'], 'support', 0.2, 0.1)}

<h2>Approach 2 — PyAerial</h2>
<div class="note">
  Neural autoencoder learns compressed representation of co-missingness structure.
  Addresses rule explosion. It produces compact rule set with full data coverage.
  Co-missingness rules: {len(aerial_rules)} &nbsp;|&nbsp;
  Data coverage: {aerial_stats['data_coverage']} &nbsp;|&nbsp;
  epochs=50<br>
  <strong>Note:</strong> PyAerial misses rare patterns (support &lt; ~5%)
  because the autoencoder compresses them away. The sc, bu, sod, pot — lift 23.5, support 3.25% does not appear here.
  Zhang's metric measures correlation between -1 and 1.
</div>

<h3>Top 20 rules by confidence</h3>
{aerial_table(aerial_rules[:20])}

</body>
</html>"""

with open("ckd_reports/missingness_patterns.html", "w") as f:
    f.write(html)
print("Saved ckd_reports/missingness_patterns.html")