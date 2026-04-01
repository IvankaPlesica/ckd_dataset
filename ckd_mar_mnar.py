"""
CKD MAR and MNAR analysis.
MAR test: extracts AUC from SemSynth's fitted missingness pipelines.
MNAR test: predicts own value in missing rows via linear regression.
Stratified by subgroup.
"""

import os
import json
import pandas as pd
from mar_mnar import mar_test, mnar_test
from ckd_subgroups import SUBGROUPS, LABELS
from ckd_data import load_ckd

df = load_ckd()

os.makedirs("ckd_reports", exist_ok=True)

# full dataset

mar  = mar_test(df,  exclude_cols=["class"])
mnar = mnar_test(df, exclude_cols=["class"])

print(f"Full dataset — mean AUC: {round(sum(mar.values())/len(mar), 4)}")
print(f"MNAR strong: {[c for c,v in mnar.items() if v['signal']=='strong']}")

# ── per subgroup ──────────────────────────────────────────────────────────────

subgroup_results = {}
print("\nPer subgroup:")
for key, sub in SUBGROUPS.items():
    sub = sub.reset_index(drop=True)
    # exclude staging columns if present
    exclude = ["class", "ckd_stage", "egfr"]
    mar_s  = mar_test(sub,  exclude_cols=exclude)
    mnar_s = mnar_test(sub, exclude_cols=exclude)
    mean_auc   = round(sum(mar_s.values()) / len(mar_s), 4) if mar_s else None
    strong     = [c for c, v in mnar_s.items() if v["signal"] == "strong"]
    subgroup_results[key] = {
        "n":          len(sub),
        "mean_auc":   mean_auc,
        "mnar_strong": strong,
        "mar":        mar_s,
        "mnar":       mnar_s,
    }
    print(f"  {LABELS[key]:45} n={len(sub):>3}  "
          f"AUC={mean_auc}  MNAR strong={strong}")

# save JSON

with open("ckd_reports/mar_mnar.json", "w") as f:
    json.dump({
        "full_dataset": {"mar": mar, "mnar": mnar},
        "subgroups":    subgroup_results,
    }, f, indent=2)
print("\nSaved ckd_reports/mar_mnar.json")

# save HTML

mar_rows = ""
for col, auc in sorted(mar.items(), key=lambda x: -x[1]):
    color = "#a32d2d" if auc > 0.8 else \
            "#854f0b" if auc > 0.7 else "#444"
    mar_rows += (f"<tr><td>{col}</td>"
                 f"<td style='color:{color}'>{auc}</td></tr>")

mnar_rows = ""
for col, v in sorted(mnar.items(),
                     key=lambda x: -x[1]["relative_diff"]):
    color = "#a32d2d" if v["signal"] == "strong" else "#444"
    mnar_rows += (f"<tr><td>{col}</td>"
                  f"<td>{v['present_mean']}</td>"
                  f"<td>{v['predicted_mean']}</td>"
                  f"<td style='color:{color}'>"
                  f"{v['relative_diff']} ({v['signal']})</td></tr>")

strat_rows = ""
for key, res in subgroup_results.items():
    strong_str = ", ".join(res["mnar_strong"]) if res["mnar_strong"] else "none"
    strat_rows += (f"<tr><td>{LABELS[key]}</td>"
                   f"<td>{res['n']}</td>"
                   f"<td>{res['mean_auc']}</td>"
                   f"<td>{strong_str}</td></tr>")

html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>CKD MAR and MNAR Analysis</title>
<style>
  body  {{ font-family: sans-serif; max-width: 900px; margin: 40px auto;
           padding: 0 24px; color: #222; line-height: 1.5; }}
  h1    {{ font-size: 22px; margin-bottom: 4px; }}
  h2    {{ font-size: 17px; margin-top: 40px; padding-bottom: 6px;
           border-bottom: 2px solid #eee; }}
  p     {{ font-size: 14px; color: #444; }}
  table {{ border-collapse: collapse; width: 100%; margin: 12px 0;
           font-size: 13px; }}
  th    {{ background: #f5f5f5; text-align: left; padding: 6px 10px;
           border: 1px solid #ddd; font-weight: 500; }}
  td    {{ padding: 6px 10px; border: 1px solid #ddd; }}
  tr:nth-child(even) {{ background: #fafafa; }}
  .note {{ background: #f9f9f9; border-left: 3px solid #ccc;
           padding: 10px 14px; font-size: 13px; color: #555; margin: 16px 0; }}
</style>
</head>
<body>
<h1>CKD MAR and MNAR Analysis</h1>
<p style="font-size:13px;color:#666">
  Dataset: UCI 336 &nbsp;|&nbsp;
  Outcome column excluded from all tests &nbsp;|&nbsp;
  Mean MAR AUC (full dataset): {round(sum(mar.values())/len(mar), 4)}
</p>

<h2>MAR Test — Full Dataset</h2>
<div class="note">
  For each column with missing values, a logistic regression predicts
  whether that column is missing using all other columns as features.
  Uses SemSynth's fitted missingness pipelines. In-sample evaluation.
</div>
<table>
<tr><th>Column</th><th>AUC</th></tr>
{mar_rows}
</table>

<h2>MNAR Test — Full Dataset</h2>
<div class="note">
  For each numeric column, linear regression predicts what the value
  would have been in rows where it is missing. Large relative difference
  between predicted mean when missing and observed mean when present
  suggests MNAR. Threshold: relative diff &gt; 0.20.
</div>
<table>
<tr><th>Column</th><th>Present mean</th>
    <th>Predicted when missing</th><th>Relative diff</th></tr>
{mnar_rows}
</table>

<h2>Stratified by Subgroup</h2>
<div class="note">
  MAR and MNAR tests run separately on each subgroup.
  AUC is less stable with fewer observations.
</div>
<table>
<tr><th>Subgroup</th><th>N</th>
    <th>Mean MAR AUC</th><th>MNAR strong columns</th></tr>
{strat_rows}
</table>

</body>
</html>"""

with open("ckd_reports/mar_mnar.html", "w") as f:
    f.write(html)
print("Saved ckd_reports/mar_mnar.html")