"""
Subgroup profiling and reporting utilities.
"""

import json
import pandas as pd
import numpy as np


def profile(subgroup_df, reference_df, label,
            profile_cols=None, comorbidity_cols=None):
    """
    Profile a subgroup against a reference dataframe.
    
    profile_cols — continuous columns to summarize (mean, std, min, max)
    comorbidity_cols — categorical columns to report positive rates for
    """
    n   = len(subgroup_df)
    pct = round(100 * n / len(reference_df), 1)

    continuous = {}
    for col in (profile_cols or []):
        s = pd.to_numeric(subgroup_df[col], errors="coerce").dropna()
        if len(s) == 0:
            continuous[col] = None
            continue
        continuous[col] = {
            "n":    int(s.count()),
            "mean": round(float(s.mean()), 2),
            "std":  round(float(s.std()), 2),
            "min":  round(float(s.min()), 2),
            "max":  round(float(s.max()), 2),
        }

    comorbidities = {}
    for col in (comorbidity_cols or []):
        if col not in subgroup_df.columns:
            continue
        yes_val = "poor" if col == "appet" else "yes"
        n_yes   = int((subgroup_df[col] == yes_val).sum())
        comorbidities[col] = {
            "n":   n_yes,
            "pct": round(100 * n_yes / n, 1) if n > 0 else 0,
        }

    missing = subgroup_df.isnull().mean().round(3)
    missing = {
        k: float(v) for k, v in missing.items()
        if v > 0
    }

    class_dist = {}
    if "class" in subgroup_df.columns:
        class_dist = {
            str(k): int(v)
            for k, v in subgroup_df["class"].value_counts(dropna=True).items()
        }

    return {
        "label":         label,
        "n":             n,
        "pct_of_total":  pct,
        "class_dist":    class_dist,
        "continuous":    continuous,
        "comorbidities": comorbidities,
        "missingness":   missing,
    }


def compute_overlap(subgroups):
    """Compute pairwise overlap between subgroups."""
    keys  = list(subgroups.keys())
    pairs = []
    for i, a in enumerate(keys):
        for b in keys[i+1:]:
            idx_a = set(subgroups[a].index)
            idx_b = set(subgroups[b].index)
            n_overlap = len(idx_a & idx_b)
            if n_overlap > 0:
                pairs.append({
                    "a": a, "b": b,
                    "n": n_overlap,
                    "pct_of_a": round(100 * n_overlap / len(idx_a), 1),
                    "pct_of_b": round(100 * n_overlap / len(idx_b), 1),
                })
    return pairs
def missingness_by_subgroup(subgroups, min_cooccurrence=3, top_k=5,
                             exclude_cols=None):
    exclude_cols = set(exclude_cols or [])
    results = {}

    for name, sub in subgroups.items():
        missing_cols = [
            c for c in sub.columns
            if sub[c].isnull().any() and c not in exclude_cols
        ]
        M = sub[missing_cols].isnull()

        rates = {
            col: round(float(sub[col].isnull().mean()), 3)
            for col in missing_cols
        }
        rates = dict(sorted(rates.items(), key=lambda x: -x[1]))

        # pairwise
        pairs = []
        for i, a in enumerate(missing_cols):
            for b in missing_cols[i+1:]:
                n = int((M[a] & M[b]).sum())
                if n >= min_cooccurrence:
                    pairs.append({
                        "col_a": a,
                        "col_b": b,
                        "n":     n,
                        "pct":   round(100 * n / len(sub), 1)
                    })
        pairs.sort(key=lambda x: -x["n"])

        # three-way
        triplets = []
        for i, a in enumerate(missing_cols):
            for j, b in enumerate(missing_cols[i+1:], i+1):
                for c in missing_cols[j+1:]:
                    n = int((M[a] & M[b] & M[c]).sum())
                    if n >= min_cooccurrence:
                        triplets.append({
                            "cols": f"{a} + {b} + {c}",
                            "n":    n,
                            "pct":  round(100 * n / len(sub), 1)
                        })
        triplets.sort(key=lambda x: -x["n"])

        results[name] = {
            "n":              len(sub),
            "n_missing_cols": len(missing_cols),
            "missing_rates":  rates,
            "top_pairs":      pairs[:top_k],
            "top_triplets":   triplets[:top_k],
        }

    return results

def build_report(df, subgroups, labels, profile_cols=None,
                 comorbidity_cols=None, meta=None,
                 exclude_from_missingness=None):
    report = {
        "total_rows": len(df),
        "subgroups":  {},
        "overlap":    compute_overlap(subgroups),
        "missingness_by_subgroup": missingness_by_subgroup(
            subgroups,
            exclude_cols=exclude_from_missingness
        ),
    }
    if meta:
        report.update(meta)

    for key, sg_df in subgroups.items():
        label = labels.get(key, key)
        print(f"  Profiling: {label} — n={len(sg_df)}")
        report["subgroups"][key] = profile(
            sg_df, df, label,
            profile_cols=profile_cols,
            comorbidity_cols=comorbidity_cols,
        )

    return report


def save_json(report, path="subgroups.json"):
    with open(path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Saved {path}")


def save_html(report, labels, colors=None, path="subgroups.html"):
    """
    Generate HTML report from a report dict.
    
    colors — dict mapping subgroup key to (bg_hex, fg_hex) tuples
             defaults to grey for all subgroups
    """
    colors = colors or {}

    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Subgroup Analysis</title>
<style>
  body {{ font-family: sans-serif; max-width: 1000px; margin: 40px auto;
          padding: 0 24px; color: #222; line-height: 1.5; }}
  h1   {{ font-size: 22px; margin-bottom: 4px; }}
  h2   {{ font-size: 17px; margin-top: 48px; padding-bottom: 6px;
          border-bottom: 2px solid #eee; }}
  h3   {{ font-size: 14px; color: #555; margin-top: 20px; margin-bottom: 6px; }}
  p    {{ font-size: 14px; color: #444; }}
  table {{ border-collapse: collapse; width: 100%; margin: 8px 0 16px;
           font-size: 13px; }}
  th   {{ background: #f5f5f5; text-align: left; padding: 6px 10px;
          border: 1px solid #ddd; font-weight: 500; }}
  td   {{ padding: 6px 10px; border: 1px solid #ddd; }}
  tr:nth-child(even) {{ background: #fafafa; }}
  .badge {{ display: inline-block; padding: 3px 12px; border-radius: 12px;
            font-size: 13px; font-weight: 600; margin-right: 8px; }}
  .meta  {{ font-size: 13px; color: #666; margin: 6px 0 16px; }}
  .note  {{ background: #f9f9f9; border-left: 3px solid #ccc;
            padding: 10px 14px; font-size: 13px; color: #555; margin: 16px 0; }}
  .grid  {{ display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }}
</style>
</head>
<body>
<h1>Subgroup Analysis</h1>
<p class="meta">Total rows: {report['total_rows']}</p>
"""

    # dataset-level notes
    if report.get("limitation"):
        html += f'<div class="note">{report["limitation"]}</div>'

    for key, data in report["subgroups"].items():
        bg, fg = colors.get(key, ("f0f0f0", "333333"))
        html += f"""
<h2>
  <span class="badge" style="background:#{bg};color:#{fg};">
    {data['label']}
  </span>
</h2>
<p class="meta">
  n = <strong>{data['n']}</strong> &nbsp;|&nbsp;
  {data['pct_of_total']}% of dataset"""

        if data.get("class_dist"):
            html += " &nbsp;|&nbsp; class: "
            html += ", ".join(f"{k}={v}" for k, v in data["class_dist"].items())

        html += "</p>"
        html += '<div class="grid"><div>'

        # continuous
        if data["continuous"]:
            html += """<h3>Continuous variables</h3>
<table>
<tr><th>Variable</th><th>N</th><th>Mean</th><th>Std</th>
    <th>Min</th><th>Max</th></tr>"""
            for col, vals in data["continuous"].items():
                if vals is None:
                    html += (f"<tr><td>{col}</td>"
                             f"<td colspan='5' style='color:#aaa'>no data</td></tr>")
                else:
                    html += (f"<tr><td>{col}</td><td>{vals['n']}</td>"
                             f"<td>{vals['mean']}</td><td>{vals['std']}</td>"
                             f"<td>{vals['min']}</td><td>{vals['max']}</td></tr>")
            html += "</table>"

        html += "</div><div>"

        # comorbidities
        if data["comorbidities"]:
            html += """<h3>Comorbidities</h3>
<table><tr><th>Variable</th><th>N</th><th>%</th></tr>"""
            for col, vals in data["comorbidities"].items():
                html += (f"<tr><td>{col}</td><td>{vals['n']}</td>"
                         f"<td>{vals['pct']}%</td></tr>")
            html += "</table>"

        html += "</div></div>"

    # missingness by subgroup
    if report.get("missingness_by_subgroup"):
        html += "<h2>Missingness by Subgroup</h2>"
        html += ("<div class='note'>Missing rates, top co-occurring pairs "
                 "and triplets per subgroup.</div>")

        for key, data in report["missingness_by_subgroup"].items():
            bg, fg = colors.get(key, ("f0f0f0", "333333"))
            label  = labels.get(key, key)
            html += (f"<h3><span class='badge' "
                     f"style='background:#{bg};color:#{fg};'>{label}</span> "
                     f"<span style='font-weight:normal;font-size:13px;color:#666'>"
                     f"(n={data['n']}, "
                     f"{data['n_missing_cols']} columns with missing)</span></h3>")

            html += "<div style='display:grid;grid-template-columns:1fr 1fr 1fr;gap:16px'>"

            # rates
            html += "<div><h4 style='font-size:13px;color:#555'>Missing rates</h4>"
            html += "<table><tr><th>Column</th><th>%</th></tr>"
            for col, rate in data["missing_rates"].items():
                pct   = round(rate * 100, 1)
                color = "#a32d2d" if pct > 40 else \
                        "#854f0b" if pct > 20 else "#444"
                html += (f"<tr><td>{col}</td>"
                         f"<td style='color:{color}'>{pct}%</td></tr>")
            html += "</table></div>"

            # pairs
            html += "<div><h4 style='font-size:13px;color:#555'>Top pairs</h4>"
            html += "<table><tr><th>Pair</th><th>N</th><th>%</th></tr>"
            for pair in data["top_pairs"]:
                html += (f"<tr><td>{pair['col_a']} + {pair['col_b']}</td>"
                         f"<td>{pair['n']}</td><td>{pair['pct']}%</td></tr>")
            html += "</table></div>"

            # triplets
            html += "<div><h4 style='font-size:13px;color:#555'>Top triplets</h4>"
            html += "<table><tr><th>Triplet</th><th>N</th><th>%</th></tr>"
            for triplet in data["top_triplets"]:
                html += (f"<tr><td>{triplet['cols']}</td>"
                         f"<td>{triplet['n']}</td>"
                         f"<td>{triplet['pct']}%</td></tr>")
            html += "</table></div></div>"

    # overlap
    if report.get("overlap"):
        html += """<h2>Subgroup Overlap</h2>
<table>
<tr><th>Subgroup A</th><th>Subgroup B</th>
    <th>N overlap</th><th>% of A</th><th>% of B</th></tr>"""
        for pair in report["overlap"]:
            label_a = labels.get(pair["a"], pair["a"])
            label_b = labels.get(pair["b"], pair["b"])
            html += (f"<tr><td>{label_a}</td><td>{label_b}</td>"
                     f"<td>{pair['n']}</td>"
                     f"<td>{pair['pct_of_a']}%</td>"
                     f"<td>{pair['pct_of_b']}%</td></tr>")
        html += "</table>"

    html += "</body></html>"

    with open(path, "w") as f:
        f.write(html)
    print(f"Saved {path}")