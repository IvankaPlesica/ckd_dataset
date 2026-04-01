"""
CKD missingness clustering.
Hierarchical clustering of CKD patients by missingness profile.
k=3 justified by dendrogram — large gap between distance 14.5 and 17.
"""

import os
import base64
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import squareform
from sklearn.cluster import AgglomerativeClustering

from ckd_data import load_ckd
df = load_ckd()

os.makedirs("ckd_reports", exist_ok=True)

# missingness matrix

missing_cols = [c for c in df.columns if df[c].isnull().any()]
M            = df[missing_cols].isnull()
M_ckd        = M[df["class"] == "ckd"]

# filter to informative columns (>2% missing in CKD patients)
inform_cols = M_ckd.columns[M_ckd.mean() > 0.02].tolist()
M_arr       = M_ckd[inform_cols].values.astype(int)

# dendrogram

linkage_matrix = linkage(M_arr, method="ward", metric="euclidean")

fig, ax = plt.subplots(figsize=(10, 5))
dendrogram(linkage_matrix, ax=ax, no_labels=True,
           color_threshold=14.5)
ax.set_title("Dendrogram — CKD patients by missingness profile")
ax.set_ylabel("Distance")
ax.axhline(y=14.5, color="grey", linestyle="--", linewidth=0.8,
           label="cut at k=3")
ax.legend(fontsize=9)
fig.tight_layout()
fig.savefig("ckd_reports/dendrogram_ckd.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved ckd_reports/dendrogram_ckd.png")

# heatmap

cols = inform_cols
n    = len(cols)
mat  = np.eye(n)

for i, a in enumerate(cols):
    for j, b in enumerate(cols):
        if i >= j:
            continue
        both      = (M_ckd[a] & M_ckd[b]).sum()
        v         = max(both / max(M_ckd[a].sum(), 1),
                        both / max(M_ckd[b].sum(), 1))
        mat[i, j] = mat[j, i] = v

dist           = 1 - mat
np.fill_diagonal(dist, 0)
lm             = linkage(squareform(dist), method="average")
order          = dendrogram(lm, no_plot=True)["leaves"]
mat_sorted     = mat[np.ix_(order, order)]
cols_sorted    = [cols[i] for i in order]

cmap = mcolors.LinearSegmentedColormap.from_list(
    "cm", ["#f0f0f8", "#f7c87a", "#C84B20"])
fig, ax = plt.subplots(figsize=(n * 0.65, n * 0.65))
ax.imshow(mat_sorted, cmap=cmap, vmin=0, vmax=1, aspect="auto")
ax.set_xticks(range(n)); ax.set_yticks(range(n))
ax.set_xticklabels(cols_sorted, rotation=45, ha="right", fontsize=10)
ax.set_yticklabels(cols_sorted, fontsize=10)
fig.tight_layout()
fig.savefig("ckd_reports/missingness_heatmap.png",
            dpi=150, bbox_inches="tight")
plt.close()
print("Saved ckd_reports/missingness_heatmap.png")

# clustering

k      = 3
hc     = AgglomerativeClustering(n_clusters=k, linkage="ward")
labels = hc.fit_predict(M_arr)

M_df            = M_ckd[inform_cols].copy()
M_df["cluster"] = labels

cluster_miss = M_df.groupby("cluster").mean().round(3)
cluster_size = M_df.groupby("cluster").size()

print(f"\nCluster sizes (k={k}):")
for c, n in cluster_size.items():
    print(f"  Cluster {c}: {n} patients")

print("\nMissingness rate by cluster:")
print(cluster_miss.T.to_string())

# helper: image to base64

def img_to_b64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

# save HTML

dendro_b64  = img_to_b64("ckd_reports/dendrogram_ckd.png")
heatmap_b64 = img_to_b64("ckd_reports/missingness_heatmap.png")

# build cluster table
cluster_rows = ""
for col in inform_cols:
    row = f"<tr><td>{col}</td>"
    for c in sorted(cluster_size.index):
        val   = cluster_miss.loc[c, col]
        pct   = round(val * 100, 1)
        color = "#a32d2d" if pct > 60 else \
                "#854f0b" if pct > 30 else "#444"
        row  += f"<td style='color:{color}'>{pct}%</td>"
    cluster_rows += row + "</tr>"

cluster_headers = "".join(
    f"<th>Cluster {c} (n={cluster_size[c]})</th>"
    for c in sorted(cluster_size.index)
)

html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>CKD Missingness Clustering</title>
<style>
  body  {{ font-family: sans-serif; max-width: 1000px; margin: 40px auto;
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
  img   {{ max-width: 100%; margin: 12px 0; }}
</style>
</head>
<body>
<h1>CKD Missingness Clustering</h1>
<p style="font-size:13px;color:#666">
  CKD patients only (n={len(M_ckd)}) &nbsp;|&nbsp;
  Hierarchical clustering (Ward linkage) on missingness profile &nbsp;|&nbsp;
  k=3 justified by dendrogram gap at distance ~14.5
</p>

<div class="note">
  Columns included: those with &gt;2% missingness in CKD patients.
  Each patient is represented as a binary vector of missing/present indicators.
  Ward linkage minimises within-cluster variance.
</div>

<h2>Dendrogram</h2>
<p>The dashed line shows the cut point at distance 14.5 giving k=3 clusters.
The large gap between this cut and the next merge (~17) confirms k=3.</p>
<img src="data:image/png;base64,{dendro_b64}" alt="dendrogram">

<h2>Co-missingness Heatmap</h2>
<p>Each cell shows the fraction of co-missing rows relative to the more
common of the two columns. Columns are ordered by hierarchical clustering
of their co-missingness structure.</p>
<img src="data:image/png;base64,{heatmap_b64}" alt="heatmap">

<h2>Missingness Rate by Cluster</h2>
<p>Red &gt;60%, orange &gt;30%, grey otherwise.</p>
<table>
<tr><th>Column</th>{cluster_headers}</tr>
{cluster_rows}
</table>

</body>
</html>"""

with open("ckd_reports/missingness_clustering.html", "w") as f:
    f.write(html)
print("Saved ckd_reports/missingness_clustering.html")