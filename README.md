# CKD Missingness & Synthetic Data Analysis

Analysis of missingness patterns and synthetic data fidelity for the
UCI Chronic Kidney Disease dataset (UCI 336, n=400, 25 columns).

---

### Subgroups
- `ckd_subgroups.py` — defines all subgroups. Exposes `SUBGROUPS`, `ALL_SUBGROUPS`, `LABELS`, `ALL_LABELS`, `COLORS`. Run directly to generate subgroup profile report.

## Subgroups (15 total)

| Key | Definition | N | Type |
|---|---|---|---|
| early_ckd | eGFR ≥ 60 | 42 | Clinical staging |
| mid_ckd | eGFR 15–59 | 122 | Clinical staging |
| late_ckd | eGFR < 15 | 66 | Clinical staging |
| triple_comorbidity | htn + dm + cad | 24 | Intersectional |
| notckd | class = notckd | 150 | Control |
| elderly_ckd | age ≥ 65, CKD | 77 | Protected attribute |
| complete_cases | 0 missing columns | 158 | Data quality |
| high_missingness | ≥5 missing columns | 96 | Data quality |
| unstaged | missing sc or age | 20 | Data quality |
| blood_count_missing | wbcc AND rbcc missing | 105 | Pattern-based |
| electrolyte_missing | sod AND pot missing | 87 | Pattern-based |
| urinalysis_missing | sg AND al AND su missing | 44 | Pattern-based |
| cluster_0 | Missingness cluster 0 | 134 | Data-driven |
| cluster_1 | Missingness cluster 1 | 42 | Data-driven |
| cluster_2 | Missingness cluster 2 | 74 | Data-driven |

---

## Key Findings

**Missingness structure**
- Two distinct co-missingness panels in full dataset: (sc, bu, sod, pot — lift 23.5) and
  (sg, al, su — lift 8.5).
- first is driven by notCKD patients. second is
  CKD-specific, concentrated in mid and late stages.
- Running on CKD patients only: 1,407 Apriori rules, first
  persists (lift 20.8) but is CKD-specific.
- PyAerial misses the panel in both cases — too rare for the
  autoencoder to learn.

**Missingness mechanism**
- Full dataset: Mixed MAR/MNAR, mean AUC 0.818.
  MNAR strong: al, su, sc.
- CKD only: mean AUC 0.771, MNAR strong: sc only.
- notCKD: mean AUC 0.835, no MNAR.
- MNAR signals shift across stages: none in early CKD, su in mid,
  al + pot in late, al + su in triple comorbidity.

**Missingness clustering**
- k=3 justified by dendrogram gap at distance 14.5.
- Clusters partially track eGFR severity but don't map cleanly to stages.
- Zero early CKD patients in Cluster 1 — missingness distinguishes
  early from late/mid without domain knowledge.

**First tests: Fidelity per subgroup**
- clg_mi2 and semi_mi5 produce identical results.
- ctgan_fast performs worst across all subgroups.
- metasyn: good on complete cases (JSD 0.025), poor on late CKD (0.335).
- Early CKD and triple comorbidity: most models generate too few
  synthetic patients to measure fidelity (n/a).

**Complete cases bias**
- Standard pipeline (SemSynth) uses complete cases only: 158 of 400 rows. This inverts class balance.
- Dropped patients are disproportionately CKD patients with complex
  missingness who are the most clinically interesting subgroups.

---

## Dataset
UCI Chronic Kidney Disease (dataset 336)
Rubini, L., Soundarapandian, P., & Eswaran, P. (2015).
https://doi.org/10.24432/C5G020
