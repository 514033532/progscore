# progscore

Single-sample programme scoring and reference-based deconvolution for bulk
proteomic or transcriptomic data.

`progscore` implements nine estimators — five single-sample scoring methods
(mean-z, ssGSEA, GSVA, PLAGE, singscore) and four deconvolution methods
(NNLS, MuSiC, CIBERSORT, BayesPrism) — plus **CMPS**, a cross-modal programme
similarity score that shrinks the mean z-score toward each sample's similarity
to a single-cell programme reference.

Dependencies are minimal: `numpy`, `scipy`, `pandas`, `scikit-learn`.

## Install

```bash
pip install -e .
```

## Quick start (Python)

```python
import pandas as pd
from progscore import score_signature, deconvolve, cmps

# bulk: genes x samples (log-normalized abundance)
bulk = pd.read_csv("example/example_bulk.tsv", sep="\t", index_col=0)

# 1. score a gene signature
score = score_signature(bulk, ["g1", "g2", "g3", "g4", "g5"], method="mean_z")

# 2. CMPS (needs a single-cell programme reference: gene -> mean expression)
ref = {"g1": 1.0, "g2": 1.5, "g3": 0.8, "g4": 1.2, "g5": 0.5}
cmps_score = cmps(bulk, ["g1", "g2", "g3", "g4", "g5"], ref)

# 3. deconvolve a cell-type fraction (needs reference profiles + variance)
frac = deconvolve(bulk, ref_profiles, ref_variance, method="nnls")
```

`cmps` reference profiles: use `ref_type='mean'` with a single-cell mean-expression
profile (e.g., AML LSC), or `ref_type='specificity'` with a contrast profile built by
`make_specificity_reference(stem_genes, diff_genes, pan_genes)` (e.g., T-ALL ETP).

## Command line

```bash
progscore score --bulk bulk.tsv --signature sig.txt --method mean_z --out scores.tsv
progscore deconvolve --bulk bulk.tsv --reference ref.tsv --ref-var ref_var.tsv --method nnls --out frac.tsv
```

## Methods

| Family | Method | Notes |
|---|---|---|
| scoring | `mean_z` | recommended baseline; robust |
| scoring | `ssgsea` | Barbie 2009 |
| scoring | `gsva` | Hänzelmann 2013 |
| scoring | `plage` | Tomfohr 2005; fragile to dropout/confounders |
| scoring | `singscore` | Foroutan 2018 |
| scoring | `cmps` | **new**; cross-modal shrinkage, best in cross-modal/confounded settings. `ref_type='mean'` (default) or `'specificity'` (recommended when the programme mean is dominated by pan-markers) |
| deconvolution | `nnls` | CIBERSORT-style least squares |
| deconvolution | `music` | MuSiC, variance-weighted |
| deconvolution | `cibersort` | ν-SVR (fast linear-SVR implementation) |
| deconvolution | `bayesprism` | simplified MAP stand-in (not the R package) |

## Input formats

- **bulk**: tab-separated, genes as rows (first column = gene symbol), samples as columns.
- **signature**: plain text, one gene per line.
- **reference**: tab-separated, genes as rows, cell types as columns (mean expression).
- **ref_var**: matching per-gene within-cell-type variance (for `music` / `bayesprism`).

## Run the example

```bash
python example/run_example.py
```

This generates a realistic AML stemness demo with real gene names, scores it with
CMPS and the mean z-score, prints a **biological interpretation**, and saves two
publication-style figures to `example/figures/`:

- `score_histogram.png` / `.pdf` — distribution of programme scores, with the mean marked;
- `score_ranking.png` / `.pdf` — the most vs least stem-like samples.

## Biological interpretation

A programme score is a per-sample number that says how active a given cell-state
programme is in that sample:

- **Higher score** = the programme is more active (e.g. more stem-like / primitive).
- **Lower score** = the programme is less active (e.g. more differentiated).

For the AML LSC example, a high score means the sample is more leukaemic-stem-cell-like,
which in AML is linked to therapy resistance and poor prognosis. The same logic
applies to any signature you supply (stemness, immune, pathway, …).

## Test

```bash
python -m pytest tests/ -q
```

## Citation

Please cite the accompanying manuscript.
