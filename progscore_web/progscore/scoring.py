"""Single-sample programme scoring methods.

Each function takes a `genes x samples` matrix of log-normalized abundance
(a pandas DataFrame with genes as the index) and a list of signature genes,
and returns a per-sample score (numpy array).
"""
import numpy as np
from scipy.stats import rankdata
from scipy import linalg

EPS = 1e-12


def mean_zscore(mat, sig_genes):
    """Mean of gene-wise z-scored expression (the benchmark's recommended method)."""
    g = [x for x in sig_genes if x in mat.index]
    sub = mat.loc[g].astype(float)
    z = (sub.sub(sub.mean(1), axis=0)
           .div(sub.std(1, ddof=0).clip(lower=EPS), axis=0))
    return z.mean(0).values


def ssgsea(mat, sig_genes, alpha=0.25):
    """Single-sample GSEA (Barbie et al., Nature 2009)."""
    g = [x for x in sig_genes if x in mat.index]
    full = mat.values.astype(float)
    G = full.shape[0]
    in_set = np.isin(mat.index, g)
    out_w = 1.0 / (G - len(g))
    scores = np.empty(full.shape[1])
    for j in range(full.shape[1]):
        x = full[:, j]
        order = np.argsort(-x, kind="mergesort")
        ranks = rankdata(x, method="average")
        w = np.where(in_set, ranks ** alpha, 0.0)
        wsum = w.sum() + EPS
        hit = np.where(in_set, w / wsum, 0.0)
        miss = np.where(in_set, 0.0, out_w)
        run = np.cumsum((hit - miss)[order])
        scores[j] = run[np.argmax(np.abs(run))]
    return scores


def gsva(mat, sig_genes, tau=1.0):
    """GSVA (Hänzelmann et al., BMC Bioinformatics 2013)."""
    g = [x for x in sig_genes if x in mat.index]
    full = mat.values.astype(float)
    G = full.shape[0]
    in_set = np.isin(mat.index, g)
    out_w = 1.0 / (G - len(g))
    scores = np.empty(full.shape[1])
    for j in range(full.shape[1]):
        x = full[:, j]
        order = np.argsort(-x, kind="mergesort")
        ranks = rankdata(x, method="average")
        w = np.where(in_set, ranks ** tau, 0.0)
        wsum = w.sum() + EPS
        hit = np.where(in_set, w / wsum, 0.0)
        miss = np.where(in_set, 0.0, out_w)
        run = np.cumsum((hit - miss)[order])
        scores[j] = run[np.argmax(np.abs(run))]
    return scores


def plage(mat, sig_genes):
    """PLAGE (Tomfohr et al., BMC Bioinformatics 2005)."""
    g = [x for x in sig_genes if x in mat.index]
    sub = mat.loc[g].values.astype(float)
    if sub.shape[0] < 2 or sub.shape[1] < 2:
        return np.full(mat.shape[1], np.nan)
    sub = sub - sub.mean(1, keepdims=True)
    U, s, Vt = linalg.svd(sub, full_matrices=False)
    scores = Vt[0]
    if np.corrcoef(scores, sub.mean(0))[0, 1] < 0:
        scores = -scores
    return scores


def singscore(mat, sig_genes):
    """singscore (Foroutan et al., BMC Bioinformatics 2018)."""
    g = [x for x in sig_genes if x in mat.index]
    full = mat.values.astype(float)
    G = full.shape[0]
    sub_idx = np.isin(mat.index, g)
    scores = np.empty(full.shape[1])
    for j in range(full.shape[1]):
        x = full[:, j]
        r = rankdata(x, method="average") - (G + 1) / 2.0
        r = r / (G / 2.0)
        scores[j] = r[sub_idx].mean()
    return scores


PROJECTION_METHODS = {
    "mean_z": mean_zscore,
    "ssgsea": ssgsea,
    "gsva": gsva,
    "plage": plage,
    "singscore": singscore,
}


def score_signature(mat, sig_genes, method="mean_z"):
    """Score a gene signature in a bulk matrix.

    Parameters
    ----------
    mat : pandas.DataFrame, genes x samples (log-normalized abundance)
    sig_genes : list of str, signature genes
    method : {'mean_z', 'ssgsea', 'gsva', 'plage', 'singscore'}

    Returns
    -------
    numpy.ndarray, per-sample score
    """
    if method not in PROJECTION_METHODS:
        raise ValueError(f"method must be one of {list(PROJECTION_METHODS)}")
    return PROJECTION_METHODS[method](mat, sig_genes)
