"""Cross-Modal Programme Similarity score (CMPS).

A shrinkage score that combines the mean z-score with the sample's similarity
to a single-cell programme reference profile.

Two reference types are supported:
  - ref_type="mean" (default): the reference is the programme's mean expression
    per gene. This works when the programme's high-expression genes are
    programme-specific (e.g., AML LSC).
  - ref_type="specificity": the reference is a specificity/contrast profile
    (e.g., +1 for programme-specific genes, -1 for alternative-programme
    genes, 0 for shared/pan markers) or a DE log-fold-change. Recommended when
    the programme's mean expression is dominated by pan-markers shared across
    cell states (e.g., T-ALL ETP, where DNTT/CD99/CD47 are pan-T-ALL).
"""
import numpy as np
from .scoring import mean_zscore


def make_specificity_reference(stem_genes, diff_genes=None, pan_genes=None):
    """Build a specificity reference: +1 stem, -1 differentiated, 0 pan.

    Parameters
    ----------
    stem_genes : list of str, programme-specific (stem-like) markers
    diff_genes : list of str, alternative-programme (differentiated) markers
    pan_genes : list of str, shared/pan markers (not programme-specific)

    Returns
    -------
    dict, gene -> specificity weight
    """
    ref = {g: 1.0 for g in stem_genes}
    for g in (diff_genes or []):
        ref[g] = -1.0
    for g in (pan_genes or []):
        ref[g] = 0.0
    return ref


def cmps(mat, sig_genes, ref_profile, lam=0.5, min_k=8, ref_type="mean"):
    """Cross-Modal Programme Similarity score.

    Parameters
    ----------
    mat : pandas.DataFrame, genes x samples (log-normalized abundance)
    sig_genes : list of str, signature genes
    ref_profile : dict or pandas.Series, the single-cell programme reference
                  (gene -> value). For ref_type="mean" the value is the mean
                  expression; for ref_type="specificity" it is a specificity
                  weight or DE log-fold-change.
    lam : float in (0, 1), shrinkage weight (default 0.5)
    min_k : int, minimum signature size for the reference similarity to be
            used; below this the score reduces to the mean z-score.
    ref_type : {'mean', 'specificity'}, the reference type (see module docstring).

    Returns
    -------
    numpy.ndarray, per-sample programme score
    """
    if ref_type not in ("mean", "specificity"):
        raise ValueError("ref_type must be 'mean' or 'specificity'")
    g = [x for x in sig_genes if x in mat.index]
    mz = mean_zscore(mat, g)
    if len(g) < min_k:
        return mz
    sub = mat.loc[g].values.astype(float)
    r = np.array([ref_profile.get(x, 0.0) for x in g])
    cons = np.array([np.corrcoef(sub[:, j], r)[0, 1] for j in range(sub.shape[1])])
    cons = (cons - cons.mean()) / (cons.std() + 1e-12)
    return lam * mz + (1 - lam) * cons
