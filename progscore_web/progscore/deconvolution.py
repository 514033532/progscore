"""Reference-based deconvolution methods.

Each function takes a `genes x samples` bulk matrix, a `genes x cell_types`
reference profile (means), and (for some methods) the matching per-gene
within-cell-type variance, and returns the fraction of a target cell type
per sample.
"""
import numpy as np
from scipy.optimize import nnls, minimize

EPS = 1e-12
MALIGNANT_DEFAULT = ("LSC_primitive", "Monocytic", "GMP_progenitor")


def _lsc_fraction(w, celltypes, target="LSC_primitive", malignant=MALIGNANT_DEFAULT):
    mal = np.sum([w[i] for i, c in enumerate(celltypes) if c in malignant])
    t = w[celltypes.index(target)] if target in celltypes else 0.0
    return t / (mal + EPS)


def deconvolve_nnls(mat, ref, ref_var=None, celltypes=None):
    """Non-negative least squares (CIBERSORT-style)."""
    if celltypes is None:
        celltypes = list(ref.columns)
    common = [g for g in ref.index if g in mat.index]
    R = ref.loc[common].values.astype(float)
    Y = mat.loc[common].values.astype(float)
    fracs = np.empty(Y.shape[1])
    for j in range(Y.shape[1]):
        w, _ = nnls(R, Y[:, j])
        fracs[j] = _lsc_fraction(w, celltypes)
    return fracs


def deconvolve_music(mat, ref, ref_var, celltypes=None, theta=0.5):
    """MuSiC (Wang et al., Nat Commun 2019), variance-weighted NNLS."""
    if celltypes is None:
        celltypes = list(ref.columns)
    common = [g for g in ref.index if g in mat.index]
    R = ref.loc[common].values.astype(float)
    V = ref_var.loc[common].values.astype(float).clip(min=EPS)
    Y = mat.loc[common].values.astype(float)
    disp = V.mean(1, keepdims=True)
    wg = (1.0 / (V + theta * disp)).mean(1)
    sw = np.sqrt(wg)
    Rw = R * sw[:, None]
    fracs = np.empty(Y.shape[1])
    for j in range(Y.shape[1]):
        w, _ = nnls(Rw, Y[:, j] * sw)
        fracs[j] = _lsc_fraction(w, celltypes)
    return fracs


def deconvolve_cibersort(mat, ref, ref_var=None, celltypes=None):
    """CIBERSORT (Newman et al., Nat Methods 2015), nu-SVR."""
    from sklearn.svm import LinearSVR
    if celltypes is None:
        celltypes = list(ref.columns)
    common = [g for g in ref.index if g in mat.index]
    R = ref.loc[common].values.astype(float)
    Y = mat.loc[common].values.astype(float)
    Rz = (R - R.mean(1, keepdims=True)) / (R.std(1, keepdims=True) + EPS)
    Yz = (Y - Y.mean(1, keepdims=True)) / (Y.std(1, keepdims=True) + EPS)
    fracs = np.empty(Y.shape[1])
    for j in range(Y.shape[1]):
        reg = LinearSVR(epsilon=0.1, C=1.0, loss="squared_epsilon_insensitive",
                        dual=False, max_iter=2000)
        reg.fit(Rz, Yz[:, j])
        w = np.clip(reg.coef_.ravel(), 0, None)
        fracs[j] = _lsc_fraction(w, celltypes)
    return fracs


def deconvolve_bayesprism_map(mat, ref, ref_var, celltypes=None, alpha=0.5):
    """BayesPrism-style MAP (Dirichlet prior; simplified stand-in, not the R package)."""
    if celltypes is None:
        celltypes = list(ref.columns)
    common = [g for g in ref.index if g in mat.index]
    R = ref.loc[common].values.astype(float)
    V = ref_var.loc[common].values.astype(float).clip(min=EPS)
    Y = mat.loc[common].values.astype(float)
    sigma2 = V.mean(1) + EPS
    C = R.shape[1]
    fracs = np.empty(Y.shape[1])
    for j in range(Y.shape[1]):
        y = Y[:, j]

        def fun(w):
            r = y - R @ w
            return 0.5 * np.sum(r * r / sigma2) - (alpha - 1.0) * np.sum(np.log(w + 1e-9))

        def jac(w):
            r = y - R @ w
            return -R.T @ (r / sigma2) - (alpha - 1.0) / (w + 1e-9)

        res = minimize(fun, np.full(C, 1.0 / C), method="SLSQP", jac=jac,
                       bounds=[(0, None)] * C,
                       constraints={"type": "eq", "fun": lambda w: w.sum() - 1.0},
                       options={"maxiter": 100, "ftol": 1e-8})
        w = np.clip(res.x, 0, None)
        fracs[j] = _lsc_fraction(w, celltypes)
    return fracs


DECONVOLUTION_METHODS = {
    "nnls": deconvolve_nnls,
    "music": deconvolve_music,
    "cibersort": deconvolve_cibersort,
    "bayesprism": deconvolve_bayesprism_map,
}


def deconvolve(mat, ref, ref_var=None, method="nnls", celltypes=None):
    """Estimate the fraction of a target cell type per bulk sample.

    Parameters
    ----------
    mat : pandas.DataFrame, genes x samples (log-normalized abundance)
    ref : pandas.DataFrame, genes x cell_types (mean expression profiles)
    ref_var : pandas.DataFrame, genes x cell_types (per-gene within-type variance,
              required by 'music' and 'bayesprism')
    method : {'nnls', 'music', 'cibersort', 'bayesprism'}

    Returns
    -------
    numpy.ndarray, per-sample fraction of the target cell type
    """
    if method not in DECONVOLUTION_METHODS:
        raise ValueError(f"method must be one of {list(DECONVOLUTION_METHODS)}")
    fn = DECONVOLUTION_METHODS[method]
    if method in ("music", "bayesprism") and ref_var is None:
        raise ValueError(f"method '{method}' requires ref_var")
    return fn(mat, ref, ref_var, celltypes)
