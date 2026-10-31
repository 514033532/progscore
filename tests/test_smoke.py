"""Smoke tests for progscore."""
import numpy as np
import pandas as pd

from progscore import score_signature, deconvolve, cmps
from progscore.scoring import PROJECTION_METHODS
from progscore.deconvolution import DECONVOLUTION_METHODS


def _bulk():
    rng = np.random.default_rng(0)
    return pd.DataFrame(rng.normal(0, 1, (300, 25)),
                        index=[f"g{i}" for i in range(300)])


def _ref():
    bulk = _bulk()
    return (pd.DataFrame(np.random.default_rng(1).normal(0, 1, (300, 4)),
                         index=bulk.index, columns=["LSC", "Mono", "GMP", "Bg"]),
            pd.DataFrame(np.random.default_rng(2).uniform(0.1, 1, (300, 4)),
                         index=bulk.index, columns=["LSC", "Mono", "GMP", "Bg"]))


def test_all_scoring():
    bulk = _bulk()
    sig = [f"g{i}" for i in range(20)]
    for m in PROJECTION_METHODS:
        assert len(score_signature(bulk, sig, method=m)) == bulk.shape[1]


def test_cmps():
    bulk = _bulk()
    sig = [f"g{i}" for i in range(20)]
    ref = {g: 1.0 for g in sig}
    assert len(cmps(bulk, sig, ref)) == bulk.shape[1]


def test_all_deconvolution():
    bulk = _bulk()
    ref, ref_var = _ref()
    for m in DECONVOLUTION_METHODS:
        assert len(deconvolve(bulk, ref, ref_var, method=m)) == bulk.shape[1]
