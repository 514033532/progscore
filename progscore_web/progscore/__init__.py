"""progscore: single-sample programme scoring and deconvolution for bulk omics.

A general-purpose, dependency-light toolkit for scoring cell-state programmes
and deconvolving cell-type fractions from bulk proteomic or transcriptomic data.

Examples
--------
>>> import pandas as pd
>>> from progscore import score_signature, deconvolve
>>> bulk = pd.read_csv("bulk.tsv", sep="\\t", index_col=0)   # genes x samples
>>> score = score_signature(bulk, ["CD34", "IL3RA", "CD99"], method="mean_z")
>>> frac = deconvolve(bulk, ref, ref_var, method="nnls")      # ref: genes x cell types
"""
from .scoring import score_signature, PROJECTION_METHODS
from .cmps import cmps, make_specificity_reference
from .deconvolution import deconvolve, DECONVOLUTION_METHODS

__version__ = "0.1.0"
__all__ = ["score_signature", "deconvolve", "cmps", "make_specificity_reference",
           "PROJECTION_METHODS", "DECONVOLUTION_METHODS"]
