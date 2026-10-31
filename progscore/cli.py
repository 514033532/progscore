"""Command-line interface for progscore."""
import argparse
import sys
import pandas as pd

from .scoring import score_signature, PROJECTION_METHODS
from .deconvolution import deconvolve, DECONVOLUTION_METHODS


def _read(path):
    return pd.read_csv(path, sep="\t", index_col=0)


def cmd_score(args):
    bulk = _read(args.bulk)
    sig = [g.strip() for g in open(args.signature) if g.strip()]
    s = score_signature(bulk, sig, method=args.method)
    pd.Series(s, index=bulk.columns, name="score").to_csv(args.out, sep="\t", header=True)
    print(f"scored {len(bulk.columns)} samples with method '{args.method}' -> {args.out}")


def cmd_deconvolve(args):
    bulk = _read(args.bulk)
    ref = _read(args.reference)
    ref_var = _read(args.ref_var) if args.ref_var else None
    f = deconvolve(bulk, ref, ref_var, method=args.method)
    pd.Series(f, index=bulk.columns, name="fraction").to_csv(args.out, sep="\t", header=True)
    print(f"deconvolved {len(bulk.columns)} samples with method '{args.method}' -> {args.out}")


def main(argv=None):
    p = argparse.ArgumentParser(prog="progscore",
                                description="Single-sample programme scoring and deconvolution")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("score", help="score a gene signature in a bulk matrix")
    s.add_argument("--bulk", required=True, help="bulk matrix (TSV, genes x samples)")
    s.add_argument("--signature", required=True, help="signature gene list (one per line)")
    s.add_argument("--method", default="mean_z", choices=list(PROJECTION_METHODS))
    s.add_argument("--out", required=True, help="output TSV")
    s.set_defaults(func=cmd_score)

    d = sub.add_parser("deconvolve", help="estimate cell-type fraction from a reference")
    d.add_argument("--bulk", required=True)
    d.add_argument("--reference", required=True, help="reference profiles (TSV, genes x cell types)")
    d.add_argument("--ref-var", default=None, help="per-gene within-type variance (TSV)")
    d.add_argument("--method", default="nnls", choices=list(DECONVOLUTION_METHODS))
    d.add_argument("--out", required=True)
    d.set_defaults(func=cmd_deconvolve)

    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
