"""progscore — Streamlit web application.

Single-sample programme scoring and reference-based deconvolution for bulk
proteomic / transcriptomic data, including the cross-modal programme
similarity score (CMPS).

Run locally:
    streamlit run app.py
"""
import sys
import io
from pathlib import Path

import pandas as pd
import numpy as np
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
from progscore import score_signature, deconvolve, cmps, make_specificity_reference  # noqa: E402

st.set_page_config(
    page_title="progscore · Cross-modal Programme Scoring",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---------------------------------------------------------------------------
# Visual identity
# ---------------------------------------------------------------------------
ACCENT = "#4f46e5"
ACCENT_DARK = "#3730a3"
INK = "#1e293b"
MUTED = "#64748b"

st.markdown(
    f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
    html, body, [class*="css"] {{ font-family: 'Inter', -apple-system, sans-serif; color: {INK}; font-size: 19px; }}
    .stMarkdown, .stText, p, li, label, .stDataFrame {{ font-size: 19px; }}
    h1 {{ font-size: 2.6rem; }} h2 {{ font-size: 1.7rem; }} h3 {{ font-size: 1.35rem; }}

    .progscore-hero {{
        background: linear-gradient(120deg, {ACCENT} 0%, #7c3aed 60%, #0ea5e9 100%);
        border-radius: 18px; padding: 2.0rem 2.4rem; margin-bottom: 1.4rem;
        color: #fff; box-shadow: 0 12px 34px -12px rgba(79,70,229,0.5);
    }}
    .progscore-hero h1 {{ color: #fff; font-weight: 800; letter-spacing: -0.02em; margin: 0; font-size: 2.8rem; }}
    .progscore-hero .tagline {{ color: rgba(255,255,255,0.94); font-size: 1.35rem; margin-top: 0.4rem; }}
    .progscore-hero .badges {{ margin-top: 0.8rem; }}
    .progscore-badge {{
        display: inline-block; background: rgba(255,255,255,0.18);
        border: 1px solid rgba(255,255,255,0.35); border-radius: 999px;
        padding: 0.24rem 0.85rem; margin-right: 0.5rem;
        font-size: 1.0rem; font-weight: 600; color: #fff;
    }}
    .stButton > button {{
        background: linear-gradient(120deg, {ACCENT}, {ACCENT_DARK});
        color: #fff; border: none; border-radius: 10px; font-weight: 600;
        padding: 0.6rem 1.8rem; font-size: 1.1rem;
        box-shadow: 0 4px 14px -4px rgba(79,70,229,0.5);
    }}
    .stButton > button:hover {{ filter: brightness(1.07); color: #fff; }}
    .progscore-card {{
        background: #fff; border: 1px solid #e2e8f0; border-radius: 14px;
        padding: 1.2rem 1.4rem; box-shadow: 0 1px 3px rgba(15,23,42,0.06);
        margin-bottom: 1.0rem;
    }}
    /* modular section header in the sidebar */
    .progscore-section {{
        font-weight: 700; color: {INK}; font-size: 1.05rem;
        border-left: 4px solid {ACCENT}; padding-left: 0.6rem; margin: 1.1rem 0 0.4rem 0;
    }}

    /* labels vs input fields — clear visual distinction */
    [data-testid="stTextArea"] label,
    [data-testid="stFileUploader"] label,
    [data-testid="stSelectbox"] label,
    [data-testid="stRadio"] label,
    [data-testid="stCheckbox"] label {{
        font-weight: 700; color: #0f172a;
    }}
    [data-testid="stTextArea"] textarea {{
        background: #f8fafc; border: 1.5px solid #cbd5e1; border-radius: 10px;
    }}
    [data-testid="stFileUploaderDropzone"] {{
        background: #f8fafc; border: 1.5px dashed #94a3b8; border-radius: 10px;
    }}
    [data-testid="stSelectbox"] > div > div {{
        background: #f8fafc; border: 1.5px solid #cbd5e1; border-radius: 10px;
    }}
    .progscore-footer {{
        margin-top: 2.6rem; padding-top: 1.1rem; border-top: 1px solid #e2e8f0;
        color: {MUTED}; font-size: 0.95rem; text-align: center;
    }}
    [data-testid="stSidebar"] {{ background: #f8fafc; }}
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Methods & demo presets
# ---------------------------------------------------------------------------
PROJECTION = {
    "cmps": "CMPS — cross-modal programme similarity (this study)",
    "mean_z": "mean z-score (λ = 1 baseline)",
    "ssgsea": "ssGSEA", "gsva": "GSVA", "plage": "PLAGE", "singscore": "singscore",
}
DECONV = {
    "nnls": "NNLS (CIBERSORT-style least squares)",
    "music": "MuSiC (variance-weighted NNLS)",
    "cibersort": "CIBERSORT (ν-SVR)",
    "bayesprism": "BayesPrism (MAP)",
}

AML_LSC_SIG = ["CD34", "CD99", "KIT", "FLT3", "PROM1", "IL3RA", "HAVCR2",
               "CD47", "CD96", "CLEC12A", "CD44", "IL1RAP", "ADGRG1", "DNTT", "CD123"]
TALL_STEM = ["DNTT", "CD34", "KIT", "CD7", "PROM1", "SPINK2", "LMO2"]
TALL_DIFF = ["CD1A", "CD4", "CD8A", "CD3D", "CD2", "LCK"]
TALL_PAN = ["CD99", "CD47", "CD44"]

DEMOS = {
    "— Select a demo —": None,
    "AML LSC (CMPS, mean reference)": dict(
        sig="\n".join(AML_LSC_SIG), method="cmps", ref_type="mean",
        bulk="example/example_bulk_real.tsv", ref="example/example_ref_aml_lsc.csv",
        label="LSC / stemness programme",
        interp=("A higher score means the leukaemic stem-cell (LSC) programme is more "
                "active — the sample is more primitive/stem-like. In AML a stem-like "
                "state is linked to therapy resistance and poor prognosis."),
        source=("Synthetic demo matrix (60 samples) using real AML gene names "
                "(CD34, CD99, KIT, FLT3, PROM1, …) with a simulated stemness gradient.")),
    "T-ALL ETP (CMPS, specificity reference)": dict(
        sig="\n".join(TALL_STEM + TALL_DIFF + TALL_PAN),
        stem="\n".join(TALL_STEM), diff="\n".join(TALL_DIFF), pan="\n".join(TALL_PAN),
        method="cmps", ref_type="specificity", bulk="example/example_bulk_tall.tsv",
        label="ETP / immature programme",
        interp=("A higher score means the early-T-precursor (ETP)/immature programme is "
                "more active — the sample is more stem-like and less differentiated."),
        source=("Synthetic demo matrix using T-ALL ETP gene names; demonstrates the "
                "specificity reference (+1 stem / −1 differentiation / 0 pan-marker).")),
    "AML LSC (mean-z baseline)": dict(
        sig="\n".join(AML_LSC_SIG), method="mean_z", ref_type=None,
        bulk="example/example_bulk_real.tsv", label="LSC / stemness programme",
        interp=("A higher score means the leukaemic stem-cell (LSC) programme is more "
                "active — the sample is more primitive/stem-like."),
        source=("Same demo matrix scored with the mean z-score (λ = 1 baseline) "
                "instead of CMPS, for comparison.")),
}


def read_matrix(uploaded):
    if uploaded is None:
        return None
    return pd.read_csv(io.BytesIO(uploaded.getvalue()), sep=None, engine="python",
                       index_col=0)


def hero():
    st.markdown(
        """
        <div class="progscore-hero">
            <h1>progscore</h1>
            <div class="tagline">Score cell-state programmes and deconvolve cell-type
            fractions from <b>your own</b> bulk omics data</div>
            <div class="badges">
                <span class="progscore-badge">🧬 CMPS · cross-modal similarity</span>
                <span class="progscore-badge">⚙️ 9 estimators</span>
                <span class="progscore-badge">📤 upload your data</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def footer():
    st.markdown(
        """
        <div class="progscore-footer">
            progscore · open-source (Python library + CLI + web) ·
            CMPS is the subject of a patent application · cite the accompanying manuscript
        </div>
        """,
        unsafe_allow_html=True,
    )


def about_tab():
    st.subheader("What this tool does")
    st.markdown(
        "Give it a **bulk matrix** (genes × samples) and a **signature** (a list of "
        "marker genes), and it returns a **per-sample programme score** — a number that "
        "says how active that programme is in each sample. Reference-based deconvolution "
        "returns a **cell-type fraction** instead."
    )
    st.subheader("How to format your data")
    st.markdown(
        """
        | File | Format |
        |---|---|
        | Bulk matrix | TSV/CSV, genes as **rows**, samples as **columns** (log-normalised abundance) |
        | Signature | plain text, **one gene symbol per line** |
        | Reference (CMPS mean) | TSV/CSV, `gene, mean` |
        | Reference (deconvolution) | TSV/CSV, genes × cell types |
        """
    )
    st.subheader("What data should you upload")
    st.markdown(
        """
        - **Bulk matrix** — any gene-level abundance matrix: bulk **RNA-seq** (e.g. TPM/CPM,
          log-normalised) or bulk **proteomics** (e.g. TMT/DIA log-ratios). Public sources:
          GEO, cBioPortal, the NCI Proteomic Data Commons, DepMap/CCLE.
        - **Signature** — the marker genes of the cell-state/programme you want to score
          (e.g. a stemness, immune, or pathway gene set, from a publication or single-cell data).
        - **Reference** (for CMPS) — a single-cell mean-expression profile of that programme;
          or use the *specificity* reference (+1 stem / −1 diff / 0 pan) if the programme's
          mean is dominated by pan-markers.
        """
    )
    st.subheader("Try a demo first")
    st.markdown(
        "Pick a demo from the sidebar and press **Run analysis** — the signature, "
        "reference and method are pre-filled, so you can see the output format before "
        "using your own data."
    )


def plot_histogram(r):
    """A clean histogram with the mean marked."""
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    n, bins, patches = ax.hist(r.values, bins=24, color="#4f46e5", alpha=0.82,
                               edgecolor="white", linewidth=0.8)
    ax.axvline(r.mean(), color="#d62728", lw=2, ls="--",
               label=f"mean = {r.mean():.3f}")
    ax.set_xlabel("Score", fontsize=11)
    ax.set_ylabel("Number of samples", fontsize=11)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, fontsize=10)
    fig.tight_layout()
    return fig


def show_results(result, label, demo):
    """Render an intuitive result view: summary, histogram, ranked table,
    biological interpretation."""
    r = result.dropna().sort_values(ascending=False)
    st.subheader("Summary")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Samples scored", f"{len(r):,}")
    c2.metric("Mean", f"{r.mean():.3f}")
    c3.metric("Std", f"{r.std():.3f}")
    c4.metric("Range", f"{r.min():.3f} … {r.max():.3f}")

    # Biological interpretation
    if demo:
        interp = demo.get("interp", "")
    else:
        interp = ("A higher score means the signature programme is more active "
                  "in that sample.")
    st.info(f"🧬 **Biological meaning** — {interp}")

    col1, col2 = st.columns([1, 1.1])
    with col1:
        st.markdown('<div class="progscore-card">', unsafe_allow_html=True)
        st.subheader("Distribution")
        st.pyplot(plot_histogram(r), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="progscore-card">', unsafe_allow_html=True)
        st.subheader("Top & bottom samples")
        top = r.head(5)
        bot = r.tail(5).sort_values(ascending=True)
        tab = pd.DataFrame({
            "Top 5 (most active)": [f"{i}  ·  {v:.3f}" for i, v in top.items()],
            "Bottom 5 (least active)": [f"{i}  ·  {v:.3f}" for i, v in bot.items()],
        })
        st.dataframe(tab, use_container_width=True, height=220)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="progscore-card">', unsafe_allow_html=True)
    st.subheader("Full results")
    st.dataframe(r.round(4), height=300, use_container_width=True)
    st.download_button("Download scores (CSV)", r.to_csv().encode(),
                       file_name="progscore_scores.csv", mime="text/csv")
    st.markdown('</div>', unsafe_allow_html=True)


def main():
    hero()

    # ---- Demo selector (full width) ----
    demo_choice = st.selectbox("🎓 Try a demo (one-click)", list(DEMOS.keys()))
    demo = DEMOS[demo_choice]
    if demo is not None:
        st.caption(f"📌 {demo.get('source', '')}")
        with st.expander("⬇️ Download demo files"):
            st.download_button("Demo matrix (TSV)", open(demo["bulk"], "rb").read(),
                               file_name="demo_bulk.tsv", mime="text/tab-separated-values")
            st.download_button("Signature (TXT)", demo.get("sig", "").encode(),
                               file_name="demo_signature.txt", mime="text/plain")
            if demo.get("ref"):
                st.download_button("Reference (CSV)", open(demo["ref"], "rb").read(),
                                   file_name="demo_reference.csv", mime="text/csv")

    # ---- conditional variables (defined across columns) ----
    cmps_ref_f = None
    ref_f = None
    var_f = None
    stem_text = diff_text = pan_text = ""
    bulk = None

    # ---- 3-column wizard ----
    c1, c2, c3 = st.columns([1.15, 1.0, 0.85], gap="large")

    with c1, st.container(border=True):
        st.markdown('<div class="progscore-section">1 · Input data</div>', unsafe_allow_html=True)
        if demo is not None:
            bulk = pd.read_csv(demo["bulk"], sep="\t", index_col=0)
            st.success(f"Demo matrix loaded: {bulk.shape[0]} genes × {bulk.shape[1]} samples")
        else:
            bulk_f = st.file_uploader(
                "Upload your bulk matrix (TSV/CSV, genes × samples)",
                type=["tsv", "csv", "txt"],
                help="A gene-level abundance matrix: bulk RNA-seq (TPM/CPM) or "
                     "proteomics (TMT/DIA log-ratios). Genes as rows, samples as columns.",
            )
            st.caption("💡 Gene × sample abundance matrix (bulk RNA-seq or proteomics, "
                       "log-normalised). Genes = rows, samples = columns. Sources: "
                       "GEO, cBioPortal, Proteomic Data Commons, DepMap/CCLE.")
            use_example = st.checkbox("Or use the bundled example", value=False)
            bulk = (pd.read_csv("example/example_bulk.tsv", sep="\t", index_col=0)
                    if use_example else read_matrix(bulk_f))

        default_sig = demo.get("sig", "g0\ng1\ng2\ng3\ng4\ng5\ng6\ng7\ng8\ng9") if demo else "g0\ng1\ng2\ng3\ng4\ng5\ng6\ng7\ng8\ng9"
        sig_text = st.text_area(
            "Signature genes (one per line)",
            default_sig,
            help="Marker genes of the cell-state / programme you want to score. "
                 "One gene per line; gene symbols must match the rows of your bulk matrix.",
            placeholder="CD34\nCD99\nKIT\nFLT3\nPROM1\nIL3RA",
        )
        st.caption("💡 Marker genes of the programme to score "
                   "(e.g. AML stemness → CD34, CD99, KIT, FLT3).")

    with c2, st.container(border=True):
        st.markdown('<div class="progscore-section">2 · Method</div>', unsafe_allow_html=True)
        task = st.radio("Task", ["Score a signature (projection)",
                                 "Deconvolve (reference-based)"])
        if task.startswith("Score"):
            method = st.selectbox("Scoring method", list(PROJECTION),
                                  index=list(PROJECTION).index(demo.get("method", "mean_z")) if demo else 0,
                                  format_func=lambda m: PROJECTION[m])
        else:
            method = st.selectbox("Deconvolution method", list(DECONV), format_func=lambda m: DECONV[m])
            ref_f = st.file_uploader("Reference profiles (genes × cell types)", type=["tsv", "csv"])
            var_f = st.file_uploader("Reference variance (optional)", type=["tsv", "csv"])

        if task.startswith("Score") and method == "cmps":
            st.markdown("**CMPS reference**")
            ref_type = st.radio("Reference type", ["mean", "specificity"],
                                index=0 if demo is None or demo.get("ref_type") == "mean" else 1,
                                help="mean = programme's per-gene mean expression (AML LSC); "
                                     "specificity = +1 stem / −1 diff / 0 pan (T-ALL ETP).")
            if ref_type == "mean":
                if demo and demo.get("ref"):
                    st.caption(f"Reference loaded: {demo['ref'].split('/')[-1]}")
                else:
                    cmps_ref_f = st.file_uploader(
                        "Single-cell programme reference (TSV: gene, mean)",
                        type=["tsv", "csv"],
                        help="The mean expression of each signature gene in the single-cell "
                             "programme (e.g. the average expression of LSC markers across "
                             "LSC cells). Two columns: gene, mean.")
                    st.caption("💡 The single-cell mean-expression profile of your programme: "
                               "two columns `gene, mean` (e.g. `CD34 3.1`, `CD99 5.2`, …). "
                               "Average each gene's expression across the single cells of "
                               "that programme.")
            else:
                stem_text = st.text_area("Stem-specific genes (one per line)",
                                         demo.get("stem", "\n".join(TALL_STEM)) if demo else "\n".join(TALL_STEM),
                                         height=90)
                diff_text = st.text_area("Differentiation genes (one per line)",
                                         demo.get("diff", "\n".join(TALL_DIFF)) if demo else "\n".join(TALL_DIFF),
                                         height=90)
                pan_text = st.text_area("Pan-marker genes (optional)",
                                        demo.get("pan", "\n".join(TALL_PAN)) if demo else "\n".join(TALL_PAN),
                                        height=70)

    with c3, st.container(border=True):
        st.markdown('<div class="progscore-section">3 · Run</div>', unsafe_allow_html=True)
        go = st.button("▶ Run analysis", type="primary", use_container_width=True)
        st.caption("Click to score every sample. Results appear below.")

    tab_analysis, tab_about = st.tabs(["📊 Analysis", "ℹ️ Help"])

    with tab_analysis:
        if bulk is None:
            st.info("Upload your bulk matrix (or pick a demo) and press **Run analysis**.")
            footer(); return
        if not go:
            st.info("Press **Run analysis** above.")
            footer(); return

        sig_genes = [g.strip() for g in sig_text.splitlines() if g.strip()]

        with st.spinner("Running…"):
            if task.startswith("Score"):
                if method == "cmps":
                    if ref_type == "specificity":
                        stem = [g.strip() for g in stem_text.splitlines() if g.strip()]
                        diff = [g.strip() for g in diff_text.splitlines() if g.strip()]
                        pan = [g.strip() for g in pan_text.splitlines() if g.strip()]
                        ref = make_specificity_reference(stem, diff, pan)
                        ref_t = "specificity"
                    else:
                        ref = {}
                        if demo and demo.get("ref"):
                            d = pd.read_csv(demo["ref"])
                            ref = dict(zip(d.iloc[:, 0].astype(str), d.iloc[:, 1].astype(float)))
                        elif cmps_ref_f is not None:
                            d = pd.read_csv(io.BytesIO(cmps_ref_f.getvalue()), sep=None, engine="python")
                            ref = dict(zip(d.iloc[:, 0].astype(str), d.iloc[:, 1].astype(float)))
                        if not ref:
                            st.warning("CMPS needs a single-cell reference; fell back to a flat reference.")
                            ref = {g: 1.0 for g in sig_genes}
                        ref_t = "mean"
                    score = cmps(bulk, sig_genes, ref, ref_type=ref_t)
                else:
                    score = score_signature(bulk, sig_genes, method=method)
                result = pd.Series(score, index=bulk.columns, name="score")
                label = demo.get("label", "signature programme") if demo else "signature programme"
            else:
                ref = read_matrix(ref_f)
                if ref is None:
                    st.error("Deconvolution needs a reference profile."); footer(); return
                ref_var = read_matrix(var_f) if var_f is not None else None
                frac = deconvolve(bulk, ref, ref_var, method=method)
                result = pd.Series(frac, index=bulk.columns, name="fraction")
                label = "cell-type fraction"

        show_results(result, label, demo)

    with tab_about:
        about_tab()

    footer()


if __name__ == "__main__":
    main()
