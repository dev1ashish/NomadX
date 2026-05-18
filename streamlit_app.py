"""Atlas Raman classifier — Streamlit demo.

Loads the Stage 15F production artifacts and lets a reviewer upload an Atlas
`.xls` / `.txt` file to see the predicted primary class + probability bars
+ mean spectrum plot.

Run locally:
    .venv/bin/streamlit run streamlit_app.py

Cloud deploy: Streamlit Community Cloud / HuggingFace Spaces. See `DEPLOY.md`.
"""
from __future__ import annotations

import json
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from atlas.band_features import BANDS, EMPIRICAL_ANCHOR_BANDS, PRIMARY_TRIPLE
from atlas.inference import (
    model_metadata,
    predict_from_xls,
)


# ---------------------------------------------------------------------------
# Page config + cached resource loaders
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Atlas Raman Bacterial Classifier",
    page_icon="🧫",
    layout="wide",
)


@st.cache_resource
def _metadata() -> dict:
    return model_metadata()


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.title("🧫 Atlas Raman classifier")
    st.write(
        "**4-class bacterial Raman classifier** trained on 87 Atlas confocal "
        "hyperspectral maps (7,122 QC-passed pixel spectra). Upload an Atlas "
        "`.xls` or `.txt` file to see the predicted class."
    )
    try:
        meta = _metadata()
        st.markdown(f"**Model:** `{meta.get('model_type', '?')}`")
        st.markdown(f"**LOSO mean parent-class accuracy:** "
                    f"`{meta.get('loso_mean_accuracy', float('nan')):.3f}` "
                    f"± `{meta.get('loso_std_accuracy', 0):.3f}`")
        st.markdown(f"**LOSO mean macro recall:** "
                    f"`{meta.get('loso_mean_macro_recall', float('nan')):.3f}`")
        st.markdown(f"**Branch verdict:** `({meta.get('branch_hit', '?')})`")
        st.markdown(f"**Features used:** `{meta.get('feature_count', '?')}` "
                    f"(MI-selected from 259)")
        with st.expander("Per-strain LOSO accuracy"):
            ps = meta.get("per_strain_accuracy", {})
            if ps:
                ps_df = pd.DataFrame(
                    sorted(ps.items()), columns=["strain", "accuracy"]
                )
                st.dataframe(ps_df, use_container_width=True, hide_index=True)
        with st.expander("Algorithm ablation"):
            ac = meta.get("algo_comparison", {})
            if ac:
                rows = []
                for a, s in ac.items():
                    rows.append({
                        "algorithm": a,
                        "loso_acc": s.get("mean_loso_accuracy"),
                        "loso_macro_recall": s.get("mean_loso_macro_recall"),
                    })
                st.dataframe(pd.DataFrame(rows), use_container_width=True,
                             hide_index=True)
    except FileNotFoundError:
        st.error(
            "**Artifacts not found.** Run "
            "`.venv/bin/python scripts/run_stage15f_final.py` first."
        )

    st.markdown("---")
    st.markdown(
        "**Project links**\n\n"
        "- [PAPER.md](https://github.com/) — full methods + results\n"
        "- [README.md](https://github.com/) — quick start\n"
        "- 9 bacterial strains: STEC (O157H7, O121H19, O103H2), "
        "Non-STEC (ATCC25922, 83972, K-12), Salmonella (Dublin, Heidelburg, "
        "Typhimurium), and H₂O blanks"
    )


# ---------------------------------------------------------------------------
# Main panel
# ---------------------------------------------------------------------------

st.title("Predict bacterial class from a Raman map")

st.write(
    "Upload an Atlas-format Raman hyperspectral map. The classifier will "
    "parse + preprocess + extract features and return a predicted primary "
    "class (STEC / Non-STEC / Salmonella / H₂O) with per-class probabilities."
)

uploaded = st.file_uploader(
    "Atlas Raman file (.xls / .txt)",
    type=["xls", "txt"],
    help="Tab-delimited Atlas format with ~44 header lines + wavenumber row + "
         "pixel rows. See `Atlas Data/STEC/O157H7/*.xls` for the canonical layout.",
)


def _plot_spectrum(wn: np.ndarray, y: np.ndarray) -> None:
    """Mean spectrum line plot with named-band annotations."""
    df = pd.DataFrame({"wavenumber (cm⁻¹)": wn, "intensity (SNV)": y})
    st.line_chart(df, x="wavenumber (cm⁻¹)", y="intensity (SNV)",
                  height=320)


def _plot_bars(proba: dict[str, float]) -> None:
    if not proba:
        st.info("Model does not expose probabilities for this run.")
        return
    rows = [{"class": c, "probability": float(p)} for c, p in proba.items()]
    df = pd.DataFrame(rows).sort_values("probability", ascending=False)
    st.bar_chart(df, x="class", y="probability", height=240)


if uploaded is not None:
    with st.spinner("Parsing + preprocessing + classifying…"):
        # Streamlit gives us a BytesIO-like; write to a tmp file so the
        # existing parse_file path is reused as-is.
        suffix = "." + uploaded.name.rsplit(".", 1)[-1]
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(uploaded.getvalue())
            tmp_path = Path(tmp.name)
        t0 = time.time()
        try:
            result = predict_from_xls(tmp_path)
        except Exception as e:
            st.error(f"Prediction failed: {e}")
            st.stop()
        latency_ms = (time.time() - t0) * 1000

    pred = result["class"]
    proba = result.get("probabilities", {})
    wn = np.asarray(result["wn"])
    spec_mean = np.asarray(result["spectrum_mean"])

    # ---- predicted class banner ----
    color = {
        "STEC":      "#d63333",
        "Non-STEC":  "#1f7a4d",
        "Salmonella":"#7a3d99",
        "H2O":       "#3070b5",
    }.get(pred, "#444")
    st.markdown(
        f"""
        <div style='padding:16px;background:{color};color:white;border-radius:8px;
                    font-size:28px;font-weight:600;text-align:center;'>
            Predicted class: <b>{pred}</b>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.caption(f"Latency: {latency_ms:.0f} ms  •  filename: {uploaded.name}")

    col_proba, col_spec = st.columns([1, 2])
    with col_proba:
        st.subheader("Per-class probability")
        _plot_bars(proba)
    with col_spec:
        st.subheader("Mean preprocessed spectrum")
        _plot_spectrum(wn, spec_mean)
        st.caption(
            "Preprocessed: cosmic-ray removed + arPLS baseline + Savitzky-Golay smoothed + "
            "cropped to fingerprint (400–1800) + C-H stretch (2800–3050) + SNV-normalized. "
            "Empirical STEC↔Non-STEC anchor: 1117 / 1194 cm⁻¹ (LPS chain region)."
        )

    with st.expander("Selected feature values (top contributors)"):
        feat_vals = result.get("feature_values", {})
        if feat_vals:
            rows = [{"feature": k, "value": v} for k, v in feat_vals.items()]
            df = pd.DataFrame(rows)
            st.dataframe(df, use_container_width=True, hide_index=True,
                         height=min(400, 28 + 28 * min(len(df), 12)))
        else:
            st.write("No feature values returned.")
else:
    st.info(
        "👈 Upload a file to get a prediction. Sample files live in "
        "`Atlas Data/<class>/<strain>/*.xls` if you cloned the repo locally."
    )

    with st.expander("How the classifier works (in plain English)"):
        st.markdown(
            """
            1. **Parse** the tab-delimited Atlas file into a stack of pixel
               spectra (one per scan position).
            2. **Preprocess** each spectrum: remove cosmic-ray spikes, subtract
               an asymmetric baseline (arPLS), Savitzky-Golay smooth, crop to
               the chemistry-informative regions (400–1800 + 2800–3050 cm⁻¹),
               and SNV-normalize.
            3. **Extract 259 features** spanning:
               - 166 band-aware AUCs, ratios, peak fits, ROI moments,
                 EMSC scatter coefficients, and derivatives,
               - 51 wavelet-energy + ROI-PCA + spectral-angle-mapper features,
               - 32 MCR-ALS spectral-unmixing concentrations (the project's
                 strongest single feature),
               - 10 per-file spatial-moment statistics.
            4. **Mean-pool** per-pixel features per file (87 files × 259
               features), then select the **top 35** via mutual information.
            5. **Classify** with the production pipeline (a `StandardScaler →
               classifier` saved from Stage 15F).
            """
        )
