"""Small shared UI helpers."""
from __future__ import annotations

import streamlit as st

from core import storage
from core.config import ROOT

CSS = """
<style>
.block-container {padding-top: 2.2rem; max-width: 1250px;}
.hero {padding: 1.4rem 1.6rem; border-radius: 14px; margin-bottom: 1rem;
       background: linear-gradient(120deg, #2b2d42 0%, #4a4e69 100%); color: #fff;}
.hero h1 {margin: 0 0 .3rem 0; font-size: 1.9rem; color: #fff;}
.hero p {margin: 0; opacity: .85;}
.rec-card {border: 1px solid rgba(128,128,128,.28); border-radius: 12px; padding: 1rem 1.1rem;}
.rec-rank {font-size: .75rem; letter-spacing: .08em; text-transform: uppercase; opacity: .65;}
.rec-name {font-size: 1.45rem; font-weight: 700; margin: .1rem 0 .4rem 0;}
.small-muted {font-size: .82rem; opacity: .7;}
div[data-testid="stMetric"] {border: 1px solid rgba(128,128,128,.22); border-radius: 10px; padding: .6rem .8rem;}
</style>
"""


def inject_css() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


def hero(title: str, subtitle: str = "") -> None:
    banner = ROOT / "assets" / "banner.jpg"
    if banner.exists():
        st.image(str(banner))
    st.markdown(f'<div class="hero"><h1>{title}</h1><p>{subtitle}</p></div>', unsafe_allow_html=True)


def require_artifacts() -> dict:
    """Load the pre-trained model or show a helpful error and stop the page."""
    art, err = storage.get_artifacts()
    if art is None:
        st.error("The pre-trained model could not be loaded.")
        st.code(err or "unknown error")
        st.caption("Check the Gist secrets / bundled files, then retry.")
        if st.button("🔄 Retry loading"):
            storage.reload_artifacts()
            st.rerun()
        st.stop()
    for note in art.get("notes", []):
        st.warning(note)
    return art


def csv_bytes(df) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


def sidebar_status() -> None:
    with st.sidebar:
        st.divider()
        st.caption("SESSION STATUS")
        src = st.session_state.get("_art_source")
        if st.session_state.get("_art_error"):
            st.markdown("🔴 Pre-trained model: **error**")
        elif src:
            st.markdown(f"🟢 Pre-trained model: **loaded**  \n<span class='small-muted'>{src}</span>",
                        unsafe_allow_html=True)
        else:
            st.markdown("⚪ Pre-trained model: *loads on first use*")

        cm = st.session_state.get("custom_bundle")
        st.markdown(f"🟢 Custom model: **trained** ({cm['trained_at']})" if cm
                    else "⚪ Custom model: *not trained*")
        res = st.session_state.get("results")
        st.markdown(f"🟢 Batch results: **{len(res['df']):,} rows**" if res
                    else "⚪ Batch results: *none*")
        st.caption("Nothing is stored: models and results live only in this browser session.")

        c1, c2 = st.columns(2)
        if c1.button("♻️ Reset", help="Clear the custom model, results and all inputs for this session."):
            for k in list(st.session_state.keys()):
                if not k.startswith("_art"):
                    del st.session_state[k]
            st.rerun()
        if c2.button("🔄 Reload", help="Re-download the pre-trained model artifacts."):
            storage.reload_artifacts()
            st.rerun()
