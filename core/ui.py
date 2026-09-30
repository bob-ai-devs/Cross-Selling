"""Small shared UI helpers."""
from __future__ import annotations

import streamlit as st

from core import storage
from core.config import ROOT


# ==============================================================================
# BANK OF BARODA BRAND PALETTE
# ==============================================================================
BOB_ORANGE = "#F7941D"       # primary — "Baroda Sun"
BOB_ORANGE_DEEP = "#E8531B"  # sun-ray gradient end
BOB_MAROON = "#8E1B3A"       # sun-ray gradient end / accents
BOB_NAVY = "#12284C"         # wordmark / headings
BOB_NAVY_LIGHT = "#1E3E73"
BOB_CREAM = "#FFF8F1"        # page background
BOB_GREY = "#5B6675"


CSS = f"""
<style>

.block-container {{
    padding-top: 2.2rem;
    max-width: 1250px;
}}

/* =============================================================================
   BANK OF BARODA HERO
   The hero is the main page banner — no separate BOB banner is required.
   ============================================================================= */

.hero {{
    background: radial-gradient(
        circle at 15% 50%,
        {BOB_ORANGE} 0%,
        {BOB_ORANGE_DEEP} 45%,
        {BOB_MAROON} 100%
    );

    padding: 1.4rem 1.6rem;
    border-radius: 14px;
    margin-bottom: 1rem;

    color: #FFFFFF;

    box-shadow: 0 4px 14px rgba(0,0,0,0.15);
}}

.hero h1 {{
    margin: 0 0 .3rem 0;
    font-size: 1.9rem;
    color: #FFFFFF;
    font-weight: 800;
    letter-spacing: 0.3px;
}}

.hero p {{
    margin: 0;
    color: #FFEFE0;
    font-size: .95rem;
}}


/* =============================================================================
   RECOMMENDATION CARDS
   ============================================================================= */

.rec-card {{
    border: 1px solid rgba(128,128,128,.28);
    border-radius: 12px;
    padding: 1rem 1.1rem;
    background-color: #FFFFFF;
}}

.rec-rank {{
    font-size: .75rem;
    letter-spacing: .08em;
    text-transform: uppercase;
    color: {BOB_GREY};
}}

.rec-name {{
    font-size: 1.45rem;
    font-weight: 700;
    margin: .1rem 0 .4rem 0;
    color: {BOB_NAVY};
}}

.small-muted {{
    font-size: .82rem;
    color: {BOB_GREY};
}}


/* =============================================================================
   METRIC CARDS
   ============================================================================= */

div[data-testid="stMetric"] {{
    border: 1px solid rgba(128,128,128,.22);
    border-radius: 10px;
    padding: .6rem .8rem;
    background-color: #FFFFFF;
}}


/* =============================================================================
   PAGE BACKGROUND
   ============================================================================= */

.stApp {{
    background-color: {BOB_CREAM};
}}


/* =============================================================================
   HEADINGS
   ============================================================================= */

h1, h2, h3 {{
    color: {BOB_NAVY};
}}

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
