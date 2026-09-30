import streamlit as st

from core import storage, ui
from core.config import ALL_ARTIFACTS, PRETRAINED_F1


def render():
    ui.hero("AI Cross-Selling Recommendation Engine",
            "Predict the next-best products for every customer — single lookups, batch scoring, "
            "custom model training and an analytics dashboard.")

    # ---- entry points ------------------------------------------------------
    c1, c2, c3, c4 = st.columns(4)
    with c1, st.container(border=True):
        st.markdown("#### 👤 Single customer")
        st.caption("Fill a form and get the top-3 products with affinity and confidence.")
    with c2, st.container(border=True):
        st.markdown("#### 📂 Batch prediction")
        st.caption("Upload a CSV/Excel file, score every customer, download the results.")
    with c3, st.container(border=True):
        st.markdown("#### 🧪 Train custom model")
        st.caption("Upload historical holdings and train a network tuned to your own data.")
    with c4, st.container(border=True):
        st.markdown("#### 📊 Insights")
        st.caption("Product mix, segment heat-maps, gap analysis, campaign lists.")

    st.info("Use the sidebar to switch modules. No data you upload or generate is stored — "
            "results stay in this browser session and can be viewed or downloaded.")

    # ---- model status (lightweight: does not import TensorFlow) -------------
    st.subheader("Pre-trained model")
    token, gist_id = storage.credentials()
    local = storage.local_files_present()
    left, right = st.columns([1, 1])
    with left:
        st.metric("Reported F1-score (original app)", f"{PRETRAINED_F1:.2f} %")
        st.markdown(
            f"- **Gist secret:** {'✅ configured' if gist_id else '➖ not set'}\n"
            f"- **GitHub token:** {'✅ configured' if token else '➖ not set (public gist only)'}\n"
            f"- **Bundled repo files:** {sum(local.values())}/{len(local)} found")
    with right:
        src = st.session_state.get("_art_source")
        if st.session_state.get("_art_error"):
            st.error(st.session_state["_art_error"])
        elif src:
            st.success(f"Loaded from: {src}")
        else:
            st.caption("The model is loaded the first time a page needs it (takes a few seconds).")
        if st.button("Test load now"):
            with st.spinner("Loading…"):
                art, err = storage.get_artifacts()
            (st.success(f"Loaded from {art['source']}") if art else st.error(err))

    with st.expander("How the engine works"):
        st.markdown("""
1. **Profile** — 15 customer attributes plus up to five current products (in acquisition order).
2. **Network** — a shared dense trunk feeds three heads (rank 1, 2, 3); each head predicts a product
   (softmax) and an **affinity** score.
3. **Business rules (optional, on by default)** — never suggest a product the customer already holds
   or the same product twice.
4. **Confidence** — the network's own probability for the suggested product.

**Custom training** builds recommendation labels by clustering customers and ranking each cluster's most
popular products the customer doesn't yet hold; the network then learns to reproduce that logic.
Reported scores therefore measure agreement with those labels, not real-world take-up.
""")

    with st.expander("Files expected in the Gist / repo"):
        st.code("\n".join(ALL_ARTIFACTS), language="text")
        st.caption("In a Gist, store each file base64-encoded with a `.b64` suffix — "
                   "`tools/publish_artifacts_to_gist.py` does this for you.")
