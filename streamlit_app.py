"""AI Cross-Selling Recommendation Engine — Streamlit edition.

Run locally:   streamlit run streamlit_app.py
Deploy:        push this folder to GitHub, then point Streamlit Community Cloud at
               streamlit_app.py and paste the secrets (see README.md).
"""
import streamlit as st

st.set_page_config(page_title="AI Cross-Selling Engine", page_icon="🎯",
                   layout="wide", initial_sidebar_state="expanded")

from app_pages import batch, home, insights, single, templates, train  # noqa: E402
from core import ui  # noqa: E402

ui.inject_css()

navigation = st.navigation({
    "Engine": [
        st.Page(home.render, title="Home", icon="🏠", url_path="home", default=True),
        st.Page(single.render, title="Single Customer", icon="👤", url_path="single"),
        st.Page(batch.render, title="Batch Prediction", icon="📂", url_path="batch"),
        st.Page(train.render, title="Train Custom Model", icon="🧪", url_path="train"),
    ],
    "Analytics": [
        st.Page(insights.render, title="Insights Dashboard", icon="📊", url_path="insights"),
    ],
    "Resources": [
        st.Page(templates.render, title="Templates & Demo Data", icon="🧰", url_path="templates"),
    ],
})
ui.sidebar_status()
navigation.run()
