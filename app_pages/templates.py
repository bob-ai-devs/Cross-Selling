import pandas as pd
import streamlit as st

from core import ui
from core.config import (ENCODED_COLS, FEATURE_COLS, ID_COL, NUMERIC_COLS, OPTIONS, PRODUCT_COLS,
                         PRODUCTS)
from core.demo import make_batch_demo, make_history_demo
from core.preprocessing import pretty


def _schema() -> pd.DataFrame:
    rows = [{"Column": ID_COL, "Type": "number / text", "Allowed values": "any unique identifier (optional)"}]
    hints = {"Age": "18–70", "Annual_Income": "number", "Credit_Score": "300–900",
             "Past_Loan_Defaults": "0/1 (or Yes/No)", "Credit_Utilization_Ratio": "0.0–1.0",
             "Liquid_Assets": "number", "Investments": "number", "Years_of_Employment": "0–40"}
    for c in FEATURE_COLS:
        if c in ENCODED_COLS:
            rows.append({"Column": c, "Type": "category", "Allowed values": ", ".join(OPTIONS[c])})
        else:
            rows.append({"Column": c, "Type": "number", "Allowed values": hints.get(c, "number")})
    for c in PRODUCT_COLS:
        rows.append({"Column": c, "Type": "category",
                     "Allowed values": "blank or: " + ", ".join(pretty(p) for p in PRODUCTS)})
    return pd.DataFrame(rows)


def render():
    ui.hero("Templates & Demo Data", "Starter files, the expected schema, and synthetic data to try the app.")

    t1, t2, t3 = st.tabs(["Batch prediction template", "Training template", "Demo data generator"])

    with t1:
        st.markdown("Layout expected by the **pre-trained network**. Header names are matched "
                    "case-insensitively (spaces/underscores ignored); anything else can be mapped in the app.")
        st.dataframe(_schema(), hide_index=True)
        sample = make_batch_demo(5, seed=3)
        st.dataframe(sample, hide_index=True)
        st.download_button("⬇️ Download template (CSV)", ui.csv_bytes(sample),
                           file_name="batch_template.csv", mime="text/csv")
        st.caption("Products are listed in the order the customer acquired them; leave unused slots blank.")

    with t2:
        st.markdown("Layout for **custom training**: any customer attributes, plus pairs of "
                    "*product* and *acquired-date* columns (up to 10 pairs). Products are re-ordered "
                    "chronologically before training.")
        sample = make_history_demo(5, seed=3)
        st.dataframe(sample, hide_index=True)
        st.download_button("⬇️ Download template (CSV)", ui.csv_bytes(sample),
                           file_name="training_template.csv", mime="text/csv")

    with t3:
        st.markdown("Generate synthetic customers (created in memory, never stored).")
        a, b = st.columns(2)
        with a:
            n1 = st.slider("Customers for batch demo", 50, 5000, 500, step=50)
            demo1 = make_batch_demo(n1)
            st.download_button("⬇️ Batch demo (CSV)", ui.csv_bytes(demo1), file_name="demo_batch.csv", mime="text/csv")
        with b:
            n2 = st.slider("Customers for training demo", 300, 10000, 2000, step=100)
            demo2 = make_history_demo(n2)
            st.download_button("⬇️ Training demo (CSV)", ui.csv_bytes(demo2), file_name="demo_history.csv", mime="text/csv")
        st.caption("Tip: the Batch and Train pages also have a built-in *Use demo data* option — "
                   "no download needed.")
