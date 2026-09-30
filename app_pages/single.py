import json

import numpy as np
import pandas as pd
import streamlit as st

from core import analytics, ui
from core.config import (FEATURE_COLS, OPTIONS, PRESETS, PRODUCT_COLS, PRODUCTS, TOP_N)
from core.inference import predict_single
from core.preprocessing import InputError, pretty

DEFAULTS = dict(
    CustomerID=1, Age=35, Gender="Male", Marital_Status="Married", Annual_Income=900000,
    Employment_Type="Salaried", Income_Stability="Stable", Credit_Score=720,
    Past_Loan_Defaults="No", Credit_Utilization_Ratio=0.30, Property_Ownership="Rent",
    Liquid_Assets=150000, Investments=100000, Job_Title="Manager", Company_Size="Medium",
    Years_of_Employment=8, p1="savings", p2="None", p3="None", p4="None", p5="None",
)


def _apply_preset():
    preset = PRESETS.get(st.session_state.get("s_preset"))
    if preset:
        for k, v in preset.items():
            st.session_state[f"s_{k}"] = v


def render():
    ui.hero("Single Customer Recommendation",
            "Enter one customer's profile to see the three best next products.")
    art = ui.require_artifacts()

    for k, v in DEFAULTS.items():
        st.session_state.setdefault(f"s_{k}", v)

    top = st.columns([2, 2, 3])
    top[0].selectbox("Start from a sample profile", list(PRESETS), key="s_preset",
                     on_change=_apply_preset)
    apply_rules = top[1].toggle("Apply business rules", value=True, key="s_rules",
                                help="Exclude products the customer already holds and avoid duplicates.")

    with st.form("single_form"):
        st.markdown("##### Profile")
        c = st.columns(3)
        c[0].number_input("Customer ID", min_value=0, step=1, key="s_CustomerID")
        c[1].number_input("Age", 18, 70, key="s_Age")
        c[2].selectbox("Gender", OPTIONS["Gender"], key="s_Gender")

        c = st.columns(3)
        c[0].selectbox("Marital status", OPTIONS["Marital_Status"], key="s_Marital_Status")
        c[1].selectbox("Employment type", OPTIONS["Employment_Type"], key="s_Employment_Type")
        c[2].selectbox("Income stability", OPTIONS["Income_Stability"], key="s_Income_Stability")

        c = st.columns(3)
        c[0].number_input("Annual income", 0, step=10000, key="s_Annual_Income")
        c[1].number_input("Liquid assets", 0, step=10000, key="s_Liquid_Assets")
        c[2].number_input("Investments", 0, step=10000, key="s_Investments")

        c = st.columns(3)
        c[0].number_input("Credit score", 300, 900, key="s_Credit_Score")
        c[1].number_input("Credit utilization ratio", 0.0, 1.0, step=0.01, key="s_Credit_Utilization_Ratio")
        c[2].selectbox("Past loan defaults", ["No", "Yes"], key="s_Past_Loan_Defaults")

        c = st.columns(3)
        c[0].selectbox("Property ownership", OPTIONS["Property_Ownership"], key="s_Property_Ownership")
        c[1].selectbox("Job title", OPTIONS["Job_Title"], key="s_Job_Title")
        c[2].selectbox("Company size", OPTIONS["Company_Size"], key="s_Company_Size")
        st.number_input("Years of employment", 0, 40, key="s_Years_of_Employment")

        st.markdown("##### Products currently held (in the order they were acquired)")
        pc = st.columns(5)
        for i in range(5):
            pc[i].selectbox(f"Product {i + 1}", ["None"] + PRODUCTS, format_func=lambda x: "None" if x == "None" else pretty(x),
                            key=f"s_p{i + 1}")
        submitted = st.form_submit_button("Get recommendations", type="primary")

    if submitted:
        ss = st.session_state
        held = [ss[f"s_p{i}"] for i in range(1, 6)]
        chosen = [h for h in held if h != "None"]
        if len(chosen) != len(set(chosen)):
            st.warning("The same product is listed more than once — the duplicates are counted as one holding.")
        values = {c: ss[f"s_{c}"] for c in FEATURE_COLS}
        values["Past_Loan_Defaults"] = 1 if values["Past_Loan_Defaults"] == "Yes" else 0
        for col, h in zip(PRODUCT_COLS, held):
            values[col] = "" if h == "None" else h
        try:
            with st.spinner("Scoring…"):
                res = predict_single(values, art, apply_rules)
        except InputError as e:
            for m in e.messages:
                st.error(m)
            st.stop()
        except Exception as e:  # keep the app alive on unexpected model errors
            st.error(f"Prediction failed: {e}")
            st.stop()
        st.session_state["single_result"] = dict(
            recs=res.recs.iloc[0].to_dict(), customer_id=ss["s_CustomerID"], values=values,
            probs=np.mean([res.preds[0][0], res.preds[2][0], res.preds[4][0]], axis=0),
            classes=[pretty(x) for x in res.product_enc.classes],
            blank=res.product_enc.blank_index, held=[pretty(h) for h in chosen], rules=apply_rules)

    out = st.session_state.get("single_result")
    if not out:
        return

    st.divider()
    st.subheader(f"Recommendations for customer {out['customer_id']}")
    cols = st.columns(TOP_N)
    for i, col in enumerate(cols, start=1):
        prod = out["recs"][f"Pred_Target_{i}"]
        aff = out["recs"][f"Pred_Affinity_{i}"]
        conf = out["recs"][f"Pred_Confidence_{i}"]
        with col:
            st.markdown(f'<div class="rec-card"><div class="rec-rank">Rank {i}</div>'
                        f'<div class="rec-name">{prod}</div></div>', unsafe_allow_html=True)
            if prod == "—" or pd.isna(aff):
                st.caption("No further product available under the business rules.")
            else:
                st.progress(min(max(aff / 100, 0.0), 1.0), text=f"Affinity {aff:.1f} / 100")
                st.caption(f"Model confidence: {conf:.1f}%")

    keep = [k for k, name in enumerate(out["classes"]) if k != out["blank"]]
    labels = [out["classes"][k] for k in keep]
    st.markdown("##### Where the model's interest lies")
    st.plotly_chart(analytics.fig_probability_bars(np.asarray(out["probs"])[keep], labels, set(out["held"])),
                    key="single_prob_chart")
    st.caption("Grey bars are products the customer already holds.")

    flat = {"CustomerID": out["customer_id"], **out["values"], **out["recs"]}
    d1, d2, _ = st.columns([1, 1, 3])
    d1.download_button("⬇️ Download CSV", ui.csv_bytes(pd.DataFrame([flat])),
                       file_name=f"recommendation_{out['customer_id']}.csv", mime="text/csv")
    d2.download_button("⬇️ Download JSON", json.dumps(flat, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o)),
                       file_name=f"recommendation_{out['customer_id']}.json", mime="application/json")
