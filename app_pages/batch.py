from datetime import datetime

import pandas as pd
import streamlit as st

from core import analytics, ui
from core.config import (ALIASES, FEATURE_COLS, ID_COL, PRODUCT_COLS, TOP_N)
from core.demo import make_batch_demo, make_history_demo
from core.inference import attach, predict_custom, predict_pretrained
from core.preprocessing import InputError, auto_map, profile_frame
from streamlit_app import insights_page

NONE_OPT = "— none (use row number) —"


def read_table(file) -> pd.DataFrame:
    name = file.name.lower()
    if name.endswith((".xlsx", ".xls")):
        return pd.read_excel(file)
    try:
        return pd.read_csv(file)
    except UnicodeDecodeError:
        file.seek(0)
        return pd.read_csv(file, encoding="latin-1")


def render():
    ui.hero("Batch Prediction", "Score a whole customer file, then explore or download the results.")
    ss = st.session_state
    custom = ss.get("custom_bundle")

    # ---- 1 model -----------------------------------------------------------
    st.subheader("1 · Model")
    choices = ["Pre-trained network"] + (["Custom network (trained this session)"] if custom else [])
    model_choice = st.radio("Recommendation model", choices, horizontal=True, label_visibility="collapsed")
    use_custom = model_choice != "Pre-trained network"
    if not custom:
        st.caption("Train a custom model on the *Train Custom Model* page to unlock the second option.")
    apply_rules = st.toggle("Apply business rules (no already-held or duplicate products)", value=True)

    # ---- 2 data ------------------------------------------------------------
    st.subheader("2 · Data")
    src = st.radio("Data source", ["Upload file", "Use demo data"], horizontal=True, label_visibility="collapsed")
    df = None
    if src == "Upload file":
        file = st.file_uploader("CSV or Excel file", type=["csv", "xlsx", "xls"])
        if file is not None:
            try:
                df = read_table(file)
            except Exception as e:
                st.error(f"Could not read the file: {e}")
    else:
        n = st.slider("Demo customers", 50, 2000, 300, step=50)
        df = make_history_demo(n, seed=99) if use_custom else make_batch_demo(n)
        st.caption("Synthetic data generated in memory for exploration.")

    if df is None or df.empty:
        st.info("Provide data to continue. Need a starting point? See *Templates & Demo Data*.")
        return

    st.success(f"{len(df):,} rows × {df.shape[1]} columns loaded")
    with st.expander("Preview and data quality"):
        st.dataframe(df.head(50), hide_index=True)
        st.dataframe(profile_frame(df), hide_index=True)

    # ---- 3 mapping ---------------------------------------------------------
    id_col = None
    colmap = {}
    if use_custom:
        guess = auto_map(df.columns, [ID_COL], ALIASES)[ID_COL]
        options = [NONE_OPT] + list(df.columns)
        pick = st.selectbox("Customer identifier column (optional)", options,
                            index=options.index(guess) if guess in options else 0)
        id_col = None if pick == NONE_OPT else pick
    else:
        required = FEATURE_COLS + PRODUCT_COLS
        colmap = auto_map(df.columns, required, ALIASES)
        missing = [c for c in required if colmap[c] is None]
        guess = auto_map(df.columns, [ID_COL], ALIASES)[ID_COL]
        if missing:
            st.warning(f"{len(missing)} required column(s) were not found automatically — "
                       "map them below (or rename them in your file).")
        with st.expander("Column mapping", expanded=bool(missing)):
            opts = list(df.columns)
            cols = st.columns(3)
            for i, req in enumerate(required):
                current = colmap[req]
                colmap[req] = cols[i % 3].selectbox(
                    req, [None] + opts, index=(opts.index(current) + 1) if current else 0,
                    format_func=lambda x: "— select —" if x is None else x, key=f"map_{req}")
            id_opts = [NONE_OPT] + opts
            pick = st.selectbox("Customer identifier column (optional)", id_opts,
                                index=id_opts.index(guess) if guess in id_opts else 0)
            id_col = None if pick == NONE_OPT else pick
        if any(v is None for v in colmap.values()):
            st.stop()

    # ---- 4 run -------------------------------------------------------------
    if not st.button("🚀 Run predictions", type="primary"):
        if not ss.get("results"):
            return
    else:
        try:
            with st.spinner("Scoring customers…"):
                if use_custom:
                    res = predict_custom(df, custom, apply_rules)
                    product_cols = custom["product_cols"]
                    label = "Custom network"
                else:
                    art = ui.require_artifacts()
                    res = predict_pretrained(df, colmap, art, apply_rules)
                    product_cols = [colmap[c] for c in PRODUCT_COLS]
                    label = "Pre-trained network"
        except InputError as e:
            for m in e.messages:
                st.error(m)
            return
        except Exception as e:
            st.error(f"Prediction failed: {e}")
            return
        for note in res.warnings:
            st.warning(note)
        ss["results"] = dict(df=attach(df, res), source=label, time=datetime.now().strftime("%H:%M:%S"),
                             product_cols=product_cols, id_col=id_col, rules=apply_rules)
        st.toast("Predictions ready", icon="✅")

    show_results()


def show_results():
    r = st.session_state["results"]
    out = r["df"]
    st.divider()
    st.subheader("3 · Results")
    st.caption(f"{r['source']} · {len(out):,} customers · scored at {r['time']} · "
               f"business rules {'on' if r['rules'] else 'off'}")

    k = analytics.kpis(out, threshold=70)
    m = st.columns(4)
    m[0].metric("Customers scored", f"{k['customers']:,}")
    m[1].metric("Avg. top-1 affinity", f"{k['avg_affinity']:.1f}")
    m[2].metric("Most recommended", k["top_product"])
    m[3].metric("Avg. confidence", f"{k['avg_conf']:.1f}%")

    f = st.columns([2, 2, 2])
    products = sorted(set(analytics.long_recs(out)["Product"]))
    pick = f[0].multiselect("Top-1 product", products)
    min_aff = f[1].slider("Min. top-1 affinity", 0, 100, 0)
    compact = f[2].toggle("Show only ID + recommendations", value=False)

    view = out
    if pick:
        view = view[view["Pred_Target_1"].isin(pick)]
    view = view[view["Pred_Affinity_1"].fillna(0) >= min_aff]
    if compact:
        rec_cols = [c for t in range(1, TOP_N + 1)
                    for c in (f"Pred_Target_{t}", f"Pred_Affinity_{t}", f"Pred_Confidence_{t}")]
        lead = [r["id_col"]] if r["id_col"] else []
        view = view[lead + rec_cols]

    cfg = {}
    for i in range(1, TOP_N + 1):
        cfg[f"Pred_Affinity_{i}"] = st.column_config.ProgressColumn(f"Affinity {i}", min_value=0, max_value=100, format="%.1f")
        cfg[f"Pred_Confidence_{i}"] = st.column_config.ProgressColumn(f"Confidence {i} %", min_value=0, max_value=100, format="%.1f")
        cfg[f"Pred_Target_{i}"] = st.column_config.TextColumn(f"Recommendation {i}")
    st.dataframe(view, hide_index=True, column_config=cfg)
    st.caption(f"Showing {len(view):,} of {len(out):,} rows")

    d = st.columns([1, 1, 3])
    d[0].download_button("⬇️ Results (CSV)", ui.csv_bytes(view), file_name="cross_sell_results.csv", mime="text/csv")
    d[1].download_button(
        "⬇️ Results (Excel)",
        analytics.to_excel_bytes({"Results": view, "Product summary": analytics.summary_table(out)}),
        file_name="cross_sell_results.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    # d[2].info("Open **Insights Dashboard** in the sidebar for charts, segment analysis and campaign lists.")
    d[2].page_link(
        insights_page,
        label="📊 Insights Dashboard"
    )
