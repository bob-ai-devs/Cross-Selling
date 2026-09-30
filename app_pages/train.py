import pandas as pd
import streamlit as st

from core import analytics, training, ui
from core.config import TOP_N
from core.demo import make_history_demo
from core.preprocessing import InputError, profile_frame

from app_pages.batch import read_table


def render():
    ui.hero("Train a Custom Model",
            "Upload historical holdings (product + acquisition date per slot) and train a network "
            "on your own data. The model exists only in this browser session.")
    ss = st.session_state

    # ---- 1 data --------------------------------------------------------------
    st.subheader("1 · Historical data")
    src = st.radio("Data source", ["Upload file", "Use demo data"], horizontal=True, label_visibility="collapsed")
    df = None
    if src == "Upload file":
        file = st.file_uploader("CSV or Excel with one row per customer", type=["csv", "xlsx", "xls"])
        if file is not None:
            try:
                df = read_table(file)
            except Exception as e:
                st.error(f"Could not read the file: {e}")
    else:
        n = st.slider("Demo customers", 300, 5000, 1500, step=100)
        df = make_history_demo(n, seed=11)
        st.caption("Synthetic history generated in memory (columns Product_1…5 / Acquired_1…5).")

    if df is None or df.empty:
        st.info("Provide historical data to continue.")
        show_report()
        return

    st.success(f"{len(df):,} customers × {df.shape[1]} columns")
    with st.expander("Preview and data quality"):
        st.dataframe(df.head(50), hide_index=True)
        st.dataframe(profile_frame(df), hide_index=True)

    # ---- 2 columns -----------------------------------------------------------
    st.subheader("2 · Column set-up")
    sig = abs(hash(tuple(df.columns))) % 10**8
    g_prod, g_date = training.guess_columns(df)
    n_pairs = st.number_input("Number of product / date pairs", 1, 10, max(1, min(len(g_prod), 10)) if g_prod else 5)
    cols = list(df.columns)
    product_cols, date_cols = [], []
    for i in range(int(n_pairs)):
        c1, c2 = st.columns(2)
        dp = g_prod[i] if i < len(g_prod) else cols[0]
        dd = g_date[i] if i < len(g_date) else cols[0]
        product_cols.append(c1.selectbox(f"Product column {i + 1}", cols, index=cols.index(dp), key=f"tr_p{i}_{sig}"))
        date_cols.append(c2.selectbox(f"Acquired-date column {i + 1}", cols, index=cols.index(dd), key=f"tr_d{i}_{sig}"))

    chosen = set(product_cols) | set(date_cols)
    id_default = training.guess_id_columns(df, exclude=chosen)
    drop_cols = st.multiselect("Identifier / unwanted columns to exclude (e.g. Customer ID, name, phone)",
                               [c for c in cols if c not in chosen], default=id_default, key=f"tr_drop_{sig}")
    dayfirst = st.checkbox("Dates are day-first (DD/MM/YYYY)", value=False)

    # ---- 3 parameters ----------------------------------------------------------
    with st.expander("Training parameters", expanded=False):
        a, b, c = st.columns(3)
        epochs = a.number_input("Max epochs", 5, 500, 100, step=5)
        patience = b.number_input("Early-stopping patience", 2, 50, 15)
        batch_size = c.selectbox("Batch size", [16, 32, 64, 128, 256], index=1)
        a, b, c = st.columns(3)
        lr = a.select_slider("Learning rate", [1e-4, 3e-4, 1e-3, 3e-3, 1e-2], value=1e-3, format_func=lambda x: f"{x:g}")
        dropout = b.slider("Dropout", 0.0, 0.5, 0.1, 0.05)
        clusters = c.number_input("Customer clusters (0 = automatic elbow)", 0, 12, 0)
        a, b, c = st.columns(3)
        test_size = a.slider("Test share", 0.1, 0.4, 0.2, 0.05)
        val_size = b.slider("Validation share (of the rest)", 0.1, 0.4, 0.2, 0.05)
        seed = c.number_input("Random seed", 0, 9999, 42)
        units = (128, 64, 32)

    params = dict(epochs=int(epochs), patience=int(patience), batch_size=int(batch_size), lr=float(lr),
                  dropout=float(dropout), clusters=int(clusters), test_size=float(test_size),
                  val_size=float(val_size), seed=int(seed), units=units)

    # ---- 4 train ---------------------------------------------------------------
    st.subheader("3 · Train")
    if st.button("🧪 Train model", type="primary"):
        with st.status("Training in progress…", expanded=True) as status:
            bar = st.progress(0.0, text="Preparing data and clustering customers…")
            text_slot = st.empty()
            chart_slot = st.empty()
            try:
                bundle, report = training.train_bundle(df, product_cols, date_cols, drop_cols, params,
                                                       bar, chart_slot, text_slot, dayfirst)
            except InputError as e:
                status.update(label="Training stopped", state="error")
                for m in e.messages:
                    st.error(m)
                show_report()
                return
            except Exception as e:
                status.update(label="Training failed", state="error")
                st.error(f"Training failed: {e}")
                show_report()
                return
            status.update(label="Training complete", state="complete", expanded=False)
        ss["custom_bundle"], ss["train_report"] = bundle, report
        st.toast("Custom model ready — use it in Batch Prediction", icon="✅")

    show_report()


def show_report():
    rep = st.session_state.get("train_report")
    if not rep:
        return
    st.divider()
    st.subheader("Training results")
    mt = rep["metrics_test"]["T1"]
    m = st.columns(5)
    m[0].metric("Test F1 (rank 1)", f"{mt['f1'] * 100:.1f}%")
    m[1].metric("Test accuracy", f"{mt['accuracy'] * 100:.1f}%", delta=f"{(mt['accuracy'] - mt['baseline']) * 100:+.1f} pp vs baseline")
    m[2].metric("Top-3 hit rate", f"{mt['top3'] * 100:.1f}%")
    m[3].metric("Affinity MAE", f"{mt['aff_mae']:.2f}")
    m[4].metric("Epochs (best)", f"{rep['epochs_run']} ({rep['best_epoch']})")
    st.caption("Scores measure how well the network reproduces the cluster-derived recommendation labels "
               "(the same labelling logic as the original app) — not real customer take-up. "
               "'Baseline' is the accuracy of always predicting the most common label.")

    rows = []
    for split, met in (("Train", rep["metrics_train"]), ("Test", rep["metrics_test"])):
        for t, v in met.items():
            rows.append({"Split": split, "Rank": t[-1], "F1 %": round(v["f1"] * 100, 2),
                         "Accuracy %": round(v["accuracy"] * 100, 2), "Top-3 %": round(v["top3"] * 100, 2),
                         "Baseline %": round(v["baseline"] * 100, 2), "Affinity MAE": round(v["aff_mae"], 3)})
    metrics_df = pd.DataFrame(rows)

    t1, t2, t3, t4 = st.tabs(["Metrics", "Learning curve", "Confusion matrix", "Clusters & labels"])
    with t1:
        st.dataframe(metrics_df, hide_index=True)
        st.markdown("**Per-product report (rank 1, test set)**")
        st.dataframe(rep["class_report"].round(3))
    with t2:
        st.plotly_chart(analytics.fig_history(rep["history"]), key="train_hist")
    with t3:
        st.plotly_chart(analytics.fig_confusion(rep["confusion"]), key="train_cm")
    with t4:
        c1, c2 = st.columns(2)
        c1.markdown(f"**Elbow curve** (chosen k = {rep['chosen_k']})")
        c1.plotly_chart(analytics.fig_elbow(rep["elbow"], rep["chosen_k"]), key="train_elbow")
        c2.markdown("**Cluster profile**")
        c2.dataframe(rep["cluster_table"], hide_index=True)
        c2.markdown("**Rank-1 label distribution**")
        c2.dataframe(rep["target_dist"], hide_index=True)

    d = st.columns([1, 1, 1, 2])
    d[0].download_button("⬇️ Training report (TXT)", training.report_text(rep).encode("utf-8"),
                         file_name="train_output.txt", mime="text/plain")
    d[1].download_button("⬇️ Metrics (CSV)", ui.csv_bytes(metrics_df), file_name="train_metrics.csv", mime="text/csv")
    d[2].download_button("⬇️ Learning curve (CSV)", ui.csv_bytes(rep["history"]),
                         file_name="learning_curve.csv", mime="text/csv")
    d[3].success("Custom model is available in **Batch Prediction** for this session.")
