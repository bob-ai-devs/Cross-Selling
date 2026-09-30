import pandas as pd
import streamlit as st

from core import analytics, ui
from core.config import TOP_N


def render():
    ui.hero("Insights Dashboard", "Explore the latest batch results: product mix, segments, gaps and target lists.")
    r = st.session_state.get("results")
    if not r:
        st.info("No results yet. Run a prediction on the **Batch Prediction** page "
                "(the demo-data option works if you just want to look around).")
        return

    df, pcols = r["df"], r["product_cols"]
    st.caption(f"{r['source']} · {len(df):,} customers · scored at {r['time']}")

    thr = st.slider("High-affinity threshold", 0, 100, 70, help="Used for the 'high-affinity' KPIs and histogram marker.")
    k = analytics.kpis(df, thr)
    m = st.columns(5)
    m[0].metric("Customers", f"{k['customers']:,}")
    m[1].metric("Avg. top-1 affinity", f"{k['avg_affinity']:.1f}")
    m[2].metric("High-affinity customers", f"{k['high_count']:,}", delta=f"{k['high_share'] * 100:.1f}% of base", delta_color="off")
    m[3].metric("Most recommended", k["top_product"])
    m[4].metric("Coverage", f"{k['coverage'] * 100:.0f}%", help="Customers who receive at least one recommendation.")

    tabs = st.tabs(["Product mix", "Affinity", "Segments", "Gap & co-holding", "Customer 360", "Campaign lists"])

    with tabs[0]:
        c1, c2 = st.columns([3, 2])
        c1.plotly_chart(analytics.fig_product_mix(df), key="ins_mix")
        c2.markdown("**Product summary**")
        c2.dataframe(analytics.summary_table(df), hide_index=True)

    with tabs[1]:
        c1, c2 = st.columns(2)
        c1.markdown("**Top-1 affinity distribution**")
        c1.plotly_chart(analytics.fig_affinity_hist(df, thr), key="ins_hist")
        c2.markdown("**Affinity by recommended product (all ranks)**")
        c2.plotly_chart(analytics.fig_affinity_box(df), key="ins_box")

    with tabs[2]:
        skip = {c for c in df.columns if c.startswith("Pred_")} | set(pcols) | ({r["id_col"]} if r["id_col"] else set())
        seg_options = [c for c in df.columns if c not in skip and df[c].nunique() > 1]
        if not seg_options:
            st.info("No usable segment columns in the data.")
        else:
            a, b = st.columns(2)
            seg = a.selectbox("Segment by", seg_options)
            metric = b.radio("Metric", ["Share of top-1 picks (%)", "Average affinity"], horizontal=True)
            st.plotly_chart(analytics.fig_segment_heatmap(df, seg, metric), key="ins_seg")
            st.caption("Numeric columns with many values are grouped into quintiles.")

    with tabs[3]:
        gap = analytics.gap_table(df, pcols)
        c1, c2 = st.columns([3, 2])
        c1.markdown("**Held today vs. recommended**")
        c1.plotly_chart(analytics.fig_gap(gap), key="ins_gap")
        c2.dataframe(gap, hide_index=True)
        st.markdown("**Products customers tend to hold together** (lift > 1 = held together more often than chance)")
        fig = analytics.fig_cooccurrence(df, pcols)
        if fig is None:
            st.info("Not enough distinct products to compute co-holding.")
        else:
            st.plotly_chart(fig, key="ins_cooc")


    with tabs[4]:
        id_col = r["id_col"]
        label = id_col or "row number"
    
        ids = (
            df[id_col].astype(str)
            if id_col
            else pd.Series((pd.RangeIndex(len(df)) + 1).astype(str))
        )
    
        # Customer selection dropdown
        customer_options = ids.astype(str).str.strip().tolist()
    
        selected_id = st.selectbox(
            f"Select a customer by {label}",
            options=["Select a customer"] + customer_options,
            index=0,
            key="selected_customer"
        )
    
        if selected_id != "Select a customer":
            hit = df[ids.str.strip() == selected_id.strip()]
    
            if hit.empty:
                st.warning("No customer with that identifier.")
            else:
                row = hit.iloc[0]
    
                l, rgt = st.columns([3, 2])
    
                profile = row[
                    [c for c in df.columns if not c.startswith("Pred_")]
                ]
    
                l.markdown("**Profile and holdings**")
                l.dataframe(
                    profile.astype(str).rename("Value").to_frame()
                )
    
                rgt.markdown("**Recommendations**")
    
                for i in range(1, TOP_N + 1):
                    prod = row[f"Pred_Target_{i}"]
                    aff = row[f"Pred_Affinity_{i}"]
    
                    rgt.markdown(f"**{i}. {prod}**")
    
                    if prod != "—" and pd.notna(aff):
                        rgt.progress(
                            min(max(float(aff) / 100, 0.0), 1.0),
                            text=(
                                f"Affinity {aff:.1f} · "
                                f"confidence "
                                f"{row[f'Pred_Confidence_{i}']:.1f}%"
                            )
                        )

    with tabs[5]:
        st.markdown("Build a ranked outreach list — highest **priority score** (affinity × confidence) first.")
        products = sorted(set(analytics.long_recs(df)["Product"]))
        a, b, c = st.columns([2, 2, 2])
        pick = a.multiselect("Products", products, default=products[:1])
        min_aff = b.slider("Minimum affinity", 0, 100, thr, key="camp_aff")
        ranks = c.multiselect("Ranks to include", list(range(1, TOP_N + 1)), default=[1])
        lst = analytics.campaign_list(df, r["id_col"], pick, min_aff, ranks or [1])
        st.write(f"**{len(lst):,}** customer–product pairs match.")
        st.dataframe(lst, hide_index=True, column_config={
            "Affinity": st.column_config.ProgressColumn(min_value=0, max_value=100, format="%.1f"),
            "Confidence %": st.column_config.ProgressColumn(min_value=0, max_value=100, format="%.1f"),
        })
        d = st.columns([1, 1, 3])
        d[0].download_button("⬇️ List (CSV)", ui.csv_bytes(lst), file_name="campaign_list.csv", mime="text/csv")
        d[1].download_button(
            "⬇️ Insights pack (Excel)",
            analytics.to_excel_bytes({"Campaign list": lst, "Product summary": analytics.summary_table(df),
                                      "Gap analysis": gap, "All results": df}),
            file_name="cross_sell_insights.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
