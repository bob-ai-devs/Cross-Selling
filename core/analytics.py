"""Aggregations and Plotly figures for the Insights dashboard (all in memory)."""
from __future__ import annotations

from io import BytesIO

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from core.config import TOP_N
from core.preprocessing import normalize_products, pretty

PALETTE = px.colors.qualitative.Set2
NONE_LABEL = "—"


# --------------------------------------------------------------------------- #
# Reshaping
# --------------------------------------------------------------------------- #
def long_recs(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (customer, rank)."""
    frames = []
    for r in range(1, TOP_N + 1):
        frames.append(pd.DataFrame({
            "row": np.arange(len(df)),
            "Rank": r,
            "Product": df[f"Pred_Target_{r}"].to_numpy(),
            "Affinity": df[f"Pred_Affinity_{r}"].to_numpy(),
            "Confidence": df[f"Pred_Confidence_{r}"].to_numpy(),
        }))
    out = pd.concat(frames, ignore_index=True)
    return out[out["Product"].notna() & (out["Product"] != NONE_LABEL)]


def held_products(df: pd.DataFrame, product_cols: list[str]) -> pd.Series:
    flat = normalize_products(df, product_cols).reshape(-1)
    flat = flat[flat != ""]
    return pd.Series([pretty(x) for x in flat], dtype="object")


def kpis(df: pd.DataFrame, threshold: float) -> dict:
    aff1 = df["Pred_Affinity_1"].dropna()
    top = df.loc[df["Pred_Target_1"] != NONE_LABEL, "Pred_Target_1"]
    return {
        "customers": len(df),
        "avg_affinity": float(aff1.mean()) if len(aff1) else 0.0,
        "high_share": float((aff1 >= threshold).sum() / max(len(df), 1)),
        "high_count": int((aff1 >= threshold).sum()),
        "top_product": top.mode().iat[0] if len(top) else "—",
        "coverage": float((df["Pred_Target_1"] != NONE_LABEL).mean()),
        "avg_conf": float(df["Pred_Confidence_1"].dropna().mean()) if df["Pred_Confidence_1"].notna().any() else 0.0,
    }


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def fig_product_mix(df: pd.DataFrame) -> go.Figure:
    lr = long_recs(df)
    t = lr.groupby(["Product", "Rank"]).size().reset_index(name="Customers")
    t["Rank"] = "Rank " + t["Rank"].astype(str)
    order = lr[lr.Rank == 1]["Product"].value_counts().index.tolist()
    fig = px.bar(t, x="Product", y="Customers", color="Rank", barmode="stack",
                 category_orders={"Product": order}, color_discrete_sequence=PALETTE)
    fig.update_layout(margin=dict(t=30, b=10), legend_title_text="")
    return fig


def fig_affinity_hist(df: pd.DataFrame, threshold: float) -> go.Figure:
    fig = px.histogram(df, x="Pred_Affinity_1", nbins=30, color_discrete_sequence=[PALETTE[0]],
                       labels={"Pred_Affinity_1": "Top-1 affinity"})
    fig.add_vline(x=threshold, line_dash="dash", line_color="crimson",
                  annotation_text=f"threshold {threshold:g}")
    fig.update_layout(margin=dict(t=30, b=10), yaxis_title="Customers")
    return fig


def fig_affinity_box(df: pd.DataFrame) -> go.Figure:
    lr = long_recs(df)
    order = lr.groupby("Product")["Affinity"].median().sort_values(ascending=False).index.tolist()
    fig = px.box(lr, x="Product", y="Affinity", color="Product", category_orders={"Product": order},
                 color_discrete_sequence=PALETTE, points=False)
    fig.update_layout(showlegend=False, margin=dict(t=30, b=10))
    return fig


def gap_table(df: pd.DataFrame, product_cols: list[str]) -> pd.DataFrame:
    n = max(len(df), 1)
    # customers holding each product (a customer counted once per product)
    arr = normalize_products(df, product_cols)
    hold_share = {}
    for p in sorted(set(arr.reshape(-1)) - {""}):
        hold_share[pretty(p)] = float((arr == p).any(axis=1).mean())
    lr = long_recs(df)
    rec_share = (lr.groupby("Product")["row"].nunique() / n).to_dict()
    names = sorted(set(hold_share) | set(rec_share))
    out = pd.DataFrame({
        "Product": names,
        "Currently held (% customers)": [round(100 * hold_share.get(p, 0), 1) for p in names],
        "Recommended in top-3 (% customers)": [round(100 * rec_share.get(p, 0), 1) for p in names],
    })
    return out.sort_values("Recommended in top-3 (% customers)", ascending=False).reset_index(drop=True)


def fig_gap(gap: pd.DataFrame) -> go.Figure:
    m = gap.melt(id_vars="Product", value_vars=["Currently held (% customers)",
                                                "Recommended in top-3 (% customers)"],
                 var_name="Metric", value_name="% of customers")
    fig = px.bar(m, x="Product", y="% of customers", color="Metric", barmode="group",
                 color_discrete_sequence=[PALETTE[2], PALETTE[1]])
    fig.update_layout(margin=dict(t=30, b=10), legend_title_text="", legend=dict(orientation="h", y=1.12))
    return fig


def fig_cooccurrence(df: pd.DataFrame, product_cols: list[str]) -> go.Figure | None:
    arr = normalize_products(df, product_cols)
    prods = sorted(set(arr.reshape(-1)) - {""})
    if len(prods) < 2:
        return None
    H = np.column_stack([(arr == p).any(axis=1) for p in prods]).astype(float)
    n = len(H)
    support = H.mean(axis=0)
    pair = (H.T @ H) / n
    with np.errstate(divide="ignore", invalid="ignore"):
        lift = pair / np.outer(support, support)
    np.fill_diagonal(lift, np.nan)
    labels = [pretty(p) for p in prods]
    fig = go.Figure(go.Heatmap(z=lift, x=labels, y=labels, colorscale="RdBu", zmid=1,
                               colorbar_title="Lift", hovertemplate="%{y} + %{x}<br>lift %{z:.2f}<extra></extra>"))
    fig.update_layout(margin=dict(t=30, b=10), height=480)
    return fig


def segment_series(df: pd.DataFrame, col: str, bins: int = 5) -> pd.Series:
    s = df[col]
    if pd.api.types.is_numeric_dtype(s) and s.nunique() > 8:
        try:
            return pd.qcut(s, q=bins, duplicates="drop").astype(str)
        except ValueError:
            return s.astype(str)
    return s.astype(str)


def fig_segment_heatmap(df: pd.DataFrame, seg_col: str, metric: str) -> go.Figure:
    seg = segment_series(df, seg_col)
    t = pd.DataFrame({"Segment": seg, "Product": df["Pred_Target_1"], "Affinity": df["Pred_Affinity_1"]})
    t = t[t["Product"] != NONE_LABEL]
    if metric == "Share of top-1 picks (%)":
        pv = (t.groupby(["Segment", "Product"]).size().unstack(fill_value=0))
        pv = pv.div(pv.sum(axis=1), axis=0) * 100
        fmt = ".0f"
    else:
        pv = t.pivot_table(index="Segment", columns="Product", values="Affinity", aggfunc="mean")
        fmt = ".1f"
    fig = px.imshow(pv.round(1), text_auto=fmt, aspect="auto", color_continuous_scale="YlOrRd",
                    labels=dict(color=metric))
    fig.update_layout(margin=dict(t=30, b=10))
    return fig


def fig_probability_bars(probs_mean: np.ndarray, labels: list[str], held: set[str]) -> go.Figure:
    order = np.argsort(probs_mean)
    colors = ["#B0B7C3" if labels[i] in held else PALETTE[0] for i in order]
    fig = go.Figure(go.Bar(x=probs_mean[order] * 100, y=[labels[i] for i in order],
                           orientation="h", marker_color=colors,
                           hovertemplate="%{y}: %{x:.1f}%<extra></extra>"))
    fig.update_layout(margin=dict(t=10, b=10, l=10, r=10), height=340,
                      xaxis_title="Model interest (avg. of the 3 ranking heads, %)")
    return fig


def fig_history(history: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_scatter(x=history["epoch"], y=history["loss"], name="loss")
    fig.add_scatter(x=history["epoch"], y=history["val_loss"], name="val_loss")
    fig.update_layout(margin=dict(t=30, b=10), xaxis_title="Epoch", yaxis_title="Loss",
                      legend=dict(orientation="h", y=1.12))
    return fig


def fig_confusion(cm: pd.DataFrame) -> go.Figure:
    fig = px.imshow(cm, text_auto=True, aspect="auto", color_continuous_scale="Blues",
                    labels=dict(x="Predicted", y="Actual (cluster label)", color="Customers"))
    fig.update_layout(margin=dict(t=30, b=10))
    return fig


def fig_elbow(elbow: pd.DataFrame, chosen: int) -> go.Figure:
    fig = px.line(elbow, x="k", y="wcss", markers=True)
    fig.add_vline(x=chosen, line_dash="dash", line_color="crimson", annotation_text=f"k = {chosen}")
    fig.update_layout(margin=dict(t=30, b=10), yaxis_title="Within-cluster sum of squares")
    return fig


# --------------------------------------------------------------------------- #
# Export helpers
# --------------------------------------------------------------------------- #
def campaign_list(df: pd.DataFrame, id_col: str | None, products: list[str] | None,
                  min_aff: float, ranks: list[int]) -> pd.DataFrame:
    lr = long_recs(df)
    lr = lr[lr["Rank"].isin(ranks) & (lr["Affinity"] >= min_aff)]
    if products:
        lr = lr[lr["Product"].isin(products)]
    ids = df[id_col].to_numpy() if id_col else np.arange(len(df)) + 1
    out = pd.DataFrame({
        "Customer": ids[lr["row"].to_numpy()],
        "Rank": lr["Rank"].to_numpy(),
        "Recommended product": lr["Product"].to_numpy(),
        "Affinity": lr["Affinity"].to_numpy(),
        "Confidence %": lr["Confidence"].to_numpy(),
    })
    out["Priority score"] = (out["Affinity"] * out["Confidence %"] / 100).round(1)
    return out.sort_values("Priority score", ascending=False).reset_index(drop=True)


def to_excel_bytes(sheets: dict[str, pd.DataFrame]) -> bytes:
    buf = BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as xl:
        for name, frame in sheets.items():
            frame.to_excel(xl, sheet_name=name[:31], index=False)
    return buf.getvalue()


def summary_table(df: pd.DataFrame) -> pd.DataFrame:
    """Per product: how often it is picked at each rank and its mean affinity."""
    lr = long_recs(df)
    counts = lr.pivot_table(index="Product", columns="Rank", values="row", aggfunc="count", fill_value=0)
    counts.columns = [f"Rank {c} picks" for c in counts.columns]
    out = counts.join(lr.groupby("Product")["Affinity"].mean().round(2).rename("Avg affinity"))
    out["Avg confidence %"] = lr.groupby("Product")["Confidence"].mean().round(1)
    return out.sort_values(counts.columns[0], ascending=False).reset_index()
