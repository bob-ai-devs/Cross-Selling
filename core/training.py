"""In-session model training. Everything lives in memory; nothing is written to disk."""
from __future__ import annotations

import os
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score, mean_absolute_error)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from core.config import TOP_N
from core.preprocessing import (CategoryEncoder, InputError, norm_col,
                                normalize_products, parse_dates, pretty,
                                sort_by_date)


# --------------------------------------------------------------------------- #
# Helpers for the UI
# --------------------------------------------------------------------------- #
def guess_columns(df: pd.DataFrame):
    """Heuristics that pre-fill the column pickers."""
    prod = [c for c in df.columns if "prod" in norm_col(c) and "date" not in norm_col(c)]
    date = [c for c in df.columns if "date" in norm_col(c) or norm_col(c).startswith(("acq", "since"))]
    return prod, date


def guess_id_columns(df: pd.DataFrame, exclude=()):
    """Identifier-like columns: known ID names, or text columns that are (almost) all unique."""
    ids = []
    n = len(df)
    for c in df.columns:
        if c in exclude:
            continue
        name = norm_col(c)
        if name in {"id", "customerid", "custid", "clientid", "accountid", "userid"}:
            ids.append(c)
        elif n > 50 and not pd.api.types.is_numeric_dtype(df[c]) \
                and df[c].nunique(dropna=True) / n > 0.9:
            ids.append(c)
    return ids


# --------------------------------------------------------------------------- #
# Preparation
# --------------------------------------------------------------------------- #
def prepare_training_frame(df: pd.DataFrame, product_cols, date_cols, drop_cols, dayfirst=False):
    if len(product_cols) != len(date_cols) or not product_cols:
        raise InputError("Choose the same number of product and date columns (at least one pair).")
    if len(set(product_cols) | set(date_cols)) != len(product_cols) + len(date_cols):
        raise InputError("A column was selected more than once. Each product/date column must be unique.")
    if len(df) < 30:
        raise InputError("At least 30 customers are needed to train a model.")

    df = df.reset_index(drop=True)
    prods = sort_by_date(normalize_products(df, product_cols),
                         parse_dates(df, date_cols, dayfirst))
    product_enc = CategoryEncoder(sorted(set(prods.reshape(-1).tolist()) | {""}), case_insensitive=True)
    codes = product_enc.transform(prods.reshape(-1)).reshape(prods.shape)
    if len(product_enc.classes) < 3:
        raise InputError("Fewer than two distinct products found — cannot learn cross-sell patterns.")

    skip = set(product_cols) | set(date_cols) | set(drop_cols)
    attr_cols = [c for c in df.columns if c not in skip]
    if not attr_cols:
        raise InputError("No customer attribute columns left after removing products, dates and identifiers.")

    attrs = pd.DataFrame(index=range(len(df)))
    cat_encoders: dict[str, CategoryEncoder] = {}
    medians: dict[str, float] = {}
    for c in attr_cols:
        s = df[c]
        numeric = pd.to_numeric(s, errors="coerce")
        if pd.api.types.is_bool_dtype(s) or pd.api.types.is_numeric_dtype(s) or numeric.notna().mean() > 0.95:
            med = float(numeric.median()) if numeric.notna().any() else 0.0
            medians[c] = med
            attrs[c] = numeric.fillna(med).astype(float).to_numpy()
        else:
            vals = s.astype("string").fillna("").str.strip()
            enc = CategoryEncoder(sorted(set(vals.tolist())), case_insensitive=False)
            cat_encoders[c] = enc
            attrs[c] = enc.transform(vals)
    return dict(attrs=attrs, codes=codes, product_enc=product_enc, cat_encoders=cat_encoders,
                medians=medians, attr_cols=attr_cols)


def find_elbow(ks, wcss):
    """Geometric elbow: the k whose (k, WCSS) point is farthest from the straight line
    joining the first and last points of the curve."""
    if len(ks) < 3:
        return ks[0]
    x = (np.asarray(ks, float) - ks[0]) / (ks[-1] - ks[0])
    y = np.asarray(wcss, float)
    y = (y - y.min()) / max(y.max() - y.min(), 1e-12)
    # distance to the chord from (0, y[0]) to (1, y[-1])
    dist = np.abs((y[-1] - y[0]) * x - (x[-1] - x[0]) * (y - y[0])) / np.hypot(y[-1] - y[0], x[-1] - x[0])
    return int(ks[int(np.argmax(dist))])


def cluster_customers(attrs: pd.DataFrame, codes: np.ndarray, product_enc: CategoryEncoder,
                      k: int | None = None, k_max: int = 10, seed: int = 42):
    n, n_classes = len(codes), len(product_enc.classes)
    hold = np.zeros((n, n_classes))
    hold[np.repeat(np.arange(n), codes.shape[1]), codes.reshape(-1)] = 1
    if product_enc.blank_index is not None:
        hold = np.delete(hold, product_enc.blank_index, axis=1)
    Z = StandardScaler().fit_transform(np.hstack([attrs.to_numpy(dtype=float), hold]))

    ks = list(range(2, max(3, min(k_max, n // 10)) + 1))
    wcss = [KMeans(n_clusters=kk, n_init=3, random_state=seed).fit(Z).inertia_ for kk in ks]
    chosen = k or find_elbow(ks, wcss)
    labels = KMeans(n_clusters=chosen, n_init=10, random_state=seed).fit_predict(Z)
    return labels, chosen, ks, wcss


def make_pseudo_labels(codes: np.ndarray, labels: np.ndarray, product_enc: CategoryEncoder):
    """Per cluster: rank products by popularity; each customer gets the top
    products they don't already hold. Affinity = 90 - 10*rank (min 10)."""
    n, n_classes = len(codes), len(product_enc.classes)
    blank = product_enc.blank_index if product_enc.blank_index is not None else 0
    targets = np.full((n, TOP_N), blank, dtype=int)
    affs = np.zeros((n, TOP_N), dtype=float)
    cluster_tables = []

    for cl in np.unique(labels):
        idx = np.where(labels == cl)[0]
        counts = np.bincount(codes[idx].reshape(-1), minlength=n_classes).astype(float)
        counts[blank] = 0
        ranking = [int(r) for r in np.argsort(-counts, kind="stable") if counts[r] > 0]
        cluster_tables.append({
            "Cluster": int(cl), "Customers": int(len(idx)),
            "Top products": ", ".join(pretty(product_enc.classes[r]) for r in ranking[:3]) or "—",
        })
        for row in idx:
            held = set(codes[row].tolist())
            slot = 0
            for j, r in enumerate(ranking):
                if slot == TOP_N:
                    break
                if r not in held:
                    targets[row, slot] = r
                    affs[row, slot] = max(10.0, 90.0 - 10.0 * j)
                    slot += 1
    return targets, affs, pd.DataFrame(cluster_tables)


# --------------------------------------------------------------------------- #
# Network
# --------------------------------------------------------------------------- #
def build_model(n_features: int, n_classes: int, units=(128, 64, 32), dropout=0.0, lr=1e-3):
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    import tensorflow as tf
    from tensorflow.keras import layers, Model

    inp = layers.Input(shape=(n_features,))
    x = layers.Dense(units[0], activation="relu")(inp)
    if dropout:
        x = layers.Dropout(dropout)(x)
    x = layers.Dense(units[1], activation="relu")(x)
    if dropout:
        x = layers.Dropout(dropout)(x)
    outs = []
    for i in range(1, TOP_N + 1):
        h = layers.Dense(units[2], activation="relu", name=f"h{i}")(x)
        outs.append(layers.Dense(n_classes, activation="softmax", name=f"t{i}")(h))
        outs.append(layers.Dense(1, activation="sigmoid", name=f"a{i}")(h))
    model = Model(inp, outs)
    losses, metrics = {}, {}
    for i in range(1, TOP_N + 1):
        losses[f"t{i}"] = "sparse_categorical_crossentropy"
        losses[f"a{i}"] = "binary_crossentropy"
        metrics[f"t{i}"] = "accuracy"
    model.compile(optimizer=tf.keras.optimizers.Adam(lr), loss=losses, metrics=metrics)
    return model


def make_progress_callback(total_epochs: int, bar, chart_slot, text_slot):
    import tensorflow as tf

    class _Progress(tf.keras.callbacks.Callback):
        def __init__(self):
            super().__init__()
            self.rows, self.best, self.best_epoch = [], float("inf"), 0

        def on_epoch_end(self, epoch, logs=None):
            logs = logs or {}
            loss, val = float(logs.get("loss", 0)), float(logs.get("val_loss", 0))
            self.rows.append({"epoch": epoch + 1, "loss": loss, "val_loss": val})
            if val < self.best:
                self.best, self.best_epoch = val, epoch + 1
            bar.progress(min((epoch + 1) / total_epochs, 1.0),
                         text=f"Epoch {epoch + 1}/{total_epochs}")
            text_slot.markdown(
                f"**loss** `{loss:.4f}` · **val_loss** `{val:.4f}` · "
                f"**best val_loss** `{self.best:.4f}` at epoch {self.best_epoch}")
            chart_slot.line_chart(pd.DataFrame(self.rows).set_index("epoch")[["loss", "val_loss"]])

    return _Progress()


def train_bundle(df, product_cols, date_cols, drop_cols, params: dict,
                 bar, chart_slot, text_slot, dayfirst=False):
    """Full pipeline. Returns (bundle, report) — both kept only in session memory."""
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    import tensorflow as tf
    from tensorflow.keras.callbacks import EarlyStopping

    seed = int(params.get("seed", 42))
    tf.keras.utils.set_random_seed(seed)

    prep = prepare_training_frame(df, product_cols, date_cols, drop_cols, dayfirst)
    pe: CategoryEncoder = prep["product_enc"]
    codes, attrs = prep["codes"], prep["attrs"]

    labels, k, ks, wcss = cluster_customers(attrs, codes, pe, params.get("clusters") or None, seed=seed)
    targets, affs, cluster_table = make_pseudo_labels(codes, labels, pe)

    feat = attrs.copy()
    for j in range(codes.shape[1]):
        feat[f"{j + 1}_product"] = codes[:, j]
    model_cols = list(feat.columns)

    idx = np.arange(len(feat))
    tr_all, te = train_test_split(idx, test_size=params["test_size"], random_state=seed)
    tr, va = train_test_split(tr_all, test_size=params["val_size"], random_state=seed)

    scaler = MinMaxScaler().fit(feat.iloc[tr].to_numpy(dtype=float))
    X = scaler.transform(feat.to_numpy(dtype=float))

    def ys(rows):
        d = {}
        for i in range(TOP_N):
            d[f"t{i + 1}"] = targets[rows, i].astype("int32")
            d[f"a{i + 1}"] = (affs[rows, i] / 100.0).astype("float32").reshape(-1, 1)
        return d

    model = build_model(X.shape[1], len(pe.classes), params["units"], params["dropout"], params["lr"])
    cb = make_progress_callback(params["epochs"], bar, chart_slot, text_slot)
    stop = EarlyStopping(monitor="val_loss", patience=params["patience"],
                         restore_best_weights=True, verbose=0)
    hist = model.fit(X[tr], ys(tr), validation_data=(X[va], ys(va)), epochs=params["epochs"],
                     batch_size=params["batch_size"], callbacks=[cb, stop], verbose=0)

    # ---------------- evaluation ----------------
    def evaluate(rows):
        p = model.predict(X[rows], batch_size=2048, verbose=0)
        res = {}
        for i in range(TOP_N):
            probs, aff = p[2 * i], p[2 * i + 1].reshape(-1) * 100
            y, yhat = targets[rows, i], probs.argmax(1)
            top3 = np.argsort(-probs, axis=1)[:, :3]
            res[f"T{i + 1}"] = dict(
                accuracy=accuracy_score(y, yhat),
                f1=f1_score(y, yhat, average="weighted", zero_division=0),
                top3=float(np.mean([y[r] in top3[r] for r in range(len(y))])),
                baseline=float(np.max(np.bincount(y)) / len(y)),
                aff_mae=float(mean_absolute_error(affs[rows, i], aff)),
            )
        return res, p

    m_train, _ = evaluate(tr)
    m_test, p_test = evaluate(te)

    y1, yhat1 = targets[te, 0], p_test[0].argmax(1)
    present = sorted(set(y1.tolist()) | set(yhat1.tolist()))
    names = [pretty(pe.classes[c]) for c in present]
    cls_report = pd.DataFrame(classification_report(
        y1, yhat1, labels=present, target_names=names, output_dict=True, zero_division=0)).T
    cm = pd.DataFrame(confusion_matrix(y1, yhat1, labels=present), index=names, columns=names)

    history = pd.DataFrame(hist.history)
    history.insert(0, "epoch", np.arange(1, len(history) + 1))

    bundle = dict(
        model=model, scaler=scaler, product_enc=pe, cat_encoders=prep["cat_encoders"],
        medians=prep["medians"], attr_cols=prep["attr_cols"], model_cols=model_cols,
        product_cols=list(product_cols), date_cols=list(date_cols), drop_cols=list(drop_cols),
        dayfirst=dayfirst, trained_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        n_customers=len(df), n_clusters=k,
    )
    report = dict(
        metrics_train=m_train, metrics_test=m_test, class_report=cls_report, confusion=cm,
        history=history, cluster_table=cluster_table, elbow=pd.DataFrame({"k": ks, "wcss": wcss}),
        chosen_k=k, best_epoch=cb.best_epoch, epochs_run=len(history), params=params,
        n_rows=len(df), split=dict(train=len(tr), val=len(va), test=len(te)),
        target_dist=pd.Series([pretty(pe.classes[c]) for c in targets[:, 0]]).value_counts().rename_axis("Product").reset_index(name="Customers"),
        trained_at=bundle["trained_at"],
    )
    return bundle, report


def report_text(report: dict) -> str:
    """Plain-text training summary (replacement for the old train_output.txt)."""
    lines = ["Model training completed successfully.", "-" * 100,
             f"Trained at: {report['trained_at']}   Rows: {report['n_rows']}   Clusters: {report['chosen_k']}",
             f"Epochs run: {report['epochs_run']} (best epoch {report['best_epoch']})",
             f"Split (train/val/test): {report['split']['train']}/{report['split']['val']}/{report['split']['test']}", ""]
    for name, m in (("Training", report["metrics_train"]), ("Test", report["metrics_test"])):
        for t, v in m.items():
            lines.append(f"Target {t[-1]} - {name} F1-Score: {v['f1'] * 100:.2f} %   "
                         f"Accuracy: {v['accuracy'] * 100:.2f} %   Top-3: {v['top3'] * 100:.2f} %   "
                         f"Affinity MAE: {v['aff_mae']:.2f}")
        lines.append("")
    lines.append("Note: scores measure agreement with cluster-derived recommendation labels, "
                 "not real customer take-up.")
    return "\n".join(lines)
