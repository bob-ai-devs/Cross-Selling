"""Prediction logic for the pre-trained and the session-trained models."""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from core.config import (ENCODED_COLS, FEATURE_COLS, ID_COL, MODEL_INPUT_COLS,
                         NUMERIC_COLS, PRODUCT_COLS, TOP_N)
from core.preprocessing import (CategoryEncoder, InputError, normalize_products,
                                parse_dates, pretty, sort_by_date)

warnings.filterwarnings("ignore", message="X does not have valid feature names")

REC_COLS = [(f"Pred_Target_{i}", f"Pred_Affinity_{i}", f"Pred_Confidence_{i}")
            for i in range(1, TOP_N + 1)]

_YES_NO = {"yes": "1", "y": "1", "true": "1", "no": "0", "n": "0", "false": "0"}


@dataclass
class PredictionResult:
    recs: pd.DataFrame                      # Pred_* columns, aligned to input rows
    preds: list                             # raw model outputs [t1,a1,t2,a2,t3,a3]
    product_enc: CategoryEncoder
    held_codes: np.ndarray                  # (n, k) encoded holdings
    warnings: list[str] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# Decoding (shared by both models)
# --------------------------------------------------------------------------- #
def decode_predictions(preds, product_enc: CategoryEncoder, aff_inverse,
                       held_codes: np.ndarray, apply_rules: bool) -> pd.DataFrame:
    """Turn the 6 network outputs into Pred_Target/Affinity/Confidence columns.

    apply_rules=True  -> never recommend blank or already-held products and never
                         repeat a product across ranks 1-3.
    apply_rules=False -> plain arg-max per head (identical to the original app).
    """
    probs = [np.asarray(preds[i]) for i in (0, 2, 4)]
    affs = [np.asarray(preds[i]).reshape(-1) for i in (1, 3, 5)]
    n, n_classes = probs[0].shape
    rows = np.arange(n)

    mask = np.zeros((n, n_classes), dtype=bool)
    if apply_rules:
        blank = product_enc.blank_index
        if blank is not None:
            mask[:, blank] = True
        r = np.repeat(rows, held_codes.shape[1])
        c = held_codes.reshape(-1)
        ok = (c >= 0) & (c < n_classes)
        mask[r[ok], c[ok]] = True

    out = {}
    for k, (p, a) in enumerate(zip(probs, affs)):
        t_col, a_col, c_col = REC_COLS[k]
        if apply_rules:
            masked = np.where(mask, -1.0, p)
            idx = masked.argmax(axis=1)
            none = masked.max(axis=1) < 0
            mask[rows, idx] = True
        else:
            idx = p.argmax(axis=1)
            none = np.zeros(n, dtype=bool)
        labels = np.array([pretty(x) for x in product_enc.inverse(idx)], dtype=object)
        aff = np.clip(np.asarray(aff_inverse(a), dtype=float), 0, 100)
        conf = p[rows, idx].astype(float) * 100
        labels[none] = "—"
        aff[none] = np.nan
        conf[none] = np.nan
        out[t_col] = labels
        out[a_col] = np.round(aff, 2)
        out[c_col] = np.round(conf, 1)
    return pd.DataFrame(out)


# --------------------------------------------------------------------------- #
# Pre-trained model
# --------------------------------------------------------------------------- #
def prepare_pretrained_frame(df: pd.DataFrame, colmap: dict, enc: dict):
    """Encode/scale-ready frame in the exact input order of the network."""
    n = len(df)
    frame = pd.DataFrame(index=range(n))
    problems: list[str] = []
    notes: list[str] = []

    for col in NUMERIC_COLS:
        v = df[colmap[col]].reset_index(drop=True)
        if col == "Past_Loan_Defaults":
            v = v.astype("string").str.strip().str.lower().replace(_YES_NO)
        v = pd.to_numeric(v, errors="coerce")
        bad = int(v.isna().sum())
        if bad:
            fill = float(v.median()) if v.notna().any() else 0.0
            notes.append(f"{col}: {bad} missing/non-numeric value(s) replaced with the median ({fill:g}).")
            v = v.fillna(fill)
        frame[col] = v.astype(float).to_numpy()

    for col, key in ENCODED_COLS.items():
        ce = CategoryEncoder.from_label_encoder(enc[key])
        raw = df[colmap[col]].reset_index(drop=True)
        codes = ce.transform(raw)
        if (codes < 0).any():
            problems.append(f"{col}: unknown value(s) {ce.unseen(raw)[:6]} — allowed: {ce.classes}")
        frame[col] = codes

    prods = normalize_products(df.reset_index(drop=True), [colmap[c] for c in PRODUCT_COLS])
    pe = CategoryEncoder.from_label_encoder(enc["le_pro"])
    codes = pe.transform(prods.reshape(-1)).reshape(prods.shape)
    if (codes < 0).any():
        problems.append(f"Product columns: unknown product(s) {pe.unseen(prods.reshape(-1))[:6]} "
                        f"— allowed: {[c or '(blank)' for c in pe.classes]}")
    for j, c in enumerate(PRODUCT_COLS):
        frame[c] = codes[:, j]

    if problems:
        raise InputError(problems)
    return frame[MODEL_INPUT_COLS], codes, notes


def predict_pretrained(df: pd.DataFrame, colmap: dict, artifacts: dict,
                       apply_rules: bool = True) -> PredictionResult:
    enc = artifacts["encoders"]
    frame, held_codes, notes = prepare_pretrained_frame(df, colmap, enc)

    scaler = enc["scaler"]
    names = getattr(scaler, "feature_names_in_", None)
    if names is not None and set(names) == set(frame.columns):
        frame = frame[list(names)]
    X = scaler.transform(frame.to_numpy(dtype=float))

    preds = artifacts["model"].predict(X, batch_size=2048, verbose=0)
    aff_sc = enc["aff_sc"]
    inv = lambda a: aff_sc.inverse_transform(np.asarray(a).reshape(-1, 1)).reshape(-1)
    pe = CategoryEncoder.from_label_encoder(enc["le_pro"])
    recs = decode_predictions(preds, pe, inv, held_codes, apply_rules)
    return PredictionResult(recs, preds, pe, held_codes, notes)


def predict_single(values: dict, artifacts: dict, apply_rules: bool = True) -> PredictionResult:
    """values: dict keyed by FEATURE_COLS + PRODUCT_COLS (+ optional CustomerID)."""
    row = {c: values.get(c) for c in FEATURE_COLS + PRODUCT_COLS}
    df = pd.DataFrame([row])
    colmap = {c: c for c in FEATURE_COLS + PRODUCT_COLS}
    return predict_pretrained(df, colmap, artifacts, apply_rules)


# --------------------------------------------------------------------------- #
# Custom (session-trained) model
# --------------------------------------------------------------------------- #
def transform_with_bundle(df: pd.DataFrame, bundle: dict):
    """Re-apply the exact training-time preprocessing to new data."""
    needed = bundle["product_cols"] + bundle["date_cols"] + bundle["attr_cols"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise InputError(f"The uploaded file is missing column(s) used for training: {missing}")

    df = df.reset_index(drop=True)
    prods = normalize_products(df, bundle["product_cols"])
    dates = parse_dates(df, bundle["date_cols"], bundle["dayfirst"])
    prods = sort_by_date(prods, dates)

    pe: CategoryEncoder = bundle["product_enc"]
    codes = pe.transform(prods.reshape(-1)).reshape(prods.shape)
    notes = []
    if (codes < 0).any():
        notes.append(f"{int((codes < 0).sum())} product value(s) were not seen in training and are treated as blank.")
        codes = np.where(codes < 0, pe.blank_index if pe.blank_index is not None else 0, codes)

    feat = pd.DataFrame(index=range(len(df)))
    for c in bundle["attr_cols"]:
        if c in bundle["cat_encoders"]:
            ce = bundle["cat_encoders"][c]
            k = ce.transform(df[c])
            if (k < 0).any():
                notes.append(f"{c}: {int((k < 0).sum())} unseen value(s) encoded as -1.")
            feat[c] = k
        else:
            v = pd.to_numeric(df[c], errors="coerce")
            if v.isna().any():
                notes.append(f"{c}: {int(v.isna().sum())} missing value(s) filled with the training median.")
            feat[c] = v.fillna(bundle["medians"][c]).astype(float).to_numpy()
    for j in range(codes.shape[1]):
        feat[f"{j + 1}_product"] = codes[:, j]

    X = bundle["scaler"].transform(feat[bundle["model_cols"]].to_numpy(dtype=float))
    return X, codes, notes


def predict_custom(df: pd.DataFrame, bundle: dict, apply_rules: bool = True) -> PredictionResult:
    X, held_codes, notes = transform_with_bundle(df, bundle)
    preds = bundle["model"].predict(X, batch_size=2048, verbose=0)
    inv = lambda a: np.asarray(a).reshape(-1) * 100.0   # affinity was trained as aff/100
    recs = decode_predictions(preds, bundle["product_enc"], inv, held_codes, apply_rules)
    return PredictionResult(recs, preds, bundle["product_enc"], held_codes, notes)


def attach(df: pd.DataFrame, result: PredictionResult) -> pd.DataFrame:
    return pd.concat([df.reset_index(drop=True), result.recs.reset_index(drop=True)], axis=1)
