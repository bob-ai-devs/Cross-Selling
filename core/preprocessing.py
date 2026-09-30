"""Shared data-preparation helpers (no Streamlit / TensorFlow imports here)."""
from __future__ import annotations

import re
from typing import Iterable

import numpy as np
import pandas as pd

BLANK_TOKENS = {"", "none", "null", "nan", "na", "n/a", "nat", "-", "--"}
_SPECIAL = {"rd": "RD", "fd": "FD"}


class InputError(Exception):
    """Raised for user-fixable data problems; carries a list of messages."""

    def __init__(self, messages):
        self.messages = [messages] if isinstance(messages, str) else list(messages)
        super().__init__("; ".join(self.messages))


def pretty(label) -> str:
    """Display label for a product code ('personal loan' -> 'Personal Loan')."""
    if label is None:
        return "—"
    s = str(label).strip()
    if s.lower() in BLANK_TOKENS:
        return "—"
    return _SPECIAL.get(s.lower(), s.title())


# --------------------------------------------------------------------------- #
# Encoders
# --------------------------------------------------------------------------- #
class CategoryEncoder:
    """Tiny LabelEncoder replacement: tolerant to case/whitespace, unseen -> -1."""

    def __init__(self, classes: Iterable, case_insensitive: bool = True):
        self.classes = [str(c) for c in classes]
        self.ci = case_insensitive
        self._lookup = {self._k(c): i for i, c in enumerate(self.classes)}

    @classmethod
    def from_label_encoder(cls, le, case_insensitive: bool = True) -> "CategoryEncoder":
        return cls(le.classes_, case_insensitive)

    def _keys(self, values) -> pd.Series:
        s = pd.Series(np.asarray(values, dtype=object)).astype("string").fillna("").str.strip()
        return s.str.lower() if self.ci else s

    def _k(self, v) -> str:
        s = "" if v is None else str(v).strip()
        return s.lower() if self.ci else s

    def transform(self, values) -> np.ndarray:
        codes = self._keys(values).map(self._lookup)
        return codes.fillna(-1).astype(int).to_numpy()

    def unseen(self, values) -> list[str]:
        keys = self._keys(values)
        bad = keys[~keys.isin(list(self._lookup))].unique().tolist()
        return sorted("(blank)" if b == "" else b for b in bad)

    def inverse(self, idx) -> np.ndarray:
        return np.asarray(self.classes, dtype=object)[np.asarray(idx, dtype=int)]

    @property
    def blank_index(self):
        return self._lookup.get("")


# --------------------------------------------------------------------------- #
# Product / date handling
# --------------------------------------------------------------------------- #
def normalize_products(df: pd.DataFrame, cols: list[str]) -> np.ndarray:
    """(n, k) object array of lower-cased product codes; blanks -> ''."""
    out = np.empty((len(df), len(cols)), dtype=object)
    for j, c in enumerate(cols):
        s = df[c].astype("string").str.strip().str.lower().fillna("")
        s = s.mask(s.isin(list(BLANK_TOKENS)), "")
        out[:, j] = s.to_numpy(dtype=object)
    return out


def parse_dates(df: pd.DataFrame, cols: list[str], dayfirst: bool = False) -> np.ndarray:
    """(n, k) datetime64[ns] array; unparseable -> NaT."""
    parsed = []
    for c in cols:
        try:
            s = pd.to_datetime(df[c], errors="coerce", dayfirst=dayfirst, format="mixed")
        except (TypeError, ValueError):
            s = pd.to_datetime(df[c], errors="coerce", dayfirst=dayfirst)
        if getattr(s.dt, "tz", None) is not None:
            s = s.dt.tz_localize(None)
        parsed.append(s.to_numpy(dtype="datetime64[ns]"))
    return np.column_stack(parsed)


def sort_by_date(products: np.ndarray, dates: np.ndarray) -> np.ndarray:
    """Order each row's products chronologically. Products lacking a date come
    after dated ones; blanks always go last (stable sort keeps ties in order)."""
    big = np.iinfo(np.int64).max
    key = dates.astype("int64")
    key = np.where(np.isnat(dates), big - 1, key)
    key = np.where(products == "", big, key)
    order = np.argsort(key, axis=1, kind="stable")
    return np.take_along_axis(products, order, axis=1)


# --------------------------------------------------------------------------- #
# Column matching
# --------------------------------------------------------------------------- #
def norm_col(name) -> str:
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


def auto_map(columns, required: list[str], aliases: dict | None = None) -> dict:
    """Map required names -> actual column names (None when not found)."""
    aliases = aliases or {}
    lookup = {norm_col(c): c for c in columns}
    out = {}
    for req in required:
        cands = [req] + aliases.get(req, [])
        out[req] = next((lookup[norm_col(c)] for c in cands if norm_col(c) in lookup), None)
    return out


# --------------------------------------------------------------------------- #
# Data-quality profile
# --------------------------------------------------------------------------- #
def profile_frame(df: pd.DataFrame) -> pd.DataFrame:
    n = max(len(df), 1)
    rows = []
    for c in df.columns:
        s = df[c]
        rows.append({
            "Column": c,
            "Type": str(s.dtype),
            "Missing": int(s.isna().sum()),
            "Missing %": round(100 * s.isna().sum() / n, 1),
            "Unique": int(s.nunique(dropna=True)),
            "Example": "" if s.dropna().empty else str(s.dropna().iloc[0])[:30],
        })
    return pd.DataFrame(rows)
