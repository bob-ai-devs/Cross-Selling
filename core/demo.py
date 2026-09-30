"""Synthetic data so the app can be explored without uploading anything."""
from __future__ import annotations

import numpy as np
import pandas as pd

from core.config import OPTIONS, PRODUCT_COLS, PRODUCTS


def _customers(n: int, rng: np.random.Generator) -> pd.DataFrame:
    age = rng.integers(21, 66, n)
    emp = rng.choice(OPTIONS["Employment_Type"], n, p=[0.55, 0.2, 0.15, 0.1])
    years = np.clip(age - 21 - rng.integers(0, 8, n), 0, 40)
    income = np.round(rng.lognormal(13.2, 0.55, n) * (1 + years / 40), -3)
    stable = np.where(np.isin(emp, ["Salaried"]), "Stable",
                      rng.choice(["Stable", "Unstable"], n, p=[0.55, 0.45]))
    credit = np.clip((520 + 40 * np.log1p(income / 1e5) + rng.normal(0, 55, n)).astype(int), 300, 900)
    return pd.DataFrame({
        "CustomerID": np.arange(10001, 10001 + n),
        "Age": age,
        "Gender": rng.choice(OPTIONS["Gender"], n),
        "Marital_Status": rng.choice(OPTIONS["Marital_Status"], n, p=[0.3, 0.55, 0.1, 0.05]),
        "Annual_Income": income.astype(int),
        "Employment_Type": emp,
        "Income_Stability": stable,
        "Credit_Score": credit,
        "Past_Loan_Defaults": (rng.random(n) < np.clip((700 - credit) / 500, 0.02, 0.5)).astype(int),
        "Credit_Utilization_Ratio": np.round(rng.beta(2, 4, n), 2),
        "Property_Ownership": rng.choice(OPTIONS["Property_Ownership"], n, p=[0.3, 0.5, 0.2]),
        "Liquid_Assets": (income * rng.uniform(0.05, 1.2, n)).astype(int),
        "Investments": (income * rng.uniform(0.0, 2.5, n)).astype(int),
        "Job_Title": rng.choice(OPTIONS["Job_Title"], n),
        "Company_Size": rng.choice(OPTIONS["Company_Size"], n),
        "Years_of_Employment": years,
    })


def _holdings(df: pd.DataFrame, rng: np.random.Generator, max_products: int = 5) -> list[list[str]]:
    """Plausible product holdings driven by the customer profile."""
    out = []
    for _, r in df.iterrows():
        w = np.ones(len(PRODUCTS))
        w[PRODUCTS.index("savings")] = 4
        w[PRODUCTS.index("credit card")] = 1 + r.Credit_Score / 300
        w[PRODUCTS.index("home loan")] = 0.5 + (r.Age > 30) * 1.5 + (r.Property_Ownership == "Mortgage") * 3
        w[PRODUCTS.index("car loan")] = 1 + (r.Annual_Income > 6e5)
        w[PRODUCTS.index("fd")] = 0.5 + r.Liquid_Assets / 8e5
        w[PRODUCTS.index("rd")] = 1 + (r.Age < 35)
        w[PRODUCTS.index("insurance")] = 0.5 + (r.Marital_Status == "Married") * 2
        w[PRODUCTS.index("current")] = 0.3 + (r.Employment_Type in ("Business Owner", "Self-Employed")) * 3
        w[PRODUCTS.index("overdraft")] = 0.2 + (r.Employment_Type == "Business Owner") * 1.5
        w[PRODUCTS.index("personal loan")] = 1 + (r.Credit_Score < 650) * 1.5
        k = int(np.clip(rng.poisson(1.3 + r.Annual_Income / 2e6), 0, max_products))
        out.append(list(rng.choice(PRODUCTS, size=k, replace=False, p=w / w.sum())) if k else [])
    return out


def make_batch_demo(n: int = 300, seed: int = 7) -> pd.DataFrame:
    """Data in the layout the pre-trained network expects."""
    rng = np.random.default_rng(seed)
    df = _customers(n, rng)
    holds = _holdings(df, rng)
    for j, col in enumerate(PRODUCT_COLS):
        df[col] = [h[j] if len(h) > j else "" for h in holds]
    return df


def make_history_demo(n: int = 1500, seed: int = 11) -> pd.DataFrame:
    """Historical holdings with acquisition dates (columns deliberately unsorted)."""
    rng = np.random.default_rng(seed)
    df = _customers(n, rng)
    holds = _holdings(df, rng)
    start = pd.Timestamp("2016-01-01")
    for j in range(5):
        prods, dates = [], []
        for h in holds:
            if len(h) > j:
                prods.append(h[j])
                dates.append(start + pd.Timedelta(days=int(rng.integers(0, 3500))))
            else:
                prods.append("")
                dates.append(pd.NaT)
        df[f"Product_{j + 1}"] = prods
        df[f"Acquired_{j + 1}"] = pd.to_datetime(dates).strftime("%Y-%m-%d")
    # dates are drawn independently per slot, so chronological sorting really matters
    return df
