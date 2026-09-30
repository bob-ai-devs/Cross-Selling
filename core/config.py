"""Static configuration: schema, product catalogue, artifact file names."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# ---- Pre-trained artifacts -------------------------------------------------
# Same names as the original Flask app. In the GitHub Gist each file is stored
# base64-encoded with an extra ".b64" suffix (e.g. "scaler.pkl.b64").
MODEL_FILE = "cross_selling.keras"
ENCODER_FILES = [
    "scaler.pkl", "le_gen.pkl", "le_mar.pkl", "le_emp.pkl", "le_inc.pkl",
    "le_prop.pkl", "le_job.pkl", "le_com.pkl", "le_pro.pkl", "aff_sc.pkl",
]
ALL_ARTIFACTS = [MODEL_FILE] + ENCODER_FILES
LOCAL_MODEL_DIR = ROOT / "models"
LOCAL_ENCODES_DIR = ROOT / "encodes"

PRETRAINED_F1 = 96.47  # figure quoted in the original UI

# ---- Product catalogue -----------------------------------------------------
PRODUCTS = [
    "rd", "fd", "personal loan", "home loan", "car loan",
    "credit card", "savings", "current", "insurance", "overdraft",
]
ORDINALS = ["1st", "2nd", "3rd", "4th", "5th"]
PRODUCT_COLS = [f"{o}_product" for o in ORDINALS]
TOP_N = 3

# ---- Customer schema used by the pre-trained network -----------------------
ID_COL = "CustomerID"
NUMERIC_COLS = [
    "Age", "Annual_Income", "Credit_Score", "Past_Loan_Defaults",
    "Credit_Utilization_Ratio", "Liquid_Assets", "Investments", "Years_of_Employment",
]
# column -> artifact key of its LabelEncoder
ENCODED_COLS = {
    "Gender": "le_gen",
    "Marital_Status": "le_mar",
    "Employment_Type": "le_emp",
    "Income_Stability": "le_inc",
    "Property_Ownership": "le_prop",
    "Job_Title": "le_job",
    "Company_Size": "le_com",
}
FEATURE_COLS = [
    "Age", "Gender", "Marital_Status", "Annual_Income", "Employment_Type",
    "Income_Stability", "Credit_Score", "Past_Loan_Defaults",
    "Credit_Utilization_Ratio", "Property_Ownership", "Liquid_Assets",
    "Investments", "Job_Title", "Company_Size", "Years_of_Employment",
]
MODEL_INPUT_COLS = FEATURE_COLS + PRODUCT_COLS

OPTIONS = {
    "Gender": ["Male", "Female"],
    "Marital_Status": ["Single", "Married", "Divorced", "Widowed"],
    "Employment_Type": ["Salaried", "Self-Employed", "Business Owner", "Freelancer"],
    "Income_Stability": ["Stable", "Unstable"],
    "Property_Ownership": ["Own", "Rent", "Mortgage"],
    "Job_Title": ["Manager", "Engineer", "Analyst", "Clerk", "Executive", "Salesperson", "Consultant"],
    "Company_Size": ["Small", "Medium", "Large"],
}

# Accepted alternative header names when auto-mapping uploaded files
ALIASES = {
    ID_COL: ["customer_id", "cust_id", "client_id", "id"],
    **{PRODUCT_COLS[i]: [f"product{i+1}", f"product_{i+1}", f"prod{i+1}", f"product {i+1}"]
       for i in range(5)},
}

PRESETS = {
    "— custom —": None,
    "Young salaried professional": dict(
        Age=27, Gender="Female", Marital_Status="Single", Annual_Income=780000,
        Employment_Type="Salaried", Income_Stability="Stable", Credit_Score=735,
        Past_Loan_Defaults="No", Credit_Utilization_Ratio=0.28, Property_Ownership="Rent",
        Liquid_Assets=210000, Investments=90000, Job_Title="Engineer", Company_Size="Large",
        Years_of_Employment=3, p1="savings", p2="credit card", p3="None", p4="None", p5="None"),
    "Established business owner": dict(
        Age=46, Gender="Male", Marital_Status="Married", Annual_Income=3200000,
        Employment_Type="Business Owner", Income_Stability="Stable", Credit_Score=810,
        Past_Loan_Defaults="No", Credit_Utilization_Ratio=0.15, Property_Ownership="Own",
        Liquid_Assets=2400000, Investments=3100000, Job_Title="Executive", Company_Size="Medium",
        Years_of_Employment=18, p1="current", p2="fd", p3="insurance", p4="overdraft", p5="None"),
    "Credit-stretched freelancer": dict(
        Age=34, Gender="Male", Marital_Status="Married", Annual_Income=520000,
        Employment_Type="Freelancer", Income_Stability="Unstable", Credit_Score=590,
        Past_Loan_Defaults="Yes", Credit_Utilization_Ratio=0.86, Property_Ownership="Rent",
        Liquid_Assets=40000, Investments=15000, Job_Title="Consultant", Company_Size="Small",
        Years_of_Employment=5, p1="savings", p2="personal loan", p3="None", p4="None", p5="None"),
}
