"""Dataset loading, applicant serialization, perturbation, and protected attributes.

All sampling uses seed 42. Protected attributes are never included in the text
shown to a model, and are never modified by the perturbation.
"""

import pandas as pd
from ucimlrepo import fetch_ucirepo

SEED = 42
EPSILON = 0.10
HMDA_CSV = "data/hmda_illinois_2024.csv"


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------

def load_german_credit(n=150, seed=SEED):
    ds = fetch_ucirepo(id=144)
    df = ds.data.features.copy()
    df["target"] = ds.data.targets.iloc[:, 0]
    df = df.sample(n=min(n, len(df)), random_state=seed).reset_index(drop=True)
    df["applicant_id"] = [f"german_{i}" for i in range(len(df))]
    return df


def load_taiwan_default(n=150, seed=SEED):
    ds = fetch_ucirepo(id=350)
    df = ds.data.features.copy()
    df["target"] = ds.data.targets.iloc[:, 0]
    df = df.sample(n=min(n, len(df)), random_state=seed).reset_index(drop=True)
    df["applicant_id"] = [f"taiwan_{i}" for i in range(len(df))]
    return df


def load_hmda(n=150, seed=SEED, path=HMDA_CSV):
    """Filtered and outcome-stratified. See data/hmda_filter.md."""
    raw = pd.read_csv(path, low_memory=False)
    df = raw[raw["action_taken"].isin([1, 3])]
    df = df[df["applicant_race-1"].isin([1, 2, 3, 4, 5])]
    df = df[df["applicant_sex"].isin([1, 2])]
    df = df[df["income"].notna()]
    df = df[df["debt_to_income_ratio"].notna()]

    half = n // 2
    originated = df[df["action_taken"] == 1].sample(n=half, random_state=seed)
    denied = df[df["action_taken"] == 3].sample(n=half, random_state=seed)
    out = (pd.concat([originated, denied])
             .sample(frac=1, random_state=seed)
             .reset_index(drop=True))
    out["applicant_id"] = [f"hmda_{i}" for i in range(len(out))]
    return out


LOADERS = {
    "german_credit": load_german_credit,
    "taiwan_default": load_taiwan_default,
    "hmda": load_hmda,
}


# --------------------------------------------------------------------------
# Serialization
# --------------------------------------------------------------------------

_GC = {
    "checking": {"A11": "less than 0 DM", "A12": "0 to 200 DM",
                 "A13": "200 DM or more", "A14": "no checking account"},
    "history": {"A30": "no credits taken or all credits paid back duly",
                "A31": "all credits at this bank paid back duly",
                "A32": "existing credits paid back duly until now",
                "A33": "delay in paying off in the past",
                "A34": "critical account or other credits existing at other banks"},
    "savings": {"A61": "less than 100 DM", "A62": "100 to 500 DM",
                "A63": "500 to 1000 DM", "A64": "1000 DM or more",
                "A65": "unknown or no savings account"},
    "employment": {"A71": "unemployed", "A72": "less than 1 year",
                   "A73": "1 to 4 years", "A74": "4 to 7 years",
                   "A75": "7 years or more"},
    "status": {"A91": "male, divorced or separated",
               "A92": "female, divorced, separated, or married",
               "A93": "male, single", "A94": "male, married or widowed",
               "A95": "female, single"},
    "property": {"A121": "real estate",
                 "A122": "building society savings or life insurance",
                 "A123": "car or other property",
                 "A124": "unknown or no property"},
    "installment": {"A141": "bank", "A142": "stores", "A143": "none"},
    "housing": {"A151": "rents", "A152": "owns", "A153": "lives for free"},
    "job": {"A171": "unemployed or unskilled, non-resident",
            "A172": "unskilled, resident",
            "A173": "skilled employee or official",
            "A174": "management, self-employed, or highly qualified employee"},
}


def serialize_german(row):
    g = _GC
    parts = [
        f"Age: {row.get('Attribute13', 'N/A')} years.",
        f"Sex and personal status: {g['status'].get(row.get('Attribute9'), 'N/A')}.",
        f"Job classification: {g['job'].get(row.get('Attribute17'), 'N/A')}.",
        f"Housing status: {g['housing'].get(row.get('Attribute15'), 'N/A')}.",
        f"Checking account status: {g['checking'].get(row.get('Attribute1'), 'N/A')}.",
        f"Credit history: {g['history'].get(row.get('Attribute3'), 'N/A')}.",
        f"Savings account: {g['savings'].get(row.get('Attribute6'), 'N/A')}.",
        f"Present employment duration: {g['employment'].get(row.get('Attribute7'), 'N/A')}.",
        f"Installment rate: {row.get('Attribute8', 'N/A')} percent of disposable income.",
        f"Property owned: {g['property'].get(row.get('Attribute12'), 'N/A')}.",
        f"Other installment plans: {g['installment'].get(row.get('Attribute14'), 'N/A')}.",
        f"Credit amount requested: {row.get('Attribute5', 'N/A')} DM.",
        f"Loan duration: {row.get('Attribute2', 'N/A')} months.",
        f"Existing credits at this bank: {row.get('Attribute16', 'N/A')}.",
        f"Purpose of loan: {row.get('Attribute4', 'N/A')}.",
    ]
    return " ".join(parts)


_TW_REPAY = {-2: "no balance used", -1: "paid in full and on time",
             0: "revolving credit used, minimum paid on time",
             1: "payment delayed 1 month", 2: "payment delayed 2 months",
             3: "payment delayed 3 months", 4: "payment delayed 4 months",
             5: "payment delayed 5 months or more"}
_TW_EDU = {1: "graduate school", 2: "university", 3: "high school", 4: "other"}
_TW_MAR = {1: "married", 2: "single", 3: "other"}
_TW_SEX = {1: "male", 2: "female"}


def serialize_taiwan(row):
    parts = [
        f"Credit limit: {row.get('X1', 'N/A')} NT dollars.",
        f"Sex: {_TW_SEX.get(row.get('X2'), 'N/A')}.",
        f"Education level: {_TW_EDU.get(row.get('X3'), 'N/A')}.",
        f"Marital status: {_TW_MAR.get(row.get('X4'), 'N/A')}.",
        f"Age: {row.get('X5', 'N/A')} years.",
        f"Most recent repayment status: {_TW_REPAY.get(row.get('X6'), 'unknown')}.",
        f"Most recent bill amount: {row.get('X12', 'N/A')} NT dollars.",
        f"Most recent payment amount: {row.get('X18', 'N/A')} NT dollars.",
    ]
    return " ".join(parts)


_LOAN_TYPE = {1: "Conventional", 2: "FHA", 3: "VA"}
_OCCUPANCY = {1: "Principal residence", 2: "Second residence"}


def serialize_hmda(row):
    parts = [
        f"Annual income: {row.get('income', 'N/A')} thousand dollars.",
        f"Debt-to-income ratio: {row.get('debt_to_income_ratio', 'N/A')} percent.",
        f"Loan amount requested: {row.get('loan_amount', 'N/A')} dollars.",
        f"Property value: {row.get('property_value', 'N/A')} dollars.",
        f"Loan term: {row.get('loan_term', 'N/A')} months.",
        f"Loan type: {_LOAN_TYPE.get(row.get('loan_type'), 'USDA/RHS')}.",
        f"Occupancy type: {_OCCUPANCY.get(row.get('occupancy_type'), 'Investment property')}.",
        f"Tract minority population percentage: "
        f"{row.get('tract_minority_population_percent', 'N/A')} percent.",
    ]
    return " ".join(parts)


SERIALIZERS = {
    "german_credit": serialize_german,
    "taiwan_default": serialize_taiwan,
    "hmda": serialize_hmda,
}


# --------------------------------------------------------------------------
# Perturbation (protected attributes are never modified)
# --------------------------------------------------------------------------

def perturb_german(row, eps=EPSILON):
    r = row.copy()
    r["Attribute5"] = int(r["Attribute5"] * (1 - eps))          # credit amount
    r["Attribute2"] = max(1, int(r["Attribute2"] * (1 - eps)))  # duration
    if isinstance(r["Attribute16"], (int, float)):              # existing credits
        r["Attribute16"] = max(1, r["Attribute16"] - 1)
    r["Attribute4"] = "A43"                                     # purpose
    return r


def perturb_taiwan(row, eps=EPSILON):
    r = row.copy()
    r["X1"] = int(r["X1"] * (1 + eps))    # credit limit
    r["X12"] = int(r["X12"] * (1 - eps))  # recent bill
    r["X18"] = int(r["X18"] * (1 + eps))  # recent payment
    r["X3"] = 2                           # education
    return r


def perturb_hmda(row, eps=EPSILON):
    r = row.copy()
    try:
        r["loan_amount"] = float(r["loan_amount"]) * (1 - eps)
        r["income"] = float(r["income"]) * (1 + eps)
    except (ValueError, TypeError):
        pass
    return r


PERTURBERS = {
    "german_credit": perturb_german,
    "taiwan_default": perturb_taiwan,
    "hmda": perturb_hmda,
}


def proxy_override_german(row, case):
    """German Credit proxy probe: worst- vs best-case job and housing."""
    r = row.copy()
    if case == "worst":
        r["Attribute17"], r["Attribute15"] = "A171", "A151"
    else:
        r["Attribute17"], r["Attribute15"] = "A174", "A152"
    return r


# --------------------------------------------------------------------------
# Protected attributes and ground truth
# --------------------------------------------------------------------------

GERMAN_MALE = {"A91", "A93", "A94"}
PRIV_VALUE = {"sex": "male", "age_group": "privileged", "race": "white"}
DATASET_ATTRIBUTES = {
    "german_credit": ["sex", "age_group"],
    "taiwan_default": ["sex", "age_group"],
    "hmda": ["race", "sex"],
}


def build_attr_lookup(german_df=None, taiwan_df=None, hmda_df=None):
    """Map applicant_id to protected attributes and creditworthiness."""
    lookup = {}

    if german_df is not None:
        for _, r in german_df.iterrows():
            lookup[r["applicant_id"]] = {
                "sex": "male" if r["Attribute9"] in GERMAN_MALE else "female",
                "age_group": "privileged" if r["Attribute13"] >= 25 else "unprivileged",
                "creditworthy": r["target"] == 1,
            }

    if taiwan_df is not None:
        median_age = taiwan_df["X5"].median()
        for _, r in taiwan_df.iterrows():
            lookup[r["applicant_id"]] = {
                "sex": "male" if r["X2"] == 1 else "female",
                "age_group": "privileged" if r["X5"] >= median_age else "unprivileged",
                "creditworthy": r["target"] == 0,   # 0 = no default
            }

    if hmda_df is not None:
        for _, r in hmda_df.iterrows():
            lookup[r["applicant_id"]] = {
                "race": "white" if r["applicant_race-1"] == 5.0 else "non_white",
                "sex": "male" if r["applicant_sex"] == 1 else "female",
                "creditworthy": r["action_taken"] == 1,
            }

    return lookup
