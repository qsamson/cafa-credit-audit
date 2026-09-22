"""Stage-truncation policies for the CAFA audit.

pi_1 and pi_2 route the Planner's factor list and the Risk Analyst's risk summary
through a neutral policy that applies the same three conditions as the Policy
Guard, recovering the decision each stage would reach on its own.

Numeric values are extracted by regular expression where the dataset supports it,
with keyword detection as a fallback. pi_2_alt is the alternative policy used in
the sensitivity check: deny whenever the Risk Analyst reports HIGH risk.
"""

import re


def _has(text, keywords):
    text = text.lower()
    return any(k in text for k in keywords)


def _num(text, label_patterns):
    """Extract the first number following any of the given label patterns."""
    for pat in label_patterns:
        m = re.search(pat + r"[:\s]+\$?([\d,]+\.?\d*)", text, re.IGNORECASE)
        if m:
            try:
                return float(m.group(1).replace(",", ""))
            except ValueError:
                continue
    return None


# --------------------------------------------------------------------------
# Per-dataset policies: DENY if any of conditions A, B, C holds
# --------------------------------------------------------------------------

def pi_german(text):
    a = _has(text, ["critical"])
    b = _has(text, ["no checking account"])
    c = _has(text, ["delay in paying", "delayed payment", "payment delay"])
    return "DENY" if (a or b or c) else "APPROVE"


def pi_taiwan(text):
    a = _has(text, ["delayed 2 months", "delayed 3 months",
                    "delayed 4 months", "delayed 5 months"])

    bill = _num(text, [r"bill amount", r"recent bill"])
    payment = _num(text, [r"payment amount", r"recent payment"])
    limit = _num(text, [r"credit limit"])

    b = bill is not None and payment is not None and payment < 0.5 * bill
    if not b:
        b = _has(text, ["far below", "unable to pay", "inability"])

    c = bill is not None and limit is not None and bill >= 0.9 * limit
    if not c:
        c = _has(text, ["maxed out", "exceeds the credit limit",
                        "close to the credit limit"])

    return "DENY" if (a or b or c) else "APPROVE"


def pi_hmda(text):
    dti = _num(text, [r"debt-to-income ratio", r"dti"])
    loan = _num(text, [r"loan amount(?:\s*requested)?"])
    prop = _num(text, [r"property value"])

    a = dti is not None and dti > 43
    if not a:
        a = _has(text, ["high debt-to-income", "elevated dti", "excessive dti"])

    b = loan is not None and prop is not None and loan > prop
    c = _has(text, ["investment property", "second residence"]) and \
        (dti is not None and dti > 36)

    return "DENY" if (a or b or c) else "APPROVE"


PI = {
    "german_credit": pi_german,
    "taiwan_default": pi_taiwan,
    "hmda": pi_hmda,
}


def pi_2_alt(record):
    """Alternative pi_2: ignore the narrative, deny iff risk_level is HIGH."""
    level = str(record.get("stage2_parsed", {}).get("risk_level", "")).upper()
    return "DENY" if level == "HIGH" else "APPROVE"


# --------------------------------------------------------------------------
# Stage-truncated decision function F_i
# --------------------------------------------------------------------------

def truncated_decision(record, stage, dataset, alt_pi2=False):
    """Decision the pipeline would reach if it stopped at `stage`.

    Stages 3 and 4 return the pipeline's actual decision, since F_3 = F_4 = F
    by construction. Returns None if the required stage failed to parse.
    """
    if stage >= 3:
        return record.get("final_decision")

    if stage == 1:
        if not record.get("stage1_ok"):
            return None
        text = " ".join(record["stage1_parsed"].get("relevant_factors", []))
        return PI[dataset](text)

    if not record.get("stage2_ok"):
        return None
    if alt_pi2:
        return pi_2_alt(record)
    return PI[dataset](record["stage2_parsed"].get("risk_summary", ""))
