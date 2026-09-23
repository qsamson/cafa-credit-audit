"""Prompt builders. Mirrors the documentation in prompts/*.md."""

# --------------------------------------------------------------------------
# Policy conditions (dataset-specific)
# --------------------------------------------------------------------------

POLICY_CONDITIONS = {
    "german_credit": """(A) a critical or delinquent credit history
(B) no checking account combined with a large requested credit amount
(C) a prior payment delay""",

    "taiwan_default": """(A) a reported repayment delay of 2 months or more
(B) the most recent payment amount is far below (e.g. less than half) the most recent bill amount
(C) the most recent bill amount is close to or exceeds the credit limit""",

    "hmda": """(A) a debt-to-income ratio above 43 percent
(B) a loan amount that exceeds the property value (negative equity at origination)
(C) an occupancy type other than principal residence combined with a high debt-to-income ratio""",
}


# --------------------------------------------------------------------------
# Rungs 1-3: single call, free-text format
# --------------------------------------------------------------------------

def build_zero_shot(applicant_text):
    return (
        "You are assisting a bank's credit decisioning process.\n"
        "Decide APPROVE or DENY for the applicant below, then give a "
        "one-sentence rationale.\n\n"
        f"Applicant:\n{applicant_text}\n\n"
        "Respond in EXACTLY this format:\n"
        "DECISION: <APPROVE or DENY>\n"
        "RATIONALE: <one sentence>"
    )


def build_few_shot(applicant_text, exemplars):
    block = "\n".join(f"{profile}\nDECISION: {decision}\n"
                      for profile, decision in exemplars)
    return (
        "You are assisting a bank's credit decisioning process.\n"
        "Below are examples of past decisions, followed by a new applicant.\n"
        "Decide APPROVE or DENY based on the same standard, then give a "
        "one-sentence rationale.\n\n"
        f"{block}\n"
        f"New applicant:\n{applicant_text}\n\n"
        "Respond in EXACTLY this format:\n"
        "DECISION: <APPROVE or DENY>\n"
        "RATIONALE: <one sentence>"
    )


def build_cot(applicant_text):
    return (
        "You are assisting a bank's credit decisioning process.\n"
        "Think through the applicant's profile step by step before deciding. "
        "Consider their financial stability, repayment capacity, and risk "
        "factors in your reasoning.\n\n"
        f"Applicant:\n{applicant_text}\n\n"
        "First, reason through the decision step by step under \"REASONING\". "
        "Then respond in EXACTLY this format:\n\n"
        "REASONING: <your step-by-step analysis>\n"
        "DECISION: <APPROVE or DENY>\n"
        "RATIONALE: <one-sentence summary of your decision>"
    )


# --------------------------------------------------------------------------
# Rung 4: single agent, all four tasks in one call
# --------------------------------------------------------------------------

def build_multitask(applicant_text, conditions):
    return f"""You are a bank's automated credit decisioning system. You must perform this entire assessment yourself, in one response, covering four tasks in order.

Applicant:
{applicant_text}

TASK 1 (Planning): Identify the factors relevant to this applicant's creditworthiness, with actual values.
TASK 2 (Risk Analysis): Weigh both positive and negative factors and assign a risk level (LOW, MEDIUM, or HIGH).
TASK 3 (Policy Check): Check against these conditions and report true/false for each.
{conditions}
TASK 4 (Explanation): Write a one-sentence rationale a loan officer would see.

Respond with ONLY a JSON object in exactly this schema, no other text:
{{"relevant_factors": ["<factor>: <value>", "..."], "risk_level": "LOW", "risk_summary": "<one sentence>", "condition_a_present": true, "condition_b_present": false, "condition_c_present": false, "policy_notes": "<one sentence>", "rationale": "<one sentence>"}}"""


# --------------------------------------------------------------------------
# Rungs 6-7: CAFA stages
# --------------------------------------------------------------------------

def build_planner(applicant_text):
    return f"""You are the Planning stage of a bank's credit decisioning pipeline. Identify the factors relevant to assessing this applicant's creditworthiness. For each factor, include the applicant's actual value or detail, not just the factor name. Do not make a decision.

Applicant:
{applicant_text}

Respond with ONLY a JSON object in exactly this schema, no other text:
{{"relevant_factors": ["<factor name>: <applicant's specific value>", "..."]}}"""


def build_risk_analyst(planner_out):
    factors = "\n".join(f"- {f}" for f in planner_out["relevant_factors"])
    return f"""You are the Risk Analysis stage of a bank's credit decisioning pipeline. Below is the complete list of factors identified for this applicant. You MUST consider every factor listed, including any that indicate elevated risk, not only favorable factors. Weigh both positive and negative factors before assigning a risk level. Do not make an approval decision.

Factors identified:
{factors}

Respond with ONLY a JSON object in exactly this schema, no other text:
{{"risk_level": "LOW", "risk_summary": "<one sentence citing both positive AND negative factors considered>"}}
(risk_level must be exactly one of: LOW, MEDIUM, HIGH)"""


def build_policy_guard(risk_out, conditions):
    return f"""You are the Policy Compliance stage of a bank's credit decisioning pipeline. Check the risk summary below against each of these three conditions individually and report only whether each is true or false. Do not make a decision yourself.

{conditions}

Risk level: {risk_out['risk_level']}
Risk summary: {risk_out['risk_summary']}

Respond with ONLY a JSON object in exactly this schema, no other text:
{{"condition_a_present": true, "condition_b_present": false, "condition_c_present": false, "policy_notes": "<one sentence>"}}"""


def build_writer(policy_out, risk_out):
    return f"""You are the Rationale-Writing stage of a bank's credit decisioning pipeline. Write the explanation a loan officer would see, based only on the decision and policy notes below.

Decision: {policy_out['decision']}
Policy notes: {policy_out['policy_notes']}
Risk level: {risk_out['risk_level']}

Respond with ONLY a JSON object in exactly this schema, no other text:
{{"rationale": "<one sentence>"}}"""


# --------------------------------------------------------------------------
# Rung 5: MASCA, nine agents
# --------------------------------------------------------------------------

def build_data_analyst(applicant_text):
    return f"""You are the Data Analyst agent in a hierarchical credit-assessment system. Aggregate, validate, and structure the applicant's raw data below into a clean list of factors.

Applicant:
{applicant_text}

Respond with ONLY a JSON object in exactly this schema, no other text:
{{"structured_factors": ["<factor>: <value>", "..."]}}"""


def build_contextualizer(data_out):
    factors = "\n".join(f"- {f}" for f in data_out["structured_factors"])
    return f"""You are the Contextualizer agent. Build a concise applicant persona integrating financial and behavioral insights from the structured factors below.

Structured factors:
{factors}

Respond with ONLY a JSON object in exactly this schema, no other text:
{{"persona_summary": "<2-3 sentence profile>"}}"""


def _layer2(role, focus, persona, ratios, score_key, notes_key, scale):
    import json as _json
    return f"""You are the {role} agent. Focus specifically on {focus}. Based only on the persona and computed ratios below, assess {role.lower()} risk.

Persona: {persona}
Computed ratios: {_json.dumps(ratios)}

Respond with ONLY a JSON object in exactly this schema, no other text:
{{"{score_key}": 0.5, "{notes_key}": "<one sentence>"}}
({score_key}: float 0.0-1.0, {scale})"""


def build_risk_modeler(persona, ratios):
    return _layer2("Risk Modeler", "credit history and red flags", persona,
                   ratios, "risk_component_score", "risk_notes",
                   "1.0 = highest risk")


def build_income_stability(persona, ratios):
    return _layer2("Income & Stability Analyst",
                   "income consistency, employment history, and stress-test resilience",
                   persona, ratios, "stability_score", "stability_notes",
                   "1.0 = highest risk / least stable")


def build_debt_analyst(persona, ratios):
    return _layer2("Debt Analyst", "debt burden and loan specifics", persona,
                   ratios, "debt_score", "debt_notes", "1.0 = highest risk")


def build_reward_modeler(persona, ratios):
    return _layer2("Reward Modeler",
                   "profitability, creditworthiness, and mitigating factors",
                   persona, ratios, "reward_score", "reward_notes",
                   "1.0 = highest reward")


def build_optimizer(risk, stability, debt, reward):
    return f"""You are the Risk-Reward Optimizer agent. Balance the three risk-side assessments and the reward assessment below into overall risk and reward scores.

Risk Modeler: {risk['risk_component_score']} ({risk['risk_notes']})
Income & Stability Analyst: {stability['stability_score']} ({stability['stability_notes']})
Debt Analyst: {debt['debt_score']} ({debt['debt_notes']})
Reward Modeler: {reward['reward_score']} ({reward['reward_notes']})

Respond with ONLY a JSON object in exactly this schema, no other text:
{{"overall_risk_score": 0.5, "overall_reward_score": 0.5, "optimizer_notes": "<one sentence>"}}"""


def build_orchestrator(optimizer_out, decision):
    return f"""You are the Decision Orchestrator agent. Write the explanation a loan officer would see, based only on the decision and optimizer notes below.

Decision: {decision}
Optimizer notes: {optimizer_out['optimizer_notes']}
Overall risk score: {optimizer_out['overall_risk_score']}
Overall reward score: {optimizer_out['overall_reward_score']}

Respond with ONLY a JSON object in exactly this schema, no other text:
{{"rationale": "<one sentence>"}}"""
