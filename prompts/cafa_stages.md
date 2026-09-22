# CAFA Stage Prompts

Four stages, exchanging JSON only. Each prompt is sent as a single user message
with the model's chat template applied. Decoding is greedy (`do_sample=False`),
with a 250-token limit per stage.

---

## 1. Planner

```
You are the Planning stage of a bank's credit decisioning pipeline. Identify the
factors relevant to assessing this applicant's creditworthiness. For each factor,
include the applicant's actual value or detail, not just the factor name. Do not
make a decision.

Applicant:
{applicant_text}

Respond with ONLY a JSON object in exactly this schema, no other text:
{"relevant_factors": ["<factor name>: <applicant's specific value>", "..."]}
```

---

## 2. Risk Analyst

```
You are the Risk Analysis stage of a bank's credit decisioning pipeline. Below is
the complete list of factors identified for this applicant. You MUST consider every
factor listed, including any that indicate elevated risk, not only favorable
factors. Weigh both positive and negative factors before assigning a risk level.
Do not make an approval decision.

Factors identified:
{factors}

Respond with ONLY a JSON object in exactly this schema, no other text:
{"risk_level": "LOW", "risk_summary": "<one sentence citing both positive AND
negative factors considered>"}
(risk_level must be exactly one of: LOW, MEDIUM, HIGH)
```

`{factors}` is the Planner's `relevant_factors` list, rendered one per line with a
leading hyphen.

---

## 3. Policy Guard

The three conditions are dataset-specific. The decision is **not** produced by the
model: it is computed in code as `DENY if (A or B or C) else APPROVE`.

### German Credit

```
You are the Policy Compliance stage of a bank's credit decisioning pipeline. Check
the risk summary below against each of these three conditions individually and
report only whether each is true or false. Do not make a decision yourself.

(A) a critical or delinquent credit history
(B) no checking account combined with a large requested credit amount
(C) a prior payment delay

Risk level: {risk_level}
Risk summary: {risk_summary}

Respond with ONLY a JSON object in exactly this schema, no other text:
{"condition_a_present": true, "condition_b_present": false,
 "condition_c_present": false, "policy_notes": "<one sentence explaining which
 conditions were found present or absent>"}
```

### Taiwan Default

Same prompt, with these conditions:

```
(A) a reported repayment delay of 2 months or more
(B) the most recent payment amount is far below (e.g. less than half) the most
    recent bill amount, indicating inability to pay down the balance
(C) the most recent bill amount is close to or exceeds the credit limit,
    indicating maxed-out credit
```

### HMDA

Same prompt, with these conditions:

```
(A) a debt-to-income ratio above 43 percent
(B) a loan amount that exceeds the property value (negative equity at origination)
(C) an occupancy type other than principal residence combined with a high
    debt-to-income ratio
```

---

## 4. Writer

```
You are the Rationale-Writing stage of a bank's credit decisioning pipeline. Write
the explanation a loan officer would see, based only on the decision and policy
notes below.

Decision: {decision}
Policy notes: {policy_notes}
Risk level: {risk_level}

Respond with ONLY a JSON object in exactly this schema, no other text:
{"rationale": "<one sentence>"}
```

---

## Parsing

Responses are parsed by attempting, in order: direct JSON load, extraction from a
fenced code block, then extraction of the outermost `{...}` span. A stage is marked
`ok` only if the parsed object contains all required keys; downstream stages are
skipped otherwise.
