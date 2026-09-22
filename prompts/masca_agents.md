# Baseline Prompts (Rungs 1–4)

Single-call conditions. Greedy decoding throughout; token limits noted per rung.

---

## Rung 1: Zero-shot (300 tokens)

```
You are assisting a bank's credit decisioning process.
Decide APPROVE or DENY for the applicant below, then give a one-sentence rationale.

Applicant:
{applicant_text}

Respond in EXACTLY this format:
DECISION: <APPROVE or DENY>
RATIONALE: <one sentence>
```

---

## Rung 2: Few-shot (300 tokens)

Six exemplars, three approved and three denied, drawn from the same dataset pool
with seed 0 and rendered in the same serialization as the target applicant.

```
You are assisting a bank's credit decisioning process.
Below are examples of past decisions, followed by a new applicant.
Decide APPROVE or DENY based on the same standard, then give a one-sentence
rationale.

{exemplar_block}

New applicant:
{applicant_text}

Respond in EXACTLY this format:
DECISION: <APPROVE or DENY>
RATIONALE: <one sentence>
```

Each exemplar is rendered as the serialized profile followed by
`DECISION: <APPROVE|DENY>`.

---

## Rung 3: Chain-of-thought (700 tokens)

```
You are assisting a bank's credit decisioning process.
Think through the applicant's profile step by step before deciding. Consider their
financial stability, repayment capacity, and risk factors in your reasoning.

Applicant:
{applicant_text}

First, reason through the decision step by step under "REASONING". Then respond in
EXACTLY this format:

REASONING: <your step-by-step analysis>
DECISION: <APPROVE or DENY>
RATIONALE: <one-sentence summary of your decision>
```

---

## Rung 4: Single-agent multitask (400 tokens)

One call performing all four pipeline tasks. Uses the same JSON schema and the same
code-computed decision rule as CAFA, so the comparison isolates decomposition
rather than output format. Conditions (A)–(C) are dataset-specific and match those
in `cafa_stages.md`.

```
You are a bank's automated credit decisioning system. You must perform this entire
assessment yourself, in one response, covering four tasks in order: identifying
relevant factors, assessing risk, checking policy conditions, and writing a
rationale.

Applicant:
{applicant_text}

TASK 1 (Planning): Identify the factors relevant to this applicant's
creditworthiness. For each factor, include the applicant's actual value, not just
the factor name.

TASK 2 (Risk Analysis): Weigh both positive and negative factors and assign a risk
level (LOW, MEDIUM, or HIGH).

TASK 3 (Policy Check): Check the risk assessment against each of these three
conditions individually and report only whether each is true or false.
(A) a critical or delinquent credit history
(B) no checking account combined with a large requested credit amount
(C) a prior payment delay

TASK 4 (Explanation): Write a one-sentence rationale a loan officer would see.

Respond with ONLY a JSON object in exactly this schema, no other text:
{"relevant_factors": ["<factor>: <value>", "..."], "risk_level": "LOW",
 "risk_summary": "<one sentence>", "condition_a_present": true,
 "condition_b_present": false, "condition_c_present": false,
 "policy_notes": "<one sentence>", "rationale": "<one sentence>"}
(risk_level must be exactly one of: LOW, MEDIUM, HIGH)
```

---

## Parsing

Rungs 1–3 are parsed with case-insensitive regular expressions on the `DECISION:`
and `RATIONALE:` lines, tolerating surrounding asterisks from Markdown-style
emphasis. Rung 4 is parsed as JSON, using the same fallback chain as the CAFA
stages, and its decision is computed in code from the three condition flags.
