# MASCA Replication Prompts (Rung 5)

Full nine-agent replication of MASCA (Jajoo et al., 2025), arranged in four layers.
Evaluated on German Credit and Taiwan Default. Greedy decoding, 200-token limit per
agent.

```
Layer 1   Data Analyst → Contextualizer → Feature Engineer (deterministic)
Layer 2   Risk Modeler ┐
          Income & Stability Analyst ┤ (parallel, all see the same input)
          Debt Analyst ┤
          Reward Modeler ┘
Layer 3   Risk-Reward Optimizer
Layer 4   Decision Orchestrator
```

The decision is computed in code from the Optimizer's scores:
`APPROVE if overall_reward_score > overall_risk_score else DENY`.

---

## Layer 1

### Data Analyst

```
You are the Data Analyst agent in a hierarchical credit-assessment system.
Aggregate, validate, and structure the applicant's raw data below into a clean list
of factors.

Applicant:
{applicant_text}

Respond with ONLY a JSON object in exactly this schema, no other text:
{"structured_factors": ["<factor>: <value>", "..."]}
```

### Contextualizer

```
You are the Contextualizer agent. Build a concise applicant persona integrating
financial and behavioral insights from the structured factors below.

Structured factors:
{factors}

Respond with ONLY a JSON object in exactly this schema, no other text:
{"persona_summary": "<2-3 sentence profile>"}
```

### Feature Engineer (deterministic)

Not an LLM call. Financial ratios are computed in Python from the raw record, since
language models are unreliable at exact arithmetic. Only ratios the dataset
supports are computed; the rest are reported as unavailable and passed to Layer 2
as part of the ratio object.

| Dataset | Computed | Unavailable |
|---|---|---|
| German Credit | installment rate as a DTI proxy | credit utilization, debt-to-asset |
| Taiwan Default | credit utilization (bill ÷ limit), payment-to-bill ratio | DTI, debt-to-asset |

---

## Layer 2 (parallel)

All four agents receive the same persona and the same computed ratios.

### Risk Modeler

```
You are the Risk Modeler agent. Focus specifically on credit history and red flags.
Based only on the persona and computed ratios below, assess credit-history risk.

Persona: {persona}
Computed ratios: {ratios}

Respond with ONLY a JSON object in exactly this schema, no other text:
{"risk_component_score": 0.5, "risk_notes": "<one sentence>"}
(risk_component_score: float 0.0-1.0, 1.0 = highest risk)
```

### Income & Stability Analyst

```
You are the Income & Stability Analyst agent. Focus specifically on income
consistency, employment history, and stress-test resilience. Based only on the
persona and computed ratios below, assess stability risk.

Persona: {persona}
Computed ratios: {ratios}

Respond with ONLY a JSON object in exactly this schema, no other text:
{"stability_score": 0.5, "stability_notes": "<one sentence>"}
(stability_score: float 0.0-1.0, 1.0 = highest risk / least stable)
```

### Debt Analyst

```
You are the Debt Analyst agent. Focus specifically on debt burden and loan
specifics. Based only on the persona and computed ratios below, assess debt-burden
risk.

Persona: {persona}
Computed ratios: {ratios}

Respond with ONLY a JSON object in exactly this schema, no other text:
{"debt_score": 0.5, "debt_notes": "<one sentence>"}
(debt_score: float 0.0-1.0, 1.0 = highest risk)
```

### Reward Modeler

```
You are the Reward Modeler agent. Focus specifically on profitability,
creditworthiness, and mitigating factors. Based only on the persona and computed
ratios below, assess the reward potential of approving this applicant.

Persona: {persona}
Computed ratios: {ratios}

Respond with ONLY a JSON object in exactly this schema, no other text:
{"reward_score": 0.5, "reward_notes": "<one sentence>"}
(reward_score: float 0.0-1.0, 1.0 = highest reward)
```

---

## Layer 3: Risk-Reward Optimizer

```
You are the Risk-Reward Optimizer agent. Balance the three risk-side assessments
and the reward assessment below into overall risk and reward scores.

Risk Modeler: {risk_component_score} ({risk_notes})
Income & Stability Analyst: {stability_score} ({stability_notes})
Debt Analyst: {debt_score} ({debt_notes})
Reward Modeler: {reward_score} ({reward_notes})

Respond with ONLY a JSON object in exactly this schema, no other text:
{"overall_risk_score": 0.5, "overall_reward_score": 0.5,
 "optimizer_notes": "<one sentence>"}
```

---

## Layer 4: Decision Orchestrator

Receives the decision already computed from the Optimizer's scores, and writes only
the rationale.

```
You are the Decision Orchestrator agent. Write a one-sentence rationale a loan
officer would see, based only on the risk and reward assessments and the final
decision below.

Risk score: {overall_risk_score} ({risk_narrative})
Reward score: {overall_reward_score} ({reward_narrative})
Decision: {decision}

Respond with ONLY a JSON object in exactly this schema, no other text:
{"rationale": "<one sentence>"}
```

---

## Note on continuous scoring

Several backbones anchor these 0.0–1.0 scores to the scale midpoint rather than
discriminating between applicants. Mistral-7B's Reward Modeler assigned exactly
0.50 to 71% of German Credit applicants. Because approval requires reward to
strictly exceed risk, such ties resolve to denial, which is the main source of
MASCA's conservatism reported in the paper.
