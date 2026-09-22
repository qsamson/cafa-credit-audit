# CAFA: Cascading Adversarial Fairness Auditing

Code, prompts, and data for *Where Does Bias Enter the Pipeline? A Theoretical and Empirical Audit of Cascading Fairness in LLM-Powered Multi-Agent Credit Decisioning*.

CAFA formalizes an agentic credit pipeline as a composition of stage-level decision functions and localizes **where** a fairness violation enters, using stage truncation, propagation coefficients, and bias-locus attribution.

<p align="center">
  <img src="CAFA-Framework.png" alt="CAFA framework" width="100%">
</p>

## Key findings

- Bias concentrates at the **Policy Guard**, which accounts for 69–100% of the perturbation-induced fairness shift across three datasets.
- Decomposition removes a proxy-discrimination effect but makes decisions **more sensitive** to perturbation, without measurably worsening group parity.
- Against a full nine-agent MASCA replication, CAFA reaches comparable fairness while approving nearly twice as many creditworthy applicants.

## Contents

```
prompts/     CAFA stage prompts, baselines, nine MASCA agents
src/         generation and analysis scripts
data/        sampled applicant IDs, HMDA query and filters
results/     raw model outputs, tables, figures
```

## Setup

```bash
pip install -r requirements.txt
```

Generation requires a GPU; analysis runs on CPU.

## Datasets

German Credit (UCI id 144) and Taiwan Default (UCI id 350) download automatically via `ucimlrepo`. HMDA comes from the CFPB Data Browser:

```
https://ffiec.cfpb.gov/v2/data-browser-api/view/csv?years=2024&states=IL&loan_purposes=1
```

Filters and the stratified 75/75 sample are documented in `data/hmda_filter.md`. Sampling uses seed 42 throughout; exact applicant IDs are in `data/applicant_ids/`.

## Reproducing the paper

```bash
python src/run_baselines.py   # rungs 1-4, all datasets
python src/run_pipeline.py    # CAFA homogeneous and heterogeneous, clean and perturbed
python src/run_masca.py       # nine-agent MASCA replication
python src/analysis.py        # all tables and figures
```

`analysis.py` reproduces every number in the paper from `results/checkpoints/`, without re-running generation.

## Models

Llama-3.1-8B-Instruct, Mistral-7B-Instruct-v0.3, Qwen2.5-7B-Instruct, Gemma-2-9B-IT, and Phi-4-mini-instruct, all 4-bit NF4 quantized with greedy decoding.

## Citation

```bibtex
@article{cafa2026,
  title   = {Where Does Bias Enter the Pipeline? A Theoretical and Empirical Audit
             of Cascading Fairness in LLM-Powered Multi-Agent Credit Decisioning},
  author  = {Ben Ayed, Ahmed and Quaye, Samson and Nobles, Calvin},
  year    = {2026}
}
```

## Acknowledgements

This research used resources of the Argonne Leadership Computing Facility, a U.S. DOE Office of Science user facility, under Contract No. DE-AC02-06CH11357, through a Director's Discretionary award (Project EDGEGUARD) on Polaris.
