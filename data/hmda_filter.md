# HMDA Data: Retrieval and Sampling

## Source

Public loan/application records from the CFPB Data Browser API.

```
GET https://ffiec.cfpb.gov/v2/data-browser-api/view/csv
    ?years=2024
    &states=IL
    &loan_purposes=1
```

`loan_purposes=1` restricts to home-purchase loans. The endpoint rejects the
default Python `requests` user-agent, so a browser user-agent header is required
(see `src/download_hmda.py`). The download is roughly 80 MB and 219,837 records.

## Filters

Applied in order:

| Filter | Condition | Remaining |
|---|---|---|
| Raw download | — | 219,837 |
| Decision made | `action_taken ∈ {1, 3}` | 143,958 |
| Race reported | `applicant_race-1 ∈ {1,2,3,4,5}` | 120,516 |
| Sex reported | `applicant_sex ∈ {1, 2}` | 119,881 |
| Income present | `income` not null | 118,341 |
| DTI present | `debt_to_income_ratio` not null | 117,037 |

`action_taken` 1 is originated and 3 is denied. Codes 2, 4, 5, 6, 7, and 8 cover
withdrawals, incomplete files, purchased loans, and preapproval outcomes, none of
which represent a fresh credit decision. Race codes 6, 7, and 8 mean not provided,
not applicable, and no co-applicant.

## Sampling

The filtered pool is 89% originated, which leaves too few denials to estimate
equalized-odds-family metrics reliably. We therefore sample **75 originated and 75
denied** with `random_state=42`, shuffle with the same seed, and assign sequential
IDs `hmda_0` through `hmda_149`.

Resulting composition:

| | Originated | Denied | Total |
|---|---|---|---|
| White | 58 | 54 | 112 |
| Non-white | 17 | 21 | 38 |
| **Total** | **75** | **75** | **150** |

The race split of 112 to 38 closely matches the filtered pool's natural ratio of
92,597 to 24,440, so race is representative even though outcome is stratified.

## Variable mapping

| Paper term | HMDA field | Encoding |
|---|---|---|
| Race (protected) | `applicant_race-1` | 5 → white; 1–4 → non-white |
| Sex (protected) | `applicant_sex` | 1 → male; 2 → female |
| Creditworthy | `action_taken` | 1 → true; 3 → false |
| Income | `income` | thousands of dollars |
| DTI | `debt_to_income_ratio` | percent |
| Loan amount | `loan_amount` | dollars |
| Property value | `property_value` | dollars |

## Serialized fields

Income, DTI, loan amount, property value, loan term, loan type, occupancy type, and
tract minority population percentage. Race and sex are **never** included in the
serialized profile shown to the model; they are used only to group applicants when
computing fairness metrics.

## Perturbation (ε = 0.10)

`loan_amount × (1 − ε)` and `income × (1 + ε)`. No other field is modified.

## Reproducibility

`data/applicant_ids/hmda_ids.csv` lists the 150 sampled IDs with their HMDA row
indices, so the exact sample can be reconstructed even if the CFPB reissues the
2024 file.
