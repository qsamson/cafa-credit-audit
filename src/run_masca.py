"""Rung 5: full nine-agent MASCA replication (Jajoo et al., 2025).

Evaluated on German Credit and Taiwan Default, matching the paper's scope.
Layer 1 includes a deterministic Feature Engineer: financial ratios are computed
in Python, not requested from the model, since language models are unreliable at
exact arithmetic. The decision is computed from the Optimizer's scores.

Usage:
    python src/run_masca.py
    python src/run_masca.py --models phi4-mini --limit 2
"""

import argparse

from common import (append, checkpoint_path, completed_ids, generate,
                    load_model, parse_json, unload, MODEL_REGISTRY)
from prompts_py import (build_contextualizer, build_data_analyst,
                        build_debt_analyst, build_income_stability,
                        build_optimizer, build_orchestrator,
                        build_reward_modeler, build_risk_modeler)
from serialization import LOADERS, SERIALIZERS

MASCA_DATASETS = ["german_credit", "taiwan_default"]
MAX_TOKENS = 200


# --------------------------------------------------------------------------
# Feature Engineer (deterministic, not an LLM call)
# --------------------------------------------------------------------------

def ratios_german(row):
    return {
        "dti_proxy": row.get("Attribute8"),   # installment rate, % of income
        "credit_utilization": None,           # no credit-limit field
        "debt_to_asset": None,                # no numeric asset field
        "note": ("German Credit lacks income, asset, and credit-limit fields; "
                 "only the installment rate is computable as a DTI proxy."),
    }


def ratios_taiwan(row):
    limit, bill, payment = row.get("X1"), row.get("X12"), row.get("X18")
    util = (bill / limit) if (limit and bill is not None and limit != 0) else None
    pay_ratio = (payment / bill) if (bill and payment is not None and bill != 0) else None
    return {
        "dti_proxy": None,                    # no income field
        "credit_utilization": round(util, 3) if util is not None else None,
        "payment_to_bill_ratio": round(pay_ratio, 3) if pay_ratio is not None else None,
        "debt_to_asset": None,
        "note": ("Taiwan Default lacks an income field, so DTI is not computable. "
                 "Utilization and payment ratio are computed directly."),
    }


RATIOS = {"german_credit": ratios_german, "taiwan_default": ratios_taiwan}


# --------------------------------------------------------------------------
# Pipeline
# --------------------------------------------------------------------------

LAYER2 = [
    ("risk_modeler", build_risk_modeler, ["risk_component_score", "risk_notes"]),
    ("income_stability", build_income_stability, ["stability_score", "stability_notes"]),
    ("debt_analyst", build_debt_analyst, ["debt_score", "debt_notes"]),
    ("reward_modeler", build_reward_modeler, ["reward_score", "reward_notes"]),
]


def run_one(model, tok, applicant_text, ratios):
    raw = generate(model, tok, build_data_analyst(applicant_text), MAX_TOKENS)
    data_out, ok_data = parse_json(raw, ["structured_factors"])

    ctx_out = None; ok_ctx = False
    if ok_data:
        raw = generate(model, tok, build_contextualizer(data_out), MAX_TOKENS)
        ctx_out, ok_ctx = parse_json(raw, ["persona_summary"])

    # Layer 2: four parallel agents, all seeing the same persona and ratios.
    layer2, flags = {}, {}
    if ok_ctx:
        persona = ctx_out["persona_summary"]
        for name, builder, keys in LAYER2:
            raw = generate(model, tok, builder(persona, ratios), MAX_TOKENS)
            layer2[name], flags[name] = parse_json(raw, keys)
    else:
        for name, _, _ in LAYER2:
            layer2[name], flags[name] = None, False

    opt_out = None; ok_opt = False
    if all(flags.values()):
        raw = generate(model, tok,
                       build_optimizer(layer2["risk_modeler"],
                                       layer2["income_stability"],
                                       layer2["debt_analyst"],
                                       layer2["reward_modeler"]), MAX_TOKENS)
        opt_out, ok_opt = parse_json(
            raw, ["overall_risk_score", "overall_reward_score", "optimizer_notes"])

    decision = None; orch_out = None; ok_orch = False
    if ok_opt:
        # Decision computed in code; ties resolve to DENY.
        decision = ("APPROVE" if opt_out["overall_reward_score"] >
                    opt_out["overall_risk_score"] else "DENY")
        raw = generate(model, tok, build_orchestrator(opt_out, decision), MAX_TOKENS)
        orch_out, ok_orch = parse_json(raw, ["rationale"])

    return {
        "data_analyst": data_out, "data_analyst_ok": ok_data,
        "contextualizer": ctx_out, "contextualizer_ok": ok_ctx,
        "ratios": ratios,
        **{k: layer2[k] for k in layer2},
        **{f"{k}_ok": flags[k] for k in flags},
        "optimizer": opt_out, "optimizer_ok": ok_opt,
        "orchestrator": orch_out, "orchestrator_ok": ok_orch,
        "final_decision": decision,
        "pipeline_ok": all([ok_data, ok_ctx, *flags.values(), ok_opt, ok_orch]),
    }


def run_dataset(model, tok, key, df, dataset, rung="masca_full"):
    path = checkpoint_path(key, dataset, rung)
    done = completed_ids(path)
    ser, ratio_fn = SERIALIZERS[dataset], RATIOS[dataset]
    print(f"[{key} | {dataset} | {rung}] {len(done)}/{len(df)} done")
    for _, row in df.iterrows():
        aid = row["applicant_id"]
        if aid in done:
            continue
        result = run_one(model, tok, ser(row), ratio_fn(row))
        append(path, {"applicant_id": aid, "model_config": key,
                      "dataset": dataset, "rung": rung, **result})
    print(f"[{key} | {dataset} | {rung}] done")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=MASCA_DATASETS)
    ap.add_argument("--models", nargs="+", default=list(MODEL_REGISTRY))
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    for key in args.models:
        print(f"=== Loading {key} ===")
        model, tok = load_model(key)
        for dataset in args.datasets:
            full = LOADERS[dataset]()
            df = full.head(args.limit) if args.limit else full
            run_dataset(model, tok, key, df, dataset)
        unload(model)
        print(f"=== Unloaded {key} ===")
    print("=== ALL DONE ===")


if __name__ == "__main__":
    main()
