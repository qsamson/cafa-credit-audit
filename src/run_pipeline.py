"""Rungs 6-7: CAFA homogeneous and heterogeneous, clean and perturbed.

Usage:
    python src/run_pipeline.py
    python src/run_pipeline.py --config homogeneous --datasets hmda
    python src/run_pipeline.py --config heterogeneous --limit 2
"""

import argparse

import pandas as pd

from common import (append, checkpoint_path, completed_ids, generate,
                    load_model, normalize, parse_json, unload, MODEL_REGISTRY)
from prompts_py import (POLICY_CONDITIONS, build_planner, build_policy_guard,
                        build_risk_analyst, build_writer)
from serialization import DATASET_ATTRIBUTES, LOADERS, PERTURBERS, SERIALIZERS

ROLES = ["planner", "risk_analyst", "policy_guard", "writer"]
POLICY_KEYS = ["condition_a_present", "condition_b_present",
               "condition_c_present", "policy_notes"]
MAX_TOKENS = 250


def role_assignment(idx, model_keys):
    """Round-robin, so a role is not confounded with one model's quirks."""
    n = len(model_keys)
    return {role: model_keys[(idx + j) % n] for j, role in enumerate(ROLES)}


def run_one(models, assignment, applicant_text, conditions):
    """Execute the four stages. Returns per-stage outputs and ok flags."""
    pm, pt = models[assignment["planner"]]
    rm, rt = models[assignment["risk_analyst"]]
    gm, gt = models[assignment["policy_guard"]]
    wm, wt = models[assignment["writer"]]

    raw1 = generate(pm, pt, build_planner(applicant_text), MAX_TOKENS)
    s1, ok1 = parse_json(raw1, ["relevant_factors"])

    s2 = None; ok2 = False
    if ok1:
        raw2 = generate(rm, rt, build_risk_analyst(s1), MAX_TOKENS)
        s2, ok2 = parse_json(raw2, ["risk_level", "risk_summary"])

    s3 = None; ok3 = False
    if ok2:
        raw3 = generate(gm, gt, build_policy_guard(s2, conditions), MAX_TOKENS)
        s3, ok3 = parse_json(raw3, POLICY_KEYS)
        if ok3:
            # Decision is computed in code, never written by the model.
            any_cond = (s3["condition_a_present"] or s3["condition_b_present"]
                        or s3["condition_c_present"])
            s3["decision"] = "DENY" if any_cond else "APPROVE"

    s4 = None; ok4 = False
    if ok3:
        raw4 = generate(wm, wt, build_writer(s3, s2), MAX_TOKENS)
        s4, ok4 = parse_json(raw4, ["rationale"])

    return {
        "stage1_parsed": s1, "stage1_ok": ok1,
        "stage2_parsed": s2, "stage2_ok": ok2,
        "stage3_parsed": s3, "stage3_ok": ok3,
        "stage4_parsed": s4, "stage4_ok": ok4,
        "final_decision": normalize(s3["decision"]) if ok3 else None,
        "pipeline_ok": ok1 and ok2 and ok3 and ok4,
    }


def run_rung(models, model_keys, df, dataset, rung, config):
    ckpt_key = "heterogeneous" if config == "heterogeneous" else model_keys[0]
    path = checkpoint_path(ckpt_key, dataset, rung)
    done = completed_ids(path)
    ser = SERIALIZERS[dataset]
    conditions = POLICY_CONDITIONS[dataset]
    print(f"[{ckpt_key} | {dataset} | {rung}] {len(done)}/{len(df)} done")

    for idx, (_, row) in enumerate(df.iterrows()):
        aid = row["applicant_id"]
        if aid in done:
            continue
        if config == "heterogeneous":
            assignment = role_assignment(idx, model_keys)
        else:
            assignment = {r: model_keys[0] for r in ROLES}
        result = run_one(models, assignment, ser(row), conditions)
        record = {"applicant_id": aid, "model_config": ckpt_key,
                  "dataset": dataset, "rung": rung, **result}
        if config == "heterogeneous":
            record["role_assignment"] = assignment
        append(path, record)
    print(f"[{ckpt_key} | {dataset} | {rung}] done")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", choices=["homogeneous", "heterogeneous", "both"],
                    default="both")
    ap.add_argument("--datasets", nargs="+", default=list(DATASET_ATTRIBUTES))
    ap.add_argument("--models", nargs="+", default=list(MODEL_REGISTRY))
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--clean-only", action="store_true")
    args = ap.parse_args()

    def frames(dataset):
        full = LOADERS[dataset]()
        df = full.head(args.limit) if args.limit else full
        pert = pd.DataFrame([PERTURBERS[dataset](r) for _, r in df.iterrows()])
        return df, pert

    # Homogeneous: one model at a time, so only one sits in memory.
    if args.config in ("homogeneous", "both"):
        for key in args.models:
            print(f"=== Loading {key} ===")
            models = {key: load_model(key)}
            for dataset in args.datasets:
                clean, pert = frames(dataset)
                run_rung(models, [key], clean, dataset,
                         "pipeline_homogeneous", "homogeneous")
                if not args.clean_only:
                    run_rung(models, [key], pert, dataset,
                             "pipeline_perturbed", "homogeneous")
            unload(models[key][0])
            print(f"=== Unloaded {key} ===")

    # Heterogeneous: all models resident at once for round-robin assignment.
    if args.config in ("heterogeneous", "both"):
        print("=== Loading all models for heterogeneous configuration ===")
        models = {k: load_model(k) for k in args.models}
        for dataset in args.datasets:
            clean, pert = frames(dataset)
            run_rung(models, args.models, clean, dataset,
                     "pipeline_heterogeneous", "heterogeneous")
            if not args.clean_only:
                run_rung(models, args.models, pert, dataset,
                         "pipeline_heterogeneous_perturbed", "heterogeneous")
        for k in args.models:
            unload(models[k][0])

    print("=== ALL DONE ===")


if __name__ == "__main__":
    main()
