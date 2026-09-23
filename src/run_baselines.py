"""Rungs 1-4: zero-shot, few-shot, chain-of-thought, single-agent multitask.

Usage:
    python src/run_baselines.py
    python src/run_baselines.py --datasets hmda --models phi4-mini --limit 2
"""

import argparse

import pandas as pd

from common import (append, checkpoint_path, completed_ids, generate,
                    load_model, normalize, parse_json, parse_single_call,
                    unload, MODEL_REGISTRY)
from prompts_py import (POLICY_CONDITIONS, build_cot, build_few_shot,
                        build_multitask, build_zero_shot)
from serialization import (DATASET_ATTRIBUTES, LOADERS, PERTURBERS,
                           SERIALIZERS)

MULTITASK_KEYS = ["relevant_factors", "risk_level", "risk_summary",
                  "condition_a_present", "condition_b_present",
                  "condition_c_present", "policy_notes", "rationale"]


def few_shot_exemplars(df, dataset, n_each=3, seed=0):
    ser = SERIALIZERS[dataset]
    label_col = {"german_credit": "target", "taiwan_default": "target",
                 "hmda": "action_taken"}[dataset]
    good = {"german_credit": 1, "taiwan_default": 0, "hmda": 1}[dataset]
    approved = df[df[label_col] == good].sample(n=n_each, random_state=seed)
    denied = df[df[label_col] != good].sample(n=n_each, random_state=seed)
    out = []
    for _, row in pd.concat([denied, approved]).iterrows():
        out.append((ser(row), "APPROVE" if row[label_col] == good else "DENY"))
    return out


def run_single_call(model, tok, key, df, dataset, rung, builder,
                    exemplars=None, max_new_tokens=300):
    path = checkpoint_path(key, dataset, rung)
    done = completed_ids(path)
    ser = SERIALIZERS[dataset]
    print(f"[{key} | {dataset} | {rung}] {len(done)}/{len(df)} done")
    for _, row in df.iterrows():
        if row["applicant_id"] in done:
            continue
        text = ser(row)
        prompt = builder(text, exemplars) if exemplars else builder(text)
        raw = generate(model, tok, prompt, max_new_tokens)
        parsed = parse_single_call(raw)
        append(path, {"applicant_id": row["applicant_id"], "model": key,
                      "dataset": dataset, "rung": rung, "raw_output": raw,
                      **parsed, "final_decision": parsed["decision"]})


def run_multitask(model, tok, key, df, dataset, rung="single_agent_multitask",
                  max_new_tokens=400):
    path = checkpoint_path(key, dataset, rung)
    done = completed_ids(path)
    ser = SERIALIZERS[dataset]
    print(f"[{key} | {dataset} | {rung}] {len(done)}/{len(df)} done")
    for _, row in df.iterrows():
        if row["applicant_id"] in done:
            continue
        raw = generate(model, tok,
                       build_multitask(ser(row), POLICY_CONDITIONS[dataset]),
                       max_new_tokens)
        parsed, ok = parse_json(raw, MULTITASK_KEYS)
        if ok:
            any_cond = (parsed["condition_a_present"] or
                        parsed["condition_b_present"] or
                        parsed["condition_c_present"])
            parsed["decision"] = "DENY" if any_cond else "APPROVE"
        append(path, {"applicant_id": row["applicant_id"], "model": key,
                      "dataset": dataset, "rung": rung, "parsed": parsed,
                      "ok": ok, "pipeline_ok": ok,
                      "final_decision": normalize(parsed["decision"]) if ok else None})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=list(DATASET_ATTRIBUTES))
    ap.add_argument("--models", nargs="+", default=list(MODEL_REGISTRY))
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--perturbed", action="store_true",
                    help="run the multitask rung on perturbed profiles")
    args = ap.parse_args()

    for key in args.models:
        print(f"=== Loading {key} ===")
        model, tok = load_model(key)
        for dataset in args.datasets:
            full = LOADERS[dataset]()
            df = full.head(args.limit) if args.limit else full

            if args.perturbed:
                pert = pd.DataFrame([PERTURBERS[dataset](r)
                                     for _, r in df.iterrows()])
                run_multitask(model, tok, key, pert, dataset,
                              rung="single_call_perturbed")
                continue

            run_single_call(model, tok, key, df, dataset, "zero_shot",
                            build_zero_shot)
            run_single_call(model, tok, key, df, dataset, "few_shot",
                            build_few_shot,
                            exemplars=few_shot_exemplars(full, dataset))
            run_single_call(model, tok, key, df, dataset, "cot",
                            build_cot, max_new_tokens=700)
            run_multitask(model, tok, key, df, dataset)
        unload(model)
        print(f"=== Unloaded {key} ===")
    print("=== ALL DONE ===")


if __name__ == "__main__":
    main()
