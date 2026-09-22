"""Reproduce every table and figure in the paper from released model outputs.

Usage:
    python src/analysis.py [--checkpoints results/checkpoints]
                           [--out results]

Runs on CPU. Writes CSVs to results/tables/ and PNGs to results/figures/.
"""

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats as st

from serialization import (DATASET_ATTRIBUTES, LOADERS, PRIV_VALUE,
                           build_attr_lookup)
from truncation import truncated_decision

MODELS = ["llama3.1-8b", "mistral-7b", "qwen2.5-7b", "gemma2-9b", "phi4-mini"]
DATASETS = ["german_credit", "taiwan_default", "hmda"]
TITLES = {"german_credit": "German Credit", "taiwan_default": "Taiwan Default",
          "hmda": "HMDA"}
COLORS = {"german_credit": "#27ae60", "taiwan_default": "#8e44ad",
          "hmda": "#e67e22"}
SEED = 42
BOOTSTRAP_B = 1000
LOCUS_MIN_MN = 0.01     # exclude configs with near-zero final shift

CKPT = Path("results/checkpoints")
ATTR = {}


# --------------------------------------------------------------------------
# I/O
# --------------------------------------------------------------------------

def path_for(model, dataset, rung):
    return CKPT / f"{model.replace('/', '_')}__{dataset}__{rung}.jsonl"


def load(model, dataset, rung):
    p = path_for(model, dataset, rung)
    if not p.exists():
        return []
    out = []
    with open(p) as f:
        for line in f:
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


# --------------------------------------------------------------------------
# Fairness metrics
# --------------------------------------------------------------------------

def group_rates(records, attribute, decision_key="final_decision"):
    """Approval rate per group, plus per-group creditworthy outcomes."""
    g = defaultdict(list)
    cw = defaultdict(list)
    for r in records:
        aid = r["applicant_id"]
        d = r.get(decision_key)
        if aid not in ATTR or attribute not in ATTR[aid]:
            continue
        if d not in ("APPROVE", "DENY"):
            continue
        key = ATTR[aid][attribute]
        g[key].append(d == "APPROVE")
        cw[key].append(ATTR[aid]["creditworthy"])
    return g, cw


def metrics(records, attribute, decision_key="final_decision"):
    """SPD is defined whenever both groups are present, including all-deny.
    DI is undefined when the privileged approval rate is zero."""
    g, cw = group_rates(records, attribute, decision_key)
    priv = PRIV_VALUE[attribute]
    others = [k for k in g if k != priv]
    if priv not in g or not others:
        return None
    unpriv = others[0]
    p_priv, p_unpriv = np.mean(g[priv]), np.mean(g[unpriv])

    def tpr(key):
        pos = [d for d, c in zip(g[key], cw[key]) if c]
        return np.mean(pos) if pos else None

    t_priv, t_unpriv = tpr(priv), tpr(unpriv)
    return {
        "SPD": p_unpriv - p_priv,
        "DI": (p_unpriv / p_priv) if p_priv > 0 else None,
        "EOD": (t_unpriv - t_priv) if (t_priv is not None and t_unpriv is not None) else None,
        "approval_rate": np.mean(g[priv] + g[unpriv]),
    }


# --------------------------------------------------------------------------
# Table 1: baseline ladder
# --------------------------------------------------------------------------

RUNGS = [
    ("zero_shot", "decision", "parsed_ok"),
    ("few_shot", "decision", "parsed_ok"),
    ("cot", "decision", "parsed_ok"),
    ("single_agent_multitask", "final_decision", "pipeline_ok"),
    ("masca_full", "final_decision", "pipeline_ok"),
    ("pipeline_homogeneous", "final_decision", "pipeline_ok"),
    ("pipeline_heterogeneous", "final_decision", "pipeline_ok"),
]


def table_ladder():
    rows = []
    for ds in DATASETS:
        for rung, dkey, okkey in RUNGS:
            models = ["heterogeneous"] if rung == "pipeline_heterogeneous" else MODELS
            for m in models:
                recs = load(m, ds, rung)
                if not recs:
                    continue
                decs = [r.get(dkey) for r in recs]
                rows.append({
                    "dataset": ds, "rung": rung, "model": m, "n": len(recs),
                    "ok_pct": 100 * sum(bool(r.get(okkey)) for r in recs) / len(recs),
                    "approve_pct": 100 * sum(d == "APPROVE" for d in decs) / len(recs),
                })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Table 2: sensitivity and signed disparity change
# --------------------------------------------------------------------------

CONDS = [("sc_c", "single_agent_multitask"), ("sc_p", "single_call_perturbed"),
         ("pl_c", "pipeline_homogeneous"), ("pl_p", "pipeline_perturbed")]


def table_sensitivity():
    rows = []
    for ds in DATASETS:
        for m in MODELS:
            R = {k: load(m, ds, rung) for k, rung in CONDS}
            for a in DATASET_ATTRIBUTES[ds]:
                M = {k: metrics(v, a) for k, v in R.items()}
                if any(v is None for v in M.values()):
                    continue
                s = {k: M[k]["SPD"] for k in M}
                rows.append({
                    "dataset": ds, "model": m, "attribute": a,
                    "sens_single": abs(s["sc_p"] - s["sc_c"]),
                    "sens_cafa": abs(s["pl_p"] - s["pl_c"]),
                    "disp_change_single": abs(s["sc_p"]) - abs(s["sc_c"]),
                    "disp_change_cafa": abs(s["pl_p"]) - abs(s["pl_c"]),
                })
    df = pd.DataFrame(rows)
    df["amplified"] = df["sens_cafa"] > df["sens_single"]
    return df


# --------------------------------------------------------------------------
# Tables 3-4: trajectory, coefficients, bias locus
# --------------------------------------------------------------------------

def stage_spd(records, dataset, attribute, stage, alt_pi2=False):
    g = defaultdict(list)
    for r in records:
        aid = r["applicant_id"]
        if aid not in ATTR or attribute not in ATTR[aid]:
            continue
        d = truncated_decision(r, stage, dataset, alt_pi2)
        if d in ("APPROVE", "DENY"):
            g[ATTR[aid][attribute]].append(d == "APPROVE")
    priv = PRIV_VALUE[attribute]
    others = [k for k in g if k != priv]
    if priv not in g or not others:
        return None
    return np.mean(g[others[0]]) - np.mean(g[priv])


def table_trajectory(alt_pi2=False):
    rows = []
    for ds in DATASETS:
        for m in MODELS:
            clean = load(m, ds, "pipeline_homogeneous")
            pert = load(m, ds, "pipeline_perturbed")
            for a in DATASET_ATTRIBUTES[ds]:
                traj, ok = {}, True
                for s in (1, 2, 3, 4):
                    c = stage_spd(clean, ds, a, s, alt_pi2)
                    p = stage_spd(pert, ds, a, s, alt_pi2)
                    if c is None or p is None:
                        ok = False
                        break
                    traj[f"M{s}"] = abs(p - c)
                if ok:
                    rows.append({"dataset": ds, "model": m, "attribute": a, **traj})
    df = pd.DataFrame(rows)
    df["monotone"] = (df.M2 >= df.M1 - 1e-9) & (df.M3 >= df.M2 - 1e-9) & (df.M4 >= df.M3 - 1e-9)
    return df


def coefficients(traj):
    """Theil-Sen transmission coefficients per dataset. Unconstrained slopes."""
    out = {}
    for ds in DATASETS:
        s = traj[traj.dataset == ds]
        a2 = st.theilslopes(s.M2, s.M1)[0] if s.M1.nunique() > 1 else np.nan
        out[ds] = {
            "alpha2": a2,
            "alpha3": st.theilslopes(s.M3, s.M2)[0],
            "alpha4": st.theilslopes(s.M4, s.M3)[0],
        }
    return out


def table_locus(traj, coef):
    """Signed innovation shares; see paper Section on cascade-specific metrics."""
    rows = []
    for _, r in traj.iterrows():
        c = coef[r.dataset]
        if r.M4 < LOCUS_MIN_MN:
            continue
        a2 = 0.0 if np.isnan(c["alpha2"]) else c["alpha2"]
        b1 = r.M1 if not np.isnan(c["alpha2"]) else 0.0
        b2 = r.M2 - a2 * r.M1
        b3 = r.M3 - c["alpha3"] * r.M2
        b4 = r.M4 - c["alpha4"] * r.M3
        rows.append({
            "dataset": r.dataset, "model": r.model, "attribute": r.attribute,
            "Planner": b1 * a2 * c["alpha3"] * c["alpha4"] / r.M4,
            "Risk Analyst": b2 * c["alpha3"] * c["alpha4"] / r.M4,
            "Policy Guard": b3 * c["alpha4"] / r.M4,
            "Writer": b4 / r.M4,
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Paired bootstrap
# --------------------------------------------------------------------------

def bootstrap(B=BOOTSTRAP_B, seed=SEED):
    rng = np.random.default_rng(seed)
    out = []
    for ds in DATASETS:
        ids = np.array([i for i in ATTR if i.startswith(ds.split("_")[0])])
        dec = {(m, k): {r["applicant_id"]: r.get("final_decision")
                        for r in load(m, ds, rung)
                        if r.get("final_decision") in ("APPROVE", "DENY")}
               for m in MODELS for k, rung in CONDS}

        def spd(sample, d, attr):
            g = {True: [], False: []}
            for i in sample:
                v = d.get(i)
                if v is None or attr not in ATTR[i]:
                    continue
                g[ATTR[i][attr] == PRIV_VALUE[attr]].append(v == "APPROVE")
            if not g[True] or not g[False]:
                return np.nan
            return np.mean(g[False]) - np.mean(g[True])

        draws = defaultdict(list)
        for _ in range(B):
            sample = rng.choice(ids, size=len(ids), replace=True)
            acc = defaultdict(list)
            for m in MODELS:
                for a in DATASET_ATTRIBUTES[ds]:
                    v = {k: spd(sample, dec[(m, k)], a) for k, _ in CONDS}
                    if any(np.isnan(x) for x in v.values()):
                        continue
                    acc["sens_single"].append(abs(v["sc_p"] - v["sc_c"]))
                    acc["sens_cafa"].append(abs(v["pl_p"] - v["pl_c"]))
                    acc["disp_cafa"].append(abs(v["pl_p"]) - abs(v["pl_c"]))
            draws["sens_diff"].append(np.mean(acc["sens_cafa"]) - np.mean(acc["sens_single"]))
            draws["disp_cafa"].append(np.mean(acc["disp_cafa"]))

        for q, vals in draws.items():
            lo, hi = np.percentile(vals, [2.5, 97.5])
            out.append({"dataset": ds, "quantity": q,
                        "mean": np.mean(vals), "ci_low": lo, "ci_high": hi})
    return pd.DataFrame(out)


# --------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------

def fig_sensitivity(sens, outdir):
    lvl = sens.groupby(["dataset", "model"])[["sens_single", "sens_cafa"]].mean().reset_index()
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    for ax, ds in zip(axes, DATASETS):
        sub = lvl[lvl.dataset == ds].set_index("model").reindex(MODELS)
        x = np.arange(len(MODELS))
        ax.bar(x - 0.2, sub.sens_single, 0.4, label="Single-call",
               color="#16a085", edgecolor="black", linewidth=0.6)
        ax.bar(x + 0.2, sub.sens_cafa, 0.4, label="CAFA (homogeneous)",
               color="#e74c3c", edgecolor="black", linewidth=0.6)
        ax.set_xticks(x)
        ax.set_xticklabels(MODELS, rotation=30, ha="right", fontsize=9)
        ax.set_title(TITLES[ds])
        ax.set_ylabel("Mean sensitivity |dSPD|")
        ax.tick_params(axis="y", labelleft=True)
        ax.grid(axis="y", linestyle=":", alpha=0.5)
    axes[0].legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(outdir / "sensitivity_by_model.png", dpi=200, bbox_inches="tight")
    plt.close()


def fig_trajectory(traj, outdir):
    fig, ax = plt.subplots(figsize=(7, 5))
    markers = {"german_credit": "o", "taiwan_default": "^", "hmda": "s"}
    for ds in DATASETS:
        s = traj[traj.dataset == ds]
        ax.plot([1, 2, 3, 4], [s.M1.mean(), s.M2.mean(), s.M3.mean(), s.M4.mean()],
                marker=markers[ds], markersize=9, linewidth=2.5,
                color=COLORS[ds], label=TITLES[ds])
    ax.set_xticks([1, 2, 3, 4])
    ax.set_xticklabels(["Planner", "Risk Analyst", "Policy Guard", "Writer"])
    ax.set_ylabel("Mean stage fairness trajectory |dSPD|")
    ax.legend()
    plt.tight_layout()
    plt.savefig(outdir / "stage_trajectory.png", dpi=200, bbox_inches="tight")
    plt.close()


def fig_locus(locus, outdir):
    stages = ["Planner", "Risk Analyst", "Policy Guard", "Writer"]
    med = locus.groupby("dataset")[stages].median()
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(stages))
    for i, ds in enumerate(DATASETS):
        ax.bar(x + (i - 1) * 0.25, med.loc[ds, stages], 0.25,
               label=TITLES[ds], color=COLORS[ds], edgecolor="black", linewidth=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(stages)
    ax.set_ylabel("Median innovation share")
    ax.legend()
    plt.tight_layout()
    plt.savefig(outdir / "bias_locus.png", dpi=200, bbox_inches="tight")
    plt.close()


# --------------------------------------------------------------------------

def main():
    global CKPT, ATTR
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoints", default="results/checkpoints")
    ap.add_argument("--out", default="results")
    ap.add_argument("--skip-bootstrap", action="store_true")
    args = ap.parse_args()

    CKPT = Path(args.checkpoints)
    tables = Path(args.out) / "tables"
    figures = Path(args.out) / "figures"
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)

    print("Loading datasets and building protected-attribute lookup ...")
    ATTR = build_attr_lookup(LOADERS["german_credit"](),
                             LOADERS["taiwan_default"](),
                             LOADERS["hmda"]())
    print(f"  {len(ATTR)} applicants")

    print("Baseline ladder ...")
    ladder = table_ladder()
    ladder.to_csv(tables / "baseline_ladder.csv", index=False)
    print(ladder.groupby(["dataset", "rung"])["approve_pct"].mean().round(1))

    print("\nSensitivity and disparity change ...")
    sens = table_sensitivity()
    sens.to_csv(tables / "sensitivity.csv", index=False)
    print(f"  {len(sens)} configurations, "
          f"{int(sens.amplified.sum())} amplified")
    print(f"  disparity worsened: CAFA {(sens.disp_change_cafa > 0).sum()}, "
          f"single-call {(sens.disp_change_single > 0).sum()}")

    print("\nStage trajectory and coefficients ...")
    traj = table_trajectory()
    traj.to_csv(tables / "trajectory.csv", index=False)
    coef = coefficients(traj)
    pd.DataFrame(coef).T.to_csv(tables / "coefficients.csv")
    print(pd.DataFrame(coef).T.round(3))
    print("  monotone:", traj.groupby("dataset").monotone.agg(["sum", "count"]).to_dict("index"))

    print("\nBias locus ...")
    locus = table_locus(traj, coef)
    locus.to_csv(tables / "bias_locus.csv", index=False)
    stages = ["Planner", "Risk Analyst", "Policy Guard", "Writer"]
    print(locus.groupby("dataset")[stages].median().round(3))

    print("\nAlternative pi_2 sensitivity check ...")
    traj_alt = table_trajectory(alt_pi2=True)
    locus_alt = table_locus(traj_alt, coefficients(traj_alt))
    locus_alt.to_csv(tables / "bias_locus_alt_pi2.csv", index=False)
    print(locus_alt.groupby("dataset")[stages].median().round(3))

    if not args.skip_bootstrap:
        print(f"\nPaired bootstrap (B={BOOTSTRAP_B}) ...")
        ci = bootstrap()
        ci.to_csv(tables / "bootstrap_ci.csv", index=False)
        print(ci.round(3).to_string(index=False))

    print("\nFigures ...")
    fig_sensitivity(sens, figures)
    fig_trajectory(traj, figures)
    fig_locus(locus, figures)

    print(f"\nDone. Tables in {tables}, figures in {figures}.")


if __name__ == "__main__":
    main()
