#!/usr/bin/env python3
"""Compare E0 (no PCGrad) vs E1 (PCGrad) across seeds. Run from the repo root."""
import json, os, statistics as st, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
import three_scores as ts

V3 = "research_results/v3"
full = json.load(open(f"{V3}/tofu_Qwen2.5-0.5B-full_FULLMETRICS/TOFU_EVAL.json"))
retain = json.load(open(f"{V3}/tofu_Qwen2.5-0.5B-retain90_FULLMETRICS/TOFU_EVAL.json"))

def find(paths):
    for p in paths:
        if os.path.exists(p + "/TOFU_EVAL.json"):
            return json.load(open(p + "/TOFU_EVAL.json"))
    return None

def run_dirs(method, seed, step):
    if method == "E0":
        if seed == 0: return [f"{V3}/tofu_V3_E0_diag_checkpoint-{step}"]
        return [f"saves/eval/tofu_V3_E0_diag_s{seed}_checkpoint-{step}",
                f"research_results/v4/e0_seeds/tofu_V3_E0_diag_s{seed}_checkpoint-{step}"]
    if seed == 0: return [f"{V3}/tofu_V3_E1_pcgrad_checkpoint-{step}"]
    return [f"research_results/v4/e1_seeds/tofu_V3_E1_pcgrad_s{seed}_checkpoint-{step}",
            f"saves/eval/tofu_V3_E1_pcgrad_s{seed}_checkpoint-{step}"]

def metrics(d):
    s = ts.score(d, full, retain)
    return {"PrivLeak": ts.agg(d, "privleak"), "MU": ts.agg(d, "model_utility"),
            "retProb": ts.agg(d, "retain_Q_A_Prob"), "FQ": ts.agg(d, "forget_quality"),
            "utility": s["utility"], "privacy": s["privacy"]}

cols = ["PrivLeak", "MU", "retProb", "FQ", "utility", "privacy"]
for step in (26, 39, 65):
    print(f"\n=== step {step}")
    print(f"{'run':<14}" + "".join(f"{c:>10}" for c in cols))
    summary = {}
    for method in ("E0", "E1"):
        vals = {c: [] for c in cols}
        for seed in (0, 1, 2):
            d = find(run_dirs(method, seed, step))
            if d is None:
                print(f"{method} seed{seed:<7}" + "   (missing)")
                continue
            m = metrics(d)
            print(f"{method} seed{seed:<7}" + "".join(f"{m[c]:10.3f}" for c in cols))
            for c in cols: vals[c].append(m[c])
        n = len(vals["MU"])
        if n >= 2:
            print(f"{method + ' mean':<14}" + "".join(f"{st.mean(vals[c]):10.3f}" for c in cols))
            print(f"{method + ' sd':<14}" + "".join(f"{st.stdev(vals[c]):10.3f}" for c in cols))
        summary[method] = (n, vals)
    if all(summary[m][0] >= 2 for m in ("E0", "E1")):
        gap = st.mean(summary["E1"][1]["MU"]) - st.mean(summary["E0"][1]["MU"])
        sdp = (((summary["E0"][0]-1)*st.variance(summary["E0"][1]["MU"]) + (summary["E1"][0]-1)*st.variance(summary["E1"][1]["MU"]))
               / (summary["E0"][0] + summary["E1"][0] - 2)) ** 0.5
        print(f"MU gap E1-E0 = {gap:+.3f}; pooled seed sd = {sdp:.3f}; gap/sd = {gap/sdp if sdp else float('nan'):.1f} (rough guide only)")
