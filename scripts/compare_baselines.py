#!/usr/bin/env python3
"""E0 / E1 / V8 versus the framework's NPO and GradDiff baselines at checkpoint 65. Run from the repo root."""
import json, os, statistics as st, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import three_scores as ts

V3 = "research_results/v3"
full = json.load(open(f"{V3}/tofu_Qwen2.5-0.5B-full_FULLMETRICS/TOFU_EVAL.json"))
retain = json.load(open(f"{V3}/tofu_Qwen2.5-0.5B-retain90_FULLMETRICS/TOFU_EVAL.json"))
E = "/TOFU_EVAL.json"
groups = {
    "E0 (no PCGrad)": [f"{V3}/tofu_V3_E0_diag_checkpoint-65"] + [f"research_results/v4/e0_seeds/tofu_V3_E0_diag_s{s}_checkpoint-65" for s in (1, 2)],
    "E1 (PCGrad)": [f"{V3}/tofu_V3_E1_pcgrad_checkpoint-65"] + [f"research_results/v4/e1_seeds/tofu_V3_E1_pcgrad_s{s}_checkpoint-65" for s in (1, 2)],
    "V8 (PCGrad, tuned)": [f"research_results/v8/tofu_V8_pcgrad_kl0.5_lr8e-6_s{s}_checkpoint-65" for s in (0, 1, 2)],
    "B1 NPO": [f"saves/eval/tofu_B1_npo_s{s}_checkpoint-65" for s in (0, 1, 2)],
    "B2 GradDiff": [f"saves/eval/tofu_B2_graddiff_s{s}_checkpoint-65" for s in (0, 1, 2)],
}
cols = [("privleak", "PrivLeak"), ("model_utility", "MU"), ("retain_Q_A_Prob", "retProb"), ("forget_quality", "FQ")]
rows = {}
for g, ps in groups.items():
    r = []
    for p in ps:
        if not os.path.exists(p + E):
            print(f"{g}: missing {p}"); continue
        d = json.load(open(p + E))
        if "privleak" not in d or "model_utility" not in d:
            print(f"{g}: incomplete {p}"); continue
        sc = ts.score(d, full, retain)
        r.append([ts.agg(d, k) for k, _ in cols] + [sc["utility"], sc["privacy"]])
    if r: rows[g] = r
names = [l for _, l in cols] + ["utilScore", "privScore"]
print("\n" + "group".ljust(22) + "n " + "".join(n.rjust(17) for n in names))
for g, r in rows.items():
    cells = []
    for c in zip(*r):
        cells.append((f"{st.mean(c):7.3f} ± {st.stdev(c):5.3f}" if len(c) > 1 else f"{st.mean(c):7.3f}        ").rjust(17))
    print(g.ljust(22) + f"{len(r)} " + "".join(cells))
print("\nPer seed (PrivLeak, MU, retProb, FQ, utilScore, privScore):")
for g, r in rows.items():
    for i, x in enumerate(r): print(f"  {g:<22} #{i}", *[f"{v:8.3f}" for v in x])
if "E1 (PCGrad)" in rows:
    for g in ("B1 NPO", "B2 GradDiff"):
        if g in rows:
            for j, k in ((1, "model utility"), (2, "retain-set probability")):
                e1 = [x[j] for x in rows["E1 (PCGrad)"]]; b = [x[j] for x in rows[g]]
                print(f"every E1 seed above every {g} seed on {k}? {min(e1) > max(b)}")
