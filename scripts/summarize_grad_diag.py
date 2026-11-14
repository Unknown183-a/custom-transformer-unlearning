#!/usr/bin/env python3
"""Summarise grad_diag.jsonl written by CustomV3 (diag_every>0).
usage: summarize_grad_diag.py path/to/grad_diag.jsonl
Reports norm imbalance and conflict (cos<0) globally, over training windows, and per layer.
Interpretation guide (README V3): cos<0 => conflict; cos~0 => orthogonal; big norm ratio => one term dominates."""
import json, statistics as st, sys
from collections import defaultdict

rows = [json.loads(l) for l in open(sys.argv[1])]
print(f"{len(rows)} measurements, params measured: {rows[0]['n_params_measured']:,}")
cos = [r["cos"] for r in rows]; ratio = [r["norm_ratio_F_over_R"] for r in rows]
print(f"global cos      mean {st.mean(cos):+.4f}  median {st.median(cos):+.4f}  frac<0 {sum(c<0 for c in cos)/len(cos):.2f}")
print(f"||gF||/||gR||   mean {st.mean(ratio):.3f}  median {st.median(ratio):.3f}  min {min(ratio):.3f}  max {max(ratio):.3f}")
n = len(rows); k = max(1, n // 5)
print("\nwindow(step range)        cos_mean   frac<0   ratio_median")
for i in range(0, n, k):
    w = rows[i:i + k]
    print(f"{w[0]['step']:>5}-{w[-1]['step']:<5}            {st.mean(r['cos'] for r in w):+.4f}   {sum(r['cos']<0 for r in w)/len(w):.2f}     {st.median(r['norm_ratio_F_over_R'] for r in w):.3f}")
if "layers" in rows[0]:
    L = defaultdict(list)
    for r in rows:
        for name, v in r["layers"].items():
            L[name].append((v["cos"], v["norm_F"] / (v["norm_R"] + 1e-12)))
    print("\nlayer        cos_mean   frac<0   normF/normR(median)")
    for name in sorted(L):
        c = [x[0] for x in L[name]]; q = [x[1] for x in L[name]]
        print(f"{name:<11}  {st.mean(c):+.4f}   {sum(x<0 for x in c)/len(c):.2f}     {st.median(q):.3f}")
