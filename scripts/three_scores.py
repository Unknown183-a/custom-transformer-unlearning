#!/usr/bin/env python3
"""Compute forgetting / privacy / utility scores from OpenUnlearning TOFU_EVAL.json files,
following the aggregation in OpenUnlearning paper Appendix F.1 (as far as the paper text specifies).

  Memorization(=forgetting) = HM(1-ES, 1-EM, 1-ParaProb, 1-TruthRatio_knowledge)
  Privacy                   = HM over MIA {loss, zlib, min_k, min_k++} of closeness to the retain model
  Utility                   = HM(MU_run/MU_full, Fluency_run/Fluency_full)   (each capped at 1)

PROXIES / ASSUMPTIONS (verify before reporting):
  * The paper's exact per-MIA closeness formula is not in the text we have. We use the PrivLeak-style
    relative difference of (1-AUC):  s = max(0, 1 - |(1-auc_run)-(1-auc_ret)| / (1-auc_ret)).
  * Truth-ratio knowledge form p_para/(p_para+p_pert) is computed per example from value_by_index,
    using the MEAN of the perturbed-answer probabilities.
  * Fluency = forget_Q_A_gibberish agg_value (assumed: higher = more fluent).
  * 'closeness_to_retain' columns are our additional view: forgetting is not rewarded for going past the retain model.
Missing metrics are skipped and the score is flagged PARTIAL.

usage: three_scores.py --full F.json --retain R.json  name1=run1.json name2=run2.json ... [--csv out.csv]
"""
import argparse, csv, json, os, sys


def hm(vals):
    vals = [v for v in vals if v is not None]
    if not vals:
        return None
    if any(v <= 1e-12 for v in vals):
        return 0.0
    return len(vals) / sum(1.0 / v for v in vals)


def agg(d, k):
    return d[k]["agg_value"] if k in d and isinstance(d[k], dict) and "agg_value" in d[k] else None


def truth_ratio_knowledge(d):
    try:
        para = d["forget_Q_A_PARA_Prob"]["value_by_index"]
        pert = d["forget_Q_A_PERT_Prob"]["value_by_index"]
        rs = []
        for k in para:
            p = para[k]["prob"]
            q = pert[k]["prob"]
            q = sum(q) / len(q) if isinstance(q, list) else q
            rs.append(p / (p + q + 1e-12))
        return sum(rs) / len(rs)
    except Exception:
        return None


def score(run, full, retain):
    out, notes = {}, []
    es, em, pp, tr = agg(run, "extraction_strength"), agg(run, "exact_memorization"), agg(run, "forget_Q_A_PARA_Prob"), truth_ratio_knowledge(run)
    parts = {"ES": es, "EM": em, "ParaProb": pp, "TR": tr}
    miss = [k for k, v in parts.items() if v is None]
    if miss:
        notes.append("mem missing " + ",".join(miss))
    out["forgetting"] = hm([None if v is None else 1 - v for v in parts.values()])

    privs = []
    if retain is None:
        notes.append("privacy needs --retain")
    else:
        for m in ("mia_loss", "mia_zlib", "mia_min_k", "mia_min_k_plus_plus"):
            a, r = agg(run, m), agg(retain, m)
            if a is None or r is None:
                notes.append(f"{m} missing")
                continue
            privs.append(max(0.0, 1 - abs((1 - a) - (1 - r)) / (1 - r + 1e-12)))
    out["privacy"] = hm(privs)

    mu, mu0 = agg(run, "model_utility"), agg(full, "model_utility")
    fl, fl0 = agg(run, "forget_Q_A_gibberish"), agg(full, "forget_Q_A_gibberish")
    u = [min(1.0, mu / mu0)] if mu is not None and mu0 else []
    if fl is not None and fl0:
        u.append(min(1.0, fl / fl0))
    else:
        notes.append("fluency missing")
    out["utility"] = hm(u)
    out["forget_quality_KS"] = agg(run, "forget_quality")
    out["MU_raw"] = mu
    out["notes"] = "; ".join(notes)
    out["PARTIAL"] = bool(notes)
    vals = [out[k] for k in ("forgetting", "privacy", "utility") if out[k] is not None]
    out["HM3"] = hm(vals) if len(vals) == 3 else None
    out["min3"] = min(vals) if len(vals) == 3 else None
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", required=True)
    ap.add_argument("--retain", default=None)
    ap.add_argument("--csv", default=None)
    ap.add_argument("runs", nargs="+", help="name=path/to/TOFU_EVAL.json")
    a = ap.parse_args()
    full = json.load(open(a.full))
    retain = json.load(open(a.retain)) if a.retain else None
    rows = []
    for spec in a.runs:
        name, path = spec.split("=", 1)
        s = score(json.load(open(path)), full, retain)
        s["name"] = name
        rows.append(s)
    f = lambda x: "  -  " if x is None else f"{x:.3f}"
    print(f"{'run':<28}{'forget':>8}{'privacy':>9}{'utility':>9}{'HM3':>7}{'min3':>7}{'FQ(KS)':>8}{'MU_raw':>8}  notes")
    for r in rows:
        print(f"{r['name']:<28}{f(r['forgetting']):>8}{f(r['privacy']):>9}{f(r['utility']):>9}{f(r['HM3']):>7}{f(r['min3']):>7}{f(r['forget_quality_KS']):>8}{f(r['MU_raw']):>8}  {r['notes']}")
    if a.csv:
        new = not os.path.exists(a.csv)
        with open(a.csv, "a", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            if new:
                w.writeheader()
            w.writerows(rows)


if __name__ == "__main__":
    main()
