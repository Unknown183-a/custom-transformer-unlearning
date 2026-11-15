python - <<'PY'
import json,glob,os
need=["model_utility","privleak","mia_loss","mia_zlib","mia_min_k","mia_min_k_plus_plus","exact_memorization","extraction_strength","forget_Q_A_gibberish","forget_quality"]
for t in ["tofu_V3_E0_diag","tofu_V3_E1_pcgrad","tofu_V3_E2_simnpo"]:
    for ck in sorted(glob.glob(f"saves/unlearn/{t}/checkpoint-*"), key=lambda p:int(p.split('-')[-1])):
        n=f"{t}_{os.path.basename(ck)}"; f=f"saves/eval/{n}/TOFU_EVAL.json"
        if not os.path.exists(f): print("NOT STARTED",n); continue
        try: d=json.load(open(f))
        except Exception: print("UNREADABLE",n); continue
        miss=[k for k in need if k not in d]
        print(("DONE    " if not miss else "PARTIAL "),n,"" if not miss else "missing: "+",".join(miss))
PY
