# V4 ablations and seed study

Base method: E1 (CustomV3 with `grad_mode=pcgrad`), Qwen2.5-0.5B-Instruct, TOFU forget10 / retain90 / holdout10, 65 steps (13 per epoch). Raw JSONs: `research_results/v3` (E1 seed 0) and `research_results/v4`.

| Run | Change vs E1 | Seeds |
|---|---|---|
| E1 | none | 0, 1, 2 |
| V4a | forget weight gamma 1.0 -> 0.5 | 0 |
| V4b | retain NLL weight alpha 1.0 -> 2.0 | 0 |

PrivLeak target is 0. Retain-model MU is 0.340. sd = sample sd over 3 seeds.

| Run | PrivLeak @26 | PrivLeak @65 | FQ(KS) @65 | MU @65 | retain Q_A prob @65 |
|---|---|---|---|---|---|
| E1 seed 0 | 3.06 | 9.94 | 0.907 | 0.314 | 0.134 |
| E1 seed 1 | -9.06 | 3.63 | 0.813 | 0.297 | 0.152 |
| E1 seed 2 | -7.81 | 3.12 | 0.942 | 0.296 | 0.156 |
| E1 mean ± sd | -4.60 ± 6.67 | 5.56 ± 3.80 | 0.887 ± 0.066 | 0.303 ± 0.010 | 0.147 ± 0.012 |
| V4a (seed 0) | 2.48 | 9.23 | 0.758 | 0.325 | 0.158 |
| V4b (seed 0) | 1.67 | 9.31 | 0.864 | 0.322 | 0.153 |

## Findings
- gamma 0.5 and alpha 2.0 moved PrivLeak by under 1 point at matched seed, against a seed sd of 3.8 (step 65) and 6.7 (step 26). No effect detectable.
- MU rose by 0.008-0.011 at matched seed; seed sd is 0.010. Not distinguishable from noise at this sample size.
- The step at which PrivLeak crosses 0 depends on the seed (between steps 13 and 26 for seed 0, near step 39 for seeds 1 and 2). A fixed stopping step is not robust. PrivLeak at step 65 is positive in all three seeds.
- The utility cost of unlearning appears in every seed: MU 0.339 at step 13 falls to 0.296-0.314 at step 65, below the retain model.
- `max_grad_norm` is the framework default 1.0 and is not overridden. Logged pre-clip gradient norms were 65-400, so every logged step was clipped.

## Limits
Seeds change training randomness only; forget and holdout examples are the same in every run. V4a and V4b have one seed each. E0 and E2 have one seed each. PrivLeak is computed against the retain90 model's MIA AUC (min_k 0.378); the stored retain-model PrivLeak (+24.4) was produced without that reference and is not meaningful.

## Next
Done: E0 seeds 1 and 2 were run (see docs/EVOLUTION_REPORT.md). Next: lower learning rate on E1, three seeds.
