# Unlearning Algorithm Evolution: V1 → V9

**Model:** Qwen2.5-0.5B-Instruct  **Benchmark:** TOFU (forget10 / retain90 / holdout10)  **Framework:** OpenUnlearning
**Status:** working draft, updated 17 November 2026. E0, E1, V6 (lr 8e-6), V7, V8 and V9 have 3 seeds each. Anything not marked "3 seeds" is **single-seed**.

---

## 1. Goal and setup

**Goal.** A hybrid unlearning algorithm that is *stable across forgetting, privacy and utility*, instead of strong on one and weak on another. **Method.** Change one mechanism at a time, measure with a fixed evaluation, keep what helps.

| Item | Value |
|---|---|
| Target model | Qwen2.5-0.5B-Instruct fine-tuned on TOFU (`checkpoint-625`) |
| Reference model | retain90-only model (gold standard) |
| Data | forget10 (400 QA pairs), retain90, holdout10 (for MIA) |
| Optimisation | lr 1e-5, weight decay 0.01, 1 warmup epoch, 5 epochs, `max_grad_norm` 1.0 (framework default) |
| Steps | V3 to V9 runs: 65 optimizer steps in total (13 per epoch). V3/V4 evaluated at checkpoints 13, 26, 39, 52, 65; V5 to V9 at 26, 39 and 65 only |
| Loss weights | NPO β = 0.1, γ = 1, α = 1, KL weight λ = 0.1 (changed only where stated) |
| Evaluation | V1/V2: default OpenUnlearning TOFU metrics. V3 to V9: full set (Exact Memorization, 4 MIAs, fluency) with the same retain-model reference JSON for every run |

---

## 2. Evolution at a glance

| Version | What changed | Why | Outcome |
|---|---|---|---|
| **V1** | NPO forgetting + retain NLL + retain KL (unmasked) | First working custom objective | Forget probability 0.242 → 0.071; model utility 0.338 → 0.269 |
| **V2** | KL computed on answer tokens only | Padding should not contribute to the KL anchor | Utility 0.2696 vs 0.2694 (negligible) |
| **V3-E0** | Same objective as V2; adds gradient diagnostics; reuses the retain forward pass | Measure whether forget and retain gradients conflict; run faster | Reproduces V2 (utility 0.2697, extraction strength 0.02995 in both) |
| **V3-E1** | PCGrad-style update (remove the part of the forget gradient that opposes the retain gradient) | Diagnostics showed strong conflict (Section 5) | **3 seeds: model utility 0.252 ± 0.016 → 0.303 ± 0.010; retain-set probability 0.084 → 0.147**; PrivLeak unchanged within noise |
| **V3-E2** | SimNPO-style length-normalised forget loss (γ 0.125, β 4.5, δ 0) | Test a mechanism from the best-balanced method in the OpenUnlearning paper | Does not unlearn: results equal the full model |
| **V4a / V4b** | E1 with forget weight γ 0.5 / retain NLL weight α 2.0 | One-change ablations on E1 | No effect detectable against seed noise |
| **V5** | E1 with peak learning rate 1e-5 → 5e-6 (seed 0) | Test the utility cost and privacy overshoot through the learning rate | Barely unlearns: PrivLeak −12.6 (full model −16.5), utility 0.337; **single seed** |
| **V6** | E1 with peak learning rate 7e-6 (seed 0) and 8e-6 (3 seeds) | Look for a rate between "too little" and "too much" forgetting | lr 8e-6, 3 seeds: PrivLeak −3.6 ± 3.9 (E1: +5.6 ± 3.8), utility 0.296 ± 0.015 (E1: 0.303 ± 0.010) |
| **V7** | E1 with KL weight λ 0.1 → 0.5 (lr 1e-5, 3 seeds) | Test whether a stronger KL anchor protects retain-set ability | 3 seeds: PrivLeak +7.6 ± 4.5, utility 0.319 ± 0.005, retain prob 0.167 ± 0.011 (E1: +5.6 ± 3.8, 0.303 ± 0.010, 0.147 ± 0.012). Higher retain-set probability and slightly higher utility; PrivLeak not lower |
| **V8** | E1 with KL weight 0.5 **and** lr 8e-6 (two changes together) | Combine the two settings that looked useful | 3 seeds: PrivLeak −3.2 ± 3.6, utility 0.308 ± 0.012, retain prob 0.160 ± 0.001. **Not distinguishable from E1 on utility or privacy**; retain-set probability about +0.013 |
| **V9** | V8 setting with PCGrad switched off (plain summed objective) | Ablation: is PCGrad still the active ingredient at the tuned setting? | 3 seeds: utility 0.246 ± 0.019, retain prob 0.083 ± 0.003. **Every V8 seed beats every V9 seed on utility and retain probability** |
| *Planned* | SimNPO with δ > 0; external baselines (NPO, GradDiff) under the same protocol | Sections 6 and 8 | Not yet run |

---

## 3. Mathematical formulation

Notation: $x$ question, $y$ answer, $\pi_\theta$ the model being unlearned, $\pi_{\text{ref}}$ the frozen original model, $D_f$ forget batch, $D_r$ retain batch.

### 3.1 V1 and V2 (same objective, different KL mask)

$$
\mathcal{L} = \gamma\,\mathcal{L}_{\text{NPO}} + \alpha\,\mathcal{L}_{\text{NLL}} + \lambda\,\mathcal{L}_{\text{KL}}
$$

$$
\mathcal{L}_{\text{NPO}} = -\frac{2}{\beta}\,\mathbb{E}_{(x,y)\sim D_f}\Big[\log\sigma\Big(-\beta\,\log\frac{\pi_\theta(y\mid x)}{\pi_{\text{ref}}(y\mid x)}\Big)\Big]
$$

$$
\mathcal{L}_{\text{NLL}} = \mathbb{E}_{(x,y)\sim D_r}\Big[-\frac{1}{|y|}\sum_t \log \pi_\theta(y_t\mid x,y_{<t})\Big]
$$

$$
\mathcal{L}_{\text{KL}} = \mathbb{E}_{D_r}\Big[\frac{1}{|M|}\sum_{t\in M} \mathrm{KL}\big(\pi_{\text{ref}}(\cdot\mid c_t)\,\|\,\pi_\theta(\cdot\mid c_t)\big)\Big]
$$

The NPO term uses sequence log-probabilities. **V1** averages the KL over all positions. **V2** restricts it to the set $M$ of answer tokens (label ≠ −100).

### 3.2 V3-E0 (diagnostic)

Same loss as V2. Every 5 optimizer steps it additionally records, over all 494 M parameters and per layer:

$$
g_F = \nabla_\theta\big(\gamma\,\mathcal{L}_{\text{NPO}}\big),\qquad
g_R = \nabla_\theta\big(\alpha\,\mathcal{L}_{\text{NLL}} + \lambda\,\mathcal{L}_{\text{KL}}\big)
$$

$$
\cos = \frac{g_F\cdot g_R}{\lVert g_F\rVert\,\lVert g_R\rVert},\qquad
\text{ratio} = \frac{\lVert g_F\rVert}{\lVert g_R\rVert}
$$

Measuring does not change the weights.

### 3.3 V3-E1 (PCGrad, retain-protecting)

$$
g_F' =
\begin{cases}
g_F - \dfrac{g_F\cdot g_R}{\lVert g_R\rVert^2}\,g_R & \text{if } g_F\cdot g_R < 0\\[2mm]
g_F & \text{otherwise}
\end{cases}
\qquad
g = g_F' + g_R
$$

Asymmetric by design: only the forgetting gradient is modified, so the retain direction is untouched. After projection, $g_F'\cdot g_R = 0$.

### 3.4 V3-E2 (SimNPO-style forget loss)

$$
\mathcal{L}_{\text{forget}} = -\frac{2}{\beta_s}\,\mathbb{E}_{D_f}\Big[\log\sigma\Big(\beta_s\Big(\frac{\mathrm{NLL}(y\mid x)}{|y|}-\delta\Big)\Big)\Big]
$$

with $\beta_s = 4.5$, $\delta = 0$, $\gamma = 0.125$. Per-token average loss; no reference model for the forget term. Retain terms as in V2.

### 3.5 Scores (see `scripts/three_scores.py`)

$$
\text{Privacy} = \mathrm{HM}_{\,m\in\{\text{LOSS, ZLib, MinK, MinK++}\}}\big(\text{closeness of MIA}_m \text{ to the retain model}\big)
$$

$$
\text{Utility} = \mathrm{HM}\Big(\tfrac{\mathrm{MU}_{\text{run}}}{\mathrm{MU}_{\text{full}}},\,\tfrac{\text{Fluency}_{\text{run}}}{\text{Fluency}_{\text{full}}}\Big)
$$

HM = harmonic mean. The privacy closeness is an *approximation* of the paper's formula (relative difference of 1 − AUC versus the retain model). A paper-style composite *forgetting* score (HM of 1−ES, 1−EM, 1−ParaProb, 1−TruthRatio) is also computed by the script, but **Section 4.1 shows it cannot separate the full model from the retain model here, so it is not used for conclusions.**

---

## 4. Results

### 4.1 Reference models: how much room is there to measure forgetting?

| Metric | Full model | Retain model | E0 (step 65) | E1 (step 65) | E2 (step 65) |
|---|---:|---:|---:|---:|---:|
| Extraction strength (ES) | 0.112 | 0.108 | 0.030 | 0.030 | 0.115 |
| Exact memorization (EM) | 0.654 | 0.628 | 0.556 | 0.547 | 0.654 |
| Para. probability | 0.141 | 0.133 | 0.047 | 0.062 | 0.140 |
| Truth ratio (knowledge form) | 0.550 | 0.538 | 0.534 | 0.536 | 0.550 |
| Forget QA probability | 0.242 | 0.210 | 0.071 | 0.101 | 0.242 |
| Retain QA probability | 0.242 | 0.242 | 0.085 | 0.134 | 0.245 |
| Model utility | 0.338 | 0.340 | 0.270 | 0.314 | 0.339 |
| MIA AUC (MinK) | 0.481 | 0.378 | 0.313 | 0.316 | 0.481 |
| PrivLeak (vs. retain reference) | −16.5 | – | +10.5 | +9.9 | −16.6 |

Reading this table:

- **The full and retain models are almost identical on all four memorisation metrics** (for example ES 0.112 vs 0.108). The model barely memorised the forget set, and EM is high (0.63) even for the retain model because many answer tokens are predictable without memorisation. The composite forgetting score therefore runs only from 0.540 (full) to 0.561 (retain), so a threshold such as 0.7 is not meaningful for forgetting on this model.
- **E2 did not unlearn:** its numbers equal the full model's in every row.
- **E0 and E1 seed 0 forget past the retain model:** ES 0.030 versus 0.108, forget probability 0.07-0.10 versus 0.21, MIA AUC 0.31 versus 0.378 (seed 0 only). Seeds 1 and 2 end closer to the retain model (step-65 PrivLeak -0.7 to +3.6), so the size of the overshoot depends on the seed.
- **Utility cost:** retain-set probability falls from 0.242 to 0.085 (E0) and 0.134 (E1).

### 4.2 V1 and V2 (original metric set, single run each)

| Metric | Full model | V1 | V2 |
|---|---:|---:|---:|
| Model utility | 0.3386 | 0.2694 | 0.2696 |
| Forget Quality (KS p-value) | 0.181 | 0.864 | 0.967 |
| Forget QA probability | 0.2425 | 0.0711 | 0.0712 |
| Extraction strength | 0.1125 | 0.0300 | 0.0300 |
| PrivLeak | −16.5 | 9.34 | 10.41 |
| MIA AUC (MinK) | 0.481 | 0.320 | 0.313 |

### 4.3 Seed study: E0 (no PCGrad) vs E1 (PCGrad), 3 seeds each

Step 65 (final):

| Run | PrivLeak | Model utility | Retain QA prob. | Forget Quality | Utility score | Privacy score |
|---|---:|---:|---:|---:|---:|---:|
| E0 seed 0 | 10.52 | 0.270 | 0.085 | 0.864 | 0.887 | 0.924 |
| E0 seed 1 | −0.68 | 0.245 | 0.084 | 0.864 | 0.840 | 0.949 |
| E0 seed 2 | 1.53 | 0.241 | 0.082 | 0.523 | 0.832 | 0.957 |
| **E0 mean ± sd** | 3.79 ± 5.93 | **0.252 ± 0.016** | **0.084 ± 0.001** | 0.750 ± 0.196 | 0.853 ± 0.030 | 0.943 ± 0.017 |
| E1 seed 0 | 9.94 | 0.314 | 0.134 | 0.907 | 0.963 | 0.924 |
| E1 seed 1 | 3.63 | 0.297 | 0.152 | 0.813 | 0.935 | 0.969 |
| E1 seed 2 | 3.12 | 0.296 | 0.156 | 0.942 | 0.934 | 0.968 |
| **E1 mean ± sd** | 5.56 ± 3.80 | **0.303 ± 0.010** | **0.147 ± 0.012** | 0.887 ± 0.066 | 0.944 ± 0.017 | 0.954 ± 0.026 |

Earlier checkpoints (model utility, mean ± sd): step 26: E0 0.189 ± 0.059, E1 0.278 ± 0.024; step 39: E0 0.242 ± 0.020, E1 0.297 ± 0.012.

What the three seeds support:

- **PCGrad's utility benefit holds across seeds.** At steps 26, 39 and 65 every E1 seed has higher model utility than every E0 seed. At step 65 the gap is +0.051 against a pooled seed sd of 0.013. Seed-matched differences are +0.044, +0.052 and +0.055 (assuming the same seed gives the same batch order, which was not verified). Retain-set probability is 0.147 versus 0.084, about 75 % higher, with no overlap between the two groups.
- **E0's utility dips mid-training and varies more.** At step 26, E0 seeds 1 and 2 drop to 0.142 and 0.169, then recover to about 0.24; E1 stays above 0.26.
- **No detectable effect on PrivLeak or the privacy score.** PrivLeak at step 65 is 3.8 +/- 5.9 (E0) versus 5.6 +/- 3.8 (E1); the groups overlap, and with three seeds neither mean is clearly different from 0.
- **Forget Quality is noisy** (0.52–0.94) and not used for conclusions.
- n = 3 per arm, so standard deviations are rough; treat the numbers as indicative, not as significance tests.

### 4.4 V4 ablations on E1 (seed 0 only; seed spread from E1 seeds 0–2)

| Run | Change vs E1 | PrivLeak @65 | Model utility @65 | Retain QA prob @65 |
|---|---|---:|---:|---:|
| E1 seed 0 | none | 9.94 | 0.314 | 0.134 |
| V4a | γ 1.0 → 0.5 | 9.23 | 0.325 | 0.158 |
| V4b | α 1.0 → 2.0 | 9.31 | 0.322 | 0.153 |

Both changes move PrivLeak by under 1 point and utility by about one seed sd (0.010). **Not distinguishable from noise.** Logged pre-clip gradient norms were 65–400 against `max_grad_norm` 1.0, so every logged step was clipped, which would limit what loss weights can do (hypothesis, not tested).

---

### 4.5 V5 to V9: learning rate, KL weight and the PCGrad ablation (checkpoint 65)

Mean ± sample sd over seeds; scores from `scripts/three_scores.py`. Reference values: retain model utility 0.340, min_k AUC 0.378 (PrivLeak target 0); full model utility 0.338, PrivLeak −16.5.

| Group | Seeds | PrivLeak | Model utility | Retain QA prob | Forget quality (KS) | Utility score | Privacy score |
|---|---:|---:|---:|---:|---:|---:|---:|
| E0 (V2 objective, lr 1e-5, KL 0.1) | 3 | 3.8 ± 5.9 | 0.252 ± 0.016 | 0.084 ± 0.001 | 0.750 ± 0.196 | 0.853 ± 0.030 | 0.943 ± 0.017 |
| E1 (PCGrad, lr 1e-5, KL 0.1) | 3 | 5.6 ± 3.8 | 0.303 ± 0.010 | 0.147 ± 0.012 | 0.887 ± 0.066 | 0.944 ± 0.017 | 0.954 ± 0.026 |
| V5 (E1, lr 5e-6) | 1 | −12.6 | 0.337 | 0.233 | 0.813 | 0.997 | 0.864 |
| V6 (E1, lr 7e-6) | 1 | −2.1 | 0.311 | 0.139 | 0.864 | 0.958 | 0.956 |
| V6 (E1, lr 8e-6) | 3 | −3.6 ± 3.9 | 0.296 ± 0.015 | 0.141 ± 0.003 | 0.947 ± 0.035 | 0.933 ± 0.024 | 0.944 ± 0.023 |
| V7 (E1, KL 0.5, lr 1e-5) | 3 | +7.6 ± 4.5 | 0.319 ± 0.005 | 0.167 ± 0.011 | 0.861 ± 0.047 | 0.970 ± 0.008 | 0.935 ± 0.036 |
| V8 (E1, KL 0.5 + lr 8e-6) | 3 | −3.2 ± 3.6 | 0.308 ± 0.012 | 0.160 ± 0.001 | 0.927 ± 0.035 | 0.953 ± 0.019 | 0.948 ± 0.023 |
| V9 (no PCGrad, KL 0.5 + lr 8e-6) | 3 | −4.9 ± 5.5 | 0.246 ± 0.019 | 0.083 ± 0.003 | 0.964 ± 0.022 | 0.842 ± 0.037 | 0.925 ± 0.025 |

Per seed (seeds 0, 1, 2) for the three groups at lr 8e-6:

| Group | PrivLeak | Model utility | Retain QA prob |
|---|---|---|---|
| V6 (KL 0.1, PCGrad) | +0.9, −5.5, −6.1 | 0.312, 0.291, 0.285 | 0.144, 0.138, 0.141 |
| V8 (KL 0.5, PCGrad) | +0.9, −5.2, −5.4 | 0.322, 0.303, 0.299 | 0.159, 0.161, 0.161 |
| V9 (KL 0.5, no PCGrad) | +1.4, −8.0, −8.1 | 0.268, 0.237, 0.233 | 0.087, 0.081, 0.081 |

**What the data supports**

1. **PCGrad is the active ingredient at both settings.** Every V8 seed is above every V9 seed on utility (0.299–0.322 against 0.233–0.268) and on retain-set probability (0.159–0.161 against 0.081–0.087). The gain is about +0.06 utility and +0.08 retain probability at the tuned setting, against +0.05 and +0.06 at the original one (E1 against E0). With 3 seeds per group, perfect separation can happen by chance about 1 time in 20 per metric, so this is consistent evidence, not proof.
2. **The learning rate moves PrivLeak.** At lr 8e-6 the mean PrivLeak is −3.6 ± 3.9 against +5.6 ± 3.8 for E1, a difference of about 9 points against a standard error of about 3, while utility stays within noise (0.296 against 0.303). Single-seed points on the same seed 0 show the trade-off: 5e-6 gives −12.6 (almost the full model, which scores −16.5), 7e-6 gives −2.1, 1e-5 gives +9.9.
3. **A larger KL weight raises retain-set probability.** At lr 8e-6, KL 0.5 gives 0.160 ± 0.001 against 0.141 ± 0.003 for KL 0.1, with every V8 seed above every V6 seed. At lr 1e-5 (V7 against E1, 3 seeds each) the direction is the same: retain probability 0.167 ± 0.011 against 0.147 ± 0.012, and utility 0.319 ± 0.005 against 0.303 ± 0.010. The utility gains (+0.012 and +0.016) are about one to two seed sd, so they are suggestive, not established.
4. **The privacy shift is best attributed to the learning rate, not the KL weight.** The learning rate alone (V6) reproduces V8's PrivLeak (−3.6 against −3.2). The KL weight alone (V7, 3 seeds, lr 1e-5) did not lower it (+7.6 ± 4.5 against +5.6 ± 3.8 for E1). Interactions were not tested.
5. **V8 is not shown to be better than E1.** Utility 0.308 against 0.303 and utility score 0.953 against 0.944 are within one seed sd; privacy score is 0.948 against 0.954. PrivLeak changes sign (−3.2 against +5.6), but the mean absolute value per seed is 3.8 against 5.6 with overlapping seeds. The one clear difference is retain-set probability (0.160 against 0.147, about one E1 sd), and it is much tighter across seeds.

**Limits.** The learning rate and KL weight were chosen after looking at PrivLeak, which is measured on the same holdout10 examples in every run, so the V8 setting is partly selected on evaluation data. V8 changed two settings at once. V5 and V6 (7e-6) have one seed each. V5 to V9 were evaluated at steps 26, 39 and 65 only. Raw files: `research_results/v5_v6` (with `summary.csv`), `v7`, `v8`, `v9`.

---

## 5. What the gradient diagnostics showed

| Run | Mean cos(g_F, g_R) | Fraction with cos < 0 | Median ‖g_F‖/‖g_R‖ |
|---|---:|---:|---:|
| E0 | −0.50 | 0.90 | 1.13 (max 44.9) |
| E1 (measured before projection) | −0.51 | 0.90 | 4.95 (max 44.8) |
| E2 | −0.008 | 0.52 | 0.002 |

- Forgetting and retain gradients strongly oppose each other (cosine about −0.5, negative on 90 % of measurements) in **every layer** (per-layer means about −0.30 to −0.58).
- Early in training the forget gradient can be about 40× larger than the retain gradient (spikes).
- With the SimNPO-style loss the forget gradient is about 500× smaller than the retain gradient, so the model barely changes.

---

## 6. What each change taught us

1. **V1 → V2:** masking the KL changed almost nothing; the bottleneck is not the KL implementation.
2. **V2 → V3-E0:** the diagnostic reproduced V2 and exposed the real problem, a persistent conflict between the forgetting and retain gradients.
3. **E0 → E1 (3 seeds):** resolving the conflict in the retain direction keeps more of the model's general ability at the same level of forgetting and privacy. This is the main positive result.
4. **E0 → E2:** swapping the forget loss alone does not help in this configuration; the model does not unlearn. The likely cause (a **hypothesis, not tested**) is that the SimNPO loss is nearly flat when the per-token loss on forget answers is small, so δ needs to be raised.
5. **E1 to V4a/V4b:** no detectable effect of the loss weights (one seed each, so only large effects would show). Pre-clip gradient norms are 65-400 against `max_grad_norm` 1.0, so every logged step is clipped, but clipping rescales the whole gradient by one factor and does not by itself explain why the weights had no effect. This is unexplained.
6. **A mild privacy overshoot may remain, but the evidence is weak:** PrivLeak at step 65 is positive in five of the six E0/E1 runs (E0 seed 1 is -0.7), but only seed 0 is large (+9.9 and +10.5); the others lie between -0.7 and +3.6, and PrivLeak from a single MIA run on a few hundred examples per set is itself noisy. The step at which PrivLeak crosses 0 varies by seed, so a fixed stopping step is not robust.
7. **E1 → V5/V6 (learning rate):** the peak learning rate sets how far the model moves along one trade-off. At 5e-6 it barely unlearns (utility 0.337, PrivLeak −12.6); at 8e-6 the 3-seed mean PrivLeak is −3.6 with utility unchanged within noise; at 1e-5 it is +5.6. Privacy is mostly a question of how far the model is allowed to move.
8. **E1 → V7 (KL weight):** a larger KL weight raises retain-set probability (0.167 against 0.147, 3 seeds each) and utility slightly (0.319 against 0.303) but does not lower PrivLeak.
9. **V8 and V9 (combination and ablation):** combining lr 8e-6 with KL 0.5 gives a model that is not distinguishable from E1 on utility or privacy, with higher and steadier retain-set probability. Switching PCGrad off at the same setting loses about 0.06 utility and 0.08 retain probability in all three seeds.
10. **Where the improvement stands:** one change reproducibly improves the model, PCGrad (3 seeds at two settings). The other changes tried either move along the learning-rate trade-off or are within seed noise. The version count overstates the progress: the real gain came at E1.

---

## 7. Status against the goal

- **Utility and privacy:** for E1, utility score 0.944 ± 0.017 and privacy score 0.954 ± 0.026 over 3 seeds, both above 0.7. E0 is at 0.853 and 0.943.
- **Forgetting:** on this model the memorisation metrics cannot separate the full from the retain model (Section 4.1), so "≥ 0.7 forgetting" is not a meaningful target. What can be said: E0 and E1 forget at least as much as the retain model by ES, forget probability and MIA, and E2 does not forget.
- **V8 (3 seeds):** utility score 0.953 ± 0.019 and privacy score 0.948 ± 0.023, both above 0.7 and equal to E1 within noise. V8 is the more stable choice on retain-set probability (0.160 ± 0.001 against 0.147 ± 0.012); it is not shown to be better on utility or privacy.
- **Stability:** E1 keeps utility higher and its early-training dip smaller than E0 (Section 4.3). Privacy remains about equally variable across seeds.
- The OpenUnlearning paper reports no existing method with all three scores above 0.7 on its own setup, so a clear process with a measured improvement is a legitimate outcome.
- A claim of success should use: *utility score and privacy score over 3 seeds, forgetting judged against the retain model's reference values*, not a 0.7 threshold on the composite forgetting score.

---

## 8. Limitations and next steps

**Limitations**

- Three seeds for E0, E1, V6 (lr 8e-6), V7, V8 and V9; one seed for E2, V4a, V4b, V5 and V6 (lr 7e-6). Seeds change training randomness only; the forget and holdout examples are the same in every run.
- The learning rate and KL weight of V8 were chosen after seeing PrivLeak on the same holdout examples used for evaluation, so that setting is partly selected on evaluation data.
- No external baseline: the framework's own NPO and GradDiff trainers have not been run under this protocol, so the report cannot say that E1 or V8 beat them.
- The privacy score uses an approximation of the paper's formula; the composite forgetting score is not discriminative on this model.
- Qwen2.5-0.5B differs from the paper's Llama-3.2-1B; numbers are not comparable with the paper's tables.
- The target model barely memorised the forget set (ES 0.11), which limits every memorisation-based metric.
- MIA-based metrics were found in the paper to be faithful but not robust.

**Next steps**

1. Run the framework's NPO and GradDiff under the same protocol (3 seeds, same checkpoints) so that PCGrad's gain can be compared with standard methods. This is the largest gap. A larger `max_grad_norm` is lower priority: clipping rescales the whole gradient, and its effect with AdamW is untested.
2. Re-run SimNPO with δ > 0 (≈ 1.4 as a starting guess) if time allows.
3. Optional: repeat the best configuration on a model that memorises more (for example the Llama-3.2-1B checkpoints used in the paper) so forgetting metrics become informative.
4. Optional: relearning or quantization stress tests, as in the paper.

---

## 9. Items to fix before presenting

- [x] Resolved. The project README's "65 steps per epoch, 325 in total" is wrong: every run has 65 steps in total (13 per epoch) per `trainer_state.json`. Its full-model PrivLeak of 3.889 is exactly what the default reference AUC of 0.5 gives (100 x (0.5 - 0.4806) / 0.5, with the full model's min_k AUC of 0.4806). With the retain-model reference (AUC 0.378) the value is -16.48, as in `research_results/references/`. The retain model's stored +24.38 also comes from the default 0.5 reference. V1 and V2 have 26 metrics against 31 in the V3/V4 full-metric set. The project README still needs correcting.
- [ ] The retain model's own PrivLeak (+24.4) was computed without the reference and is not meaningful; do not quote it.
- [ ] Check the V1/V2 `trainer_state.json` files to confirm their step counts.
- [ ] Forget Quality for E0 seed 0 (0.864) and V2 (0.967) differ for the same objective, which again points to run-to-run variation.

---

## 10. Reproducing the experiments

```bash
# environment: conda activate unlearning ; choose a GPU that nvidia-smi shows as nearly empty
GPU=<id> bash scripts/v3_run_fast.sh eval_refs                  # retain + full model, full metric set
GPU=<id> bash scripts/v3_run_fast.sh e0_diag                    # V2 objective + gradient diagnostics
GPU=<id> bash scripts/v3_run_fast.sh e1_pcgrad                  # PCGrad
GPU=<id> bash scripts/v3_run_fast.sh e2_simnpo                  # SimNPO-style forget loss
GPU=<id> SEED=<n> bash scripts/v3_run_fast.sh e0_diag_seed      # another seed for E0 (e1_pcgrad_seed for E1)
GPU=<id> bash scripts/v3_run_fast.sh eval_ckpts <run>           # evaluate every saved checkpoint of a run
GPU=<id> SEED=<n> bash scripts/run_v5_seed.sh                  # V5: E1 at lr 5e-6 (trains, then evaluates steps 26, 39, 65)
GPU=<id> LR=<lr> SEED=<n> bash scripts/run_v6.sh                # V6: E1 at another learning rate
GPU=<id> KL=<w> SEED=<n> bash scripts/run_v7.sh                 # V7: E1 with a chosen KL weight
GPU=<id> KL=<w> LR=<lr> SEED=<n> bash scripts/run_v8.sh         # V8: E1 with KL weight and learning rate
GPU=<id> KL=<w> LR=<lr> SEED=<n> bash scripts/run_v9.sh         # V9: same setting without PCGrad
python scripts/compare_seeds.py                                 # E0 vs E1 across seeds
python scripts/three_scores.py --full <full.json> --retain <retain.json> name=<TOFU_EVAL.json> ...
python scripts/summarize_grad_diag.py <run>/grad_diag.jsonl
```

Code: `src/trainer/unlearn/custom_v3.py`, `configs/trainer/CustomV3.yaml`, `configs/eval/tofu_full.yaml`, `scripts/`. Raw results: `research_results/v3` to `research_results/v9`.
