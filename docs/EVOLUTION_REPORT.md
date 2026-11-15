# Unlearning Algorithm Evolution: V1 → V3

**Model:** Qwen2.5-0.5B-Instruct  **Benchmark:** TOFU (forget10 / retain90)  **Framework:** OpenUnlearning
**Status:** working draft, 15 November 2026. All results are **single-seed** unless stated.

> Items marked **TODO** must be filled or verified before presenting (see Section 9).

---

## 1. Goal and setup

**Goal.** A hybrid unlearning algorithm that is *stable on all three criteria* (forgetting, privacy, utility), instead of strong on one and weak on another. **Method.** Change one mechanism at a time, measure, keep what helps.

| Item | Value |
|---|---|
| Target model | Qwen2.5-0.5B-Instruct fine-tuned on TOFU (`checkpoint-625`) |
| Reference model | retain90-only model (gold standard) |
| Data | forget10 (400 QA pairs), retain90, holdout10 (for MIA) |
| Optimisation | lr 1e-5, weight decay 0.01, 1 warmup epoch, 5 epochs |
| Steps | V3 runs: 65 optimizer steps in total (13 per epoch; checkpoints at 13, 26, 39, 52, 65) |
| Loss weights | NPO β = 0.1, γ = 1, α = 1, KL weight λ = 0.1 |
| Evaluation | V1/V2: default OpenUnlearning TOFU metrics. V3: full set (Exact Memorization, 4 MIAs, fluency) |

---

## 2. Evolution at a glance

| Version | What changed | Why | Outcome |
|---|---|---|---|
| **V1** | NPO forgetting + retain NLL + retain KL (unmasked) | First working custom objective | Forget probability 0.243 → 0.071, but retain probability also falls; model utility 0.339 → 0.269 |
| **V2** | KL computed on answer tokens only (padding / ignored labels masked) | Padding should not contribute to the KL anchor | Utility 0.2696 vs 0.2694 (negligible); Forget Quality 0.967 vs 0.864 (single runs) |
| **V3-E0** | Same objective as V2; adds in-training gradient diagnostics; reuses the retain forward pass for the KL term | Measure whether forget and retain gradients conflict; run faster | Reproduces V2 closely (utility 0.2697, extraction strength 0.02995 in both) |
| **V3-E1** | PCGrad-style update: remove the part of the forget gradient that opposes the retain gradient | Diagnostics showed strong conflict (Section 5) | Utility score 0.887 → 0.963; forgetting and privacy unchanged |
| **V3-E2** | Forget loss replaced by SimNPO-style length-normalised, reference-free loss (γ = 0.125, β = 4.5, δ = 0) | Test a mechanism from the best-balanced method in the OpenUnlearning paper | Under-forgets: forget gradient ≈ 500× smaller than retain gradient; Forget Quality 0.13–0.21 |
| *Planned* | SimNPO with δ > 0; two more seeds for E0/E1; hybrid of what helped | Section 8 | Not yet run |

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

The NPO term uses sequence log-probabilities (only the "negative" NPO term). **V1** averages the KL over all positions. **V2** restricts it to the set $M$ of answer tokens (label ≠ −100).

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

with $\beta_s = 4.5$, $\delta = 0$, $\gamma = 0.125$. It uses the per-token average loss and needs no reference model for the forget term. Retain terms are as in V2.

### 3.5 Scores used to compare runs

$$
\text{Forgetting} = \mathrm{HM}\big(1-\text{ES},\,1-\text{EM},\,1-\text{ParaProb},\,1-\text{TruthRatio}\big)
$$

$$
\text{Privacy} = \mathrm{HM}_{\,m\in\{\text{LOSS, ZLib, MinK, MinK++}\}}\big(\text{closeness of MIA}_m \text{ to the retain model}\big)
$$

$$
\text{Utility} = \mathrm{HM}\Big(\tfrac{\mathrm{MU}_{\text{run}}}{\mathrm{MU}_{\text{full}}},\,\tfrac{\text{Fluency}_{\text{run}}}{\text{Fluency}_{\text{full}}}\Big),\qquad
\text{HM3} = \mathrm{HM}(\text{Forgetting},\text{Privacy},\text{Utility})
$$

HM = harmonic mean. **Caveats:** the privacy closeness uses an *approximation* of the paper's formula (relative difference of 1 − AUC versus the retain model); the truth-ratio and fluency parts are also approximations. See `scripts/three_scores.py`.

---

## 4. Results

### 4.1 V1 and V2 (original metric set, single run each)

| Metric | Full model | V1 | V2 |
|---|---:|---:|---:|
| Model utility | 0.3386 | 0.2694 | 0.2696 |
| Forget Quality (KS p-value) | n/a | 0.864 | 0.967 |
| Forget QA probability | 0.2425 | 0.0711 | 0.0712 |
| Extraction strength | 0.1125 | 0.0300 | 0.0300 |
| PrivLeak | **TODO** (see Section 9) | 9.34 | 10.41 |
| MIA AUC (MinK) | 0.481 | 0.320 | 0.313 |

The retain-model reference AUC implied by the PrivLeak values is about 0.378. V1 and V2 sit *below* it, i.e. they forget past the retain model.

### 4.2 V3 runs (full metric set, final checkpoint = step 65, single seed)

| Run | Forgetting | Privacy | Utility | HM3 | Forget Quality | Raw model utility |
|---|---:|---:|---:|---:|---:|---:|
| E0 (V2 objective) | 0.617 | 0.924 | 0.887 | 0.783 | 0.864 | 0.270 |
| E1 (PCGrad) | 0.619 | 0.924 | 0.963 | 0.803 | 0.907 | 0.314 |
| E2 (SimNPO, δ = 0) | 0.540 | 0.836 | 1.000 | 0.741 | 0.211 | 0.339 |
| Retain model (reference) | **TODO** | **TODO** | **TODO** | **TODO** | – | **TODO** |
| Full model (reference) | **TODO** | **TODO** | **TODO** | **TODO** | – | 0.339 |

Other E1 checkpoints: step 26 has the best privacy (0.963); step 39 has the best HM3 (0.808). E1 scores change little from step 26 onward.

The reference rows double as a check that the scoring script is correct: retain-model privacy should be 1.000 and full-model utility 1.000.

---

## 5. What the gradient diagnostics showed

| Run | Mean cos(g_F, g_R) | Fraction of measurements with cos < 0 | Median ‖g_F‖/‖g_R‖ |
|---|---:|---:|---:|
| E0 | −0.50 | 0.90 | 1.13 (max 44.9) |
| E1 (measured before projection) | −0.51 | 0.90 | 4.95 (max 44.8) |
| E2 | −0.008 | 0.52 | 0.002 |

- Forgetting and retain gradients strongly oppose each other (cosine ≈ −0.5, negative on 90 % of measurements) in **every layer** (per-layer means about −0.30 to −0.58).
- Early in training the forget gradient is up to ≈ 40× larger than the retain gradient, producing occasional spikes.
- With the SimNPO-style loss the forget gradient is ≈ 500× smaller than the retain gradient, so the model barely changes.
- One run each: this shows a clear pattern, not precise values.

---

## 6. What each change taught us

1. **V1 → V2:** masking the KL changed almost nothing. The bottleneck is not the KL implementation.
2. **V2 → V3-E0:** the diagnostic reproduced V2 and exposed the real problem, a persistent conflict between the forgetting and retain gradients.
3. **E0 → E1:** resolving the conflict in the retain direction improved the utility score with no visible cost to forgetting or privacy. This is the strongest positive result so far, but it is one seed.
4. **E0 → E2:** swapping the forget loss alone does not help in this configuration. The likely cause (a **hypothesis, not yet tested**) is that the SimNPO loss is almost flat when the per-token loss on forget answers is small, so δ needs to be raised.

---

## 7. Status against the goal

- Utility (0.96) and privacy (0.92) are above 0.7 for E1.
- Forgetting is 0.62 on the paper-style memorization score, **below 0.7**. This score rewards forgetting *more than the retain model does*; in the OpenUnlearning paper the retain model itself scores only about 0.31. Under Forget Quality (closeness to the retain model) the runs score 0.81–0.99 at later checkpoints, but that metric is noisy (0.58–0.98 across E1 checkpoints).
- Which forgetting definition counts as the target must be chosen on principle and stated, not chosen after seeing the numbers.
- The paper reports no existing method with all three scores above 0.7, so partial progress with a clear process is a legitimate outcome.

---

## 8. Limitations and next steps

**Limitations**

- Single seed per run; no variance estimate.
- The privacy score uses an approximation of the paper's formula.
- Different base model and tuning from the paper; numbers are not comparable with the paper's tables.
- MIA-based metrics were found in the paper to be faithful but not robust.
- The same 400 forget examples are used for training and evaluation.

**Next steps**

1. Fill the reference rows (retain and full model) from `research_results/v3/results_table.txt`.
2. Run E0 and E1 with two more seeds; evaluate steps 26 and 65.
3. Re-run the SimNPO variant with δ > 0 (≈ 1.4 as a starting guess).
4. Build one hybrid from the pieces that helped and compare it with the baselines under the same scoring.
5. Optional: relearning / quantization stress tests, as in the paper.

---

## 9. Corrections to make before presenting

- [ ] The project README states 65 steps per epoch and 325 steps in total. The V3 runs show 65 steps in total (13 per epoch), and E0 reproduces V2's results at that length, so the README figure is probably wrong. Check the V1/V2 `trainer_state.json` files.
- [ ] The full model's PrivLeak is 3.89 in the README but −16.48 in the repository JSON. Reconcile and update Section 4.1.
- [ ] Forget Quality for E0 (0.864) and V2 (0.967) differ although the objectives are identical. This points to run-to-run variation and supports the need for more seeds.
- [ ] Fill the **TODO** rows in Section 4.2.

---

## 10. Reproducing the V3 experiments

```bash
# environment: conda activate unlearning ; pick a free GPU with nvidia-smi
GPU=<id> bash scripts/v3_run_fast.sh eval_refs        # retain + full model, full metric set
GPU=<id> bash scripts/v3_run_fast.sh e0_diag          # V2 objective + gradient diagnostics
GPU=<id> bash scripts/v3_run_fast.sh e1_pcgrad        # PCGrad
GPU=<id> bash scripts/v3_run_fast.sh e2_simnpo        # SimNPO-style forget loss
GPU=<id> bash scripts/v3_run_fast.sh eval_ckpts <run> # evaluate every saved checkpoint of a run
python scripts/three_scores.py --full <full.json> --retain <retain.json> name=<TOFU_EVAL.json> ...
python scripts/summarize_grad_diag.py <run>/grad_diag.jsonl
```

Code: `src/trainer/unlearn/custom_v3.py`, `configs/trainer/CustomV3.yaml`, `configs/eval/tofu_full.yaml`, `scripts/`.
