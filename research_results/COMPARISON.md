# Transformer Unlearning: Experimental Results

## Experimental Setup

- Dataset: TOFU
- Base model: Qwen2.5-0.5B-Instruct
- Forget split: forget10
- Retain split: retain90
- Training epochs: 5
- Training steps: 65 per epoch
- Evaluation: OpenUnlearning TOFU evaluation

## 1. Model Comparison

| Metric | Full Model | CustomV1 | CustomV2 |
|---|---:|---:|---:|
| Forget Truth Ratio | 0.73259 | 0.74136 | 0.73902 |
| Forget Quality | Not available | 0.86353 | 0.96741 |
| Forget QA Probability | 0.24253 | 0.07109 | 0.07124 |
| Forget QA ROUGE | 0.36485 | 0.36838 | 0.37883 |
| Model Utility | 0.33862 | 0.26937 | 0.26956 |
| PrivLeak | 3.88875 | 9.34452 | 10.41379 |
| Extraction Strength | 0.11248 | 0.02995 | 0.02995 |

Note: Comparisons are meaningful only when evaluation configurations,
reference logs, and metric implementations are consistent.
The full-model Forget Quality was unavailable in the latest JSON.

## 2. CustomV1 Objective

CustomV1 combines:

- NPO forgetting objective
- Retain negative log-likelihood
- Retain KL regularization

L = gamma * L_NPO + alpha * L_retain + kl_alpha * L_KL

## 3. CustomV2 Objective

CustomV2 uses the same objective as V1 but replaces the original
retain KL computation with a masked KL implementation.

The mask excludes ignored labels and padding positions.

## 4. Observations

1. Both V1 and V2 substantially reduce forget-answer probability.
2. Retain-answer probability also decreases substantially.
3. Model utility falls from approximately 0.339 to 0.270.
4. V2 produces only a very small utility improvement over V1.
5. V2 shows modest changes in Forget Quality and PrivLeak.
6. Extraction Strength is identical for V1 and V2 at the reported precision.

These are observed results, not proof of statistical significance.
Only one reported training run per method is available here.

## 5. Research Questions for V3

1. Are forgetting and retain gradients conflicting?
2. Are the gradient magnitudes poorly balanced?
3. Which transformer layers experience the strongest gradient conflict?
4. Does adaptive gradient handling improve the forgetting-utility trade-off?

## 6. V3 Plan

First measure:

- NPO gradient norm
- Retain NLL gradient norm
- Retain KL gradient norm
- Pairwise gradient cosine similarities
- Layer-wise gradient norms

Do not introduce gradient projection or adaptive weighting until
the diagnostics provide evidence that they are justified.

## 7. Limitations

- Results require repeated seeds for statistical confidence.
- Privacy metrics must be interpreted using their exact evaluation
  configuration and attack conventions.
- Forgetting, privacy, and utility are distinct properties.
- Reduced answer probability alone does not prove complete knowledge removal.
