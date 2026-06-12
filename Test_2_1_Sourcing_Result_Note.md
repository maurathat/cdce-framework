# Test 2.1 Sourcing — Activation-Grounding Result

**Status:** Closed as a negative sourcing result.
**Date:** 2026-05-27
**Commits:** `78d2f7b` (data + scripts), follow-up commit for this note.

## Summary

Test 2.1 Sourcing asks whether the operator-count metric — parsed from a model's generated text by a verb-based parser — corresponds to internal structure in the model's activations, or is a surface-text phenomenon. We test this on an open model where activations are accessible.

**We find no robust evidence that operator count tracks residual-stream participation ratio after controlling for generation length, on this metric, model, and domain.**

## Method

- **Model:** `google/gemma-2-2b-it`, loaded via HuggingFace `transformers` on Apple MPS, with chat template (`add_generation_prompt=True`).
- **Prompts:** 45 GSM8K questions, stratified across low / medium / high reference step counts (from the gold-answer `<<calculator>>` annotations, used only as a sampling aid).
- **Pipeline:** for each prompt, generate the response, parse operator count `n_text` from the generated text with the project's operator parser, then re-forward the full sequence through the model with `output_hidden_states=True, use_cache=False`. The activation effective-dimension `d_act` is computed as the participation ratio `(Σλ)² / Σ(λ²)` over the eigenspectrum of the *centered* covariance of the generated-token hidden states (prompt and template tokens sliced out). Run for three layer aggregations: last layer only, mean of middle layers, mean of all layers.
- **Statistic:** partial correlation (Pearson and Spearman) of `n_text` vs `d_act`, controlling for generated token count, computed by residualizing both variables against the control and correlating residuals.
- **Robustness:** leave-one-out — recompute the partial Spearman with each row removed in turn; if any single drop pushes p above 0.05 the result is reported as fragile.

## Result

| Layer mode | Partial Pearson r (p) | Partial Spearman r (p) | Leave-one-out fragility |
|---|---|---|---|
| last | +0.192 (0.206) | **+0.268 (0.076)** | fragile (drop row 16 → p = 0.163) |
| midmean | +0.034 (0.826) | +0.001 (0.996) | fragile (drop row 15 → p = 0.999) |
| allmean | +0.050 (0.744) | −0.026 (0.866) | fragile (drop row 33 → p = 0.999) |

- The **last-layer** partial Spearman is weakly positive (r = 0.268) but does not reach significance at p < 0.05 and is fragile under leave-one-out.
- **Middle-layer** and **all-layer** aggregations are effectively null (r ≈ 0).
- All three layer modes are leave-one-out fragile; in midmean and allmean, dropping any single point reduces an already-null estimate to essentially zero — the signature of no underlying signal.

## Interpretation, with bounds

The operator metric and activation participation ratio are approximately independent on this substrate, once length is controlled. The metric does not get *more* tied to activations at deeper or earlier representations — there is no layer mode where a substantive signal emerges.

Three explanations are consistent with these numbers; the data does not distinguish between them:

1. The metric is a surface-text artifact and does not correspond to the kind of internal dimensional structure that participation ratio measures.
2. Participation ratio over a residual-stream trajectory is the wrong instrument for whatever the metric actually measures.
3. The GSM8K arithmetic domain is too narrow, or 45 prompts too small, to surface a relationship that exists more broadly.

We therefore close this experiment with the **narrower claim** that operator count behaves as a text-side measurement with respect to *this activation instrument, model, and domain.* We do not claim that the metric is text-only in general, nor does this experiment resolve the separate question of whether operator count correlates with human-annotated reasoning operations (the validity half of Test 2.1, which remains open and requires human annotation).

## Methodological note: corrected analysis

The original last-layer run reported a partial Spearman of **r = 0.455, p = 0.002**, which appeared significant. Independent reimplementation of the partial-correlation computation (using `sklearn.LinearRegression` for residualization) returned **r = 0.268, p = 0.076**. The two implementations disagreed by enough to flip the result across the conventional significance threshold.

A diagnostic enumerating partial correlations under all combinations of control variable (`token_count` / `prompt_token_count` / `total_token_count`) × rank order (raw / ranked) isolated the cause: the original `pinv`-based residualization was projecting out the control variable *without an intercept term* — the design matrix was just the control column, with no constant. Without the intercept, the residuals are forced through the origin of residual space rather than being mean-centered, which propagates the variables' non-zero means into the inner product of "residuals" and inflates the partial correlation.

Fix: add a constant column to the design matrix:
```python
# before:  design = control.reshape(-1, 1)
# after:   design = np.column_stack([np.ones(len(control)), control])
```

With the intercept term added, the corrected `partial_corr` reproduces the independent reimplementation's r = 0.268 exactly. The bug was confined to the post-loop summary statistic; the per-prompt CSV columns (`n_text`, `d_act`, `token_count`) are produced upstream of the partial-correlation call and were not affected — so the midmean and allmean runs from earlier (which predated the fix) were valid as data and only needed the summary recomputed with the corrected function.

This is recorded transparently because the discovery of the bug is part of the result.

## Scope and what this does not say

- **One small model.** Gemma-2-2B-it is a moderate-capability instruction-tuned model. A higher-capability local model (e.g. a 26B-class reasoner) might surface a relationship this one does not. We do not claim otherwise.
- **One proxy.** Participation ratio over the residual stream is a coarse instrument; a richer activation measure (e.g. trained probes for reasoning operations, layer-resolved attention patterns) could find structure that PR does not capture.
- **One domain.** GSM8K is arithmetic word problems. Non-arithmetic reasoning may behave differently.
- **n = 45.** Modest power. A null at this n is not the same as a null at n = 500.

The negative result is real for what was tested. It does not generalize without further evidence.

## Artifacts

- Experiment script: `test_2_1b_activation_grounding.py` (intercept fix at the partial-correlation function)
- Smoke test: `test_2_1b_smoke_tinyllama.py` (pipeline validation on TinyLlama)
- Prompt builder: `build_prompts_2_1b.py`
- Checker (independent reimplementation): `check_2_1b_result.py`
- Diagnostic that surfaced the bug: `diagnose_discrepancy.py`
- Provenance package: `provenance/`
- Prompt set: `data/prompts_2_1b.txt`, `data/prompts_2_1b.manifest.json`
- Per-layer results and provenance sidecars:
  - `results_2_1b.csv` / `results_2_1b.provenance.json` (last layer)
  - `results_2_1b_midmean.csv` / `results_2_1b_midmean.provenance.json`
  - `results_2_1b_allmean.csv` / `results_2_1b_allmean.provenance.json`
- Scatter plots: `check_2_1b_scatter*.png`

## Status of Test 2.1 overall

- **Sourcing half (this experiment):** closed as a negative sourcing result on this metric / model / domain. Two convergent arguments now support the text-side reading: the prior closed-API argument (commercial APIs do not expose activations, so the metric is necessarily a post-hoc text parse there), and this direct empirical test on an open model.
- **Validity half (operator count vs human-annotated reasoning operations, target r ≥ 0.7):** still open. The local-model infrastructure built for this experiment does not address it — validity is a human-labeling question regardless of model. Resolution requires blind annotation by a second annotator.
