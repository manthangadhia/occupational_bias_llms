# OLS Regression: Summary of Significant Results

## Model Overview

| Metric | Transformation | R² | Prob(F) | Residual Normality |
|---|---|---|---|---|
| mean_entropy | Box-Cox (λ = −0.035) | 0.967 | 0.00 | Mild violation (skew = 0.295) |
| mean_entropy_nucleus | Box-Cox (λ = −0.077) | 0.962 | 0.00 | Mild violation (skew = 0.245) |
| semantic_div | None | 0.600 | 0.00 | ✓ Normal (skew = −0.018) |
| self_bleu | Box-Cox (λ = 0.435) | 0.857 | 0.00 | ✓ Normal (skew = −0.092) |
| perplexity | Box-Cox (λ = −0.431) | 0.647 | 0.00 | Mild violation (skew = 0.564) |

All predictors and outcomes were standardised (z-scored) after transformation. Coefficients for continuous predictors (temperature) reflect change in outcome SDs per 1-SD increase in the predictor. Coefficients for binary predictors reflect the difference in outcome SDs between the two groups, holding all else constant. 

> ⚠️ **Interpretation caveat for Box-Cox-transformed metrics:** Because Box-Cox was applied before standardisation, coefficients for `mean_entropy`, `mean_entropy_nucleus`, `self_bleu`, and `perplexity` are in standard deviations of the *transformed* metric, not the original raw scale. Conclusions about direction and significance are fully valid; exact magnitude should be interpreted with this in mind. `semantic_div` required no transformation and can be interpreted directly in SD units of the original metric.

---

## Significant Results (p < 0.05)

| Metric | Predictor | Coef | p-value | Natural Language Interpretation |
|---|---|---|---|---|
| mean_entropy† | temperature | +0.982 | < 0.001 | A one-SD increase in temperature is associated with a 0.98-SD increase in entropy — the dominant driver of entropy in the data. |
| mean_entropy† | is_female | −0.028 | 0.020 | Female-gendered prompts produce slightly lower entropy than male-gendered prompts (−0.028 SD), holding model and temperature constant. |
| mean_entropy† | is_sft | −0.073 | < 0.001 | The SFT model produces lower entropy than the base model (−0.073 SD), holding gender and temperature constant. |
| mean_entropy† | is_rlvr | −0.138 | < 0.001 | The RLVR model produces the largest entropy reduction relative to base (−0.138 SD), suggesting RL training most strongly constrains generation uncertainty. |
| mean_entropy† | is_female:is_sft | +0.046 | 0.008 | The gender gap in entropy (female < male) is partially offset in the SFT model — female prompts produce 0.046 SD more entropy in SFT than the base gender gap would predict. |
| mean_entropy_nucleus† | temperature | +0.979 | < 0.001 | A one-SD increase in temperature is associated with a 0.98-SD increase in nucleus entropy — mirrors the full-distribution entropy finding closely. |
| mean_entropy_nucleus† | is_female | −0.029 | 0.028 | Female-gendered prompts produce slightly lower nucleus entropy than male-gendered prompts (−0.029 SD), consistent with the full-entropy result. |
| mean_entropy_nucleus† | is_sft | −0.076 | < 0.001 | The SFT model produces lower nucleus entropy than the base model (−0.076 SD). |
| mean_entropy_nucleus† | is_rlvr | −0.148 | < 0.001 | The RLVR model produces the largest nucleus entropy reduction relative to base (−0.148 SD). |
| mean_entropy_nucleus† | is_female:is_sft | +0.047 | 0.011 | As with full entropy, the SFT model partially offsets the female entropy disadvantage (+0.047 SD), suggesting SFT treats gendered prompts more uniformly than base. |
| semantic_div | temperature | +0.653 | < 0.001 | A one-SD increase in temperature is associated with a 0.65-SD increase in semantic diversity — the strongest driver of semantic diversity after model type. |
| semantic_div | is_sft | +0.200 | < 0.001 | The SFT model produces slightly higher semantic diversity than the base model (+0.200 SD). |
| semantic_div | is_dpo | −0.595 | < 0.001 | The DPO model produces substantially lower semantic diversity than the base model (−0.595 SD). |
| semantic_div | is_rlvr | −0.764 | < 0.001 | The RLVR model produces the lowest semantic diversity of all models (−0.764 SD relative to base), suggesting RL training most strongly constrains the range of meanings generated. |
| semantic_div | is_female:is_dpo | −0.122 | 0.044 | Within the DPO model specifically, female-gendered prompts produce 0.122 SD less semantic diversity than male-gendered prompts — a gender gap not seen in other models. |
| self_bleu† | temperature | −0.924 | < 0.001 | A one-SD increase in temperature is associated with a 0.924-SD decrease in self-BLEU — higher temperature produces more surface-level variation across generated sequences, as expected. |
| self_bleu† | is_dpo | −0.062 | 0.015 | The DPO model produces slightly lower self-BLEU than the base model (−0.062 SD), meaning its responses are marginally more surface-diverse across generations. |
| self_bleu† | is_rlvr | +0.103 | < 0.001 | The RLVR model produces higher self-BLEU than the base model (+0.103 SD), meaning its responses are more surface-repetitive across generations — despite having low semantic diversity, the phrasing patterns are also more constrained. |
| perplexity† | temperature | +0.663 | < 0.001 | A one-SD increase in temperature is associated with a 0.663-SD increase in perplexity — higher temperatures produce more surprising token choices. |
| perplexity† | is_female | −0.114 | 0.004 | Female-gendered prompts produce lower perplexity than male-gendered prompts (−0.114 SD), suggesting the model finds female-gendered outputs less surprising on average. |
| perplexity† | is_sft | +0.284 | < 0.001 | The SFT model produces higher perplexity than the base model (+0.284 SD). |
| perplexity† | is_dpo | +1.049 | < 0.001 | The DPO model produces substantially higher perplexity than the base model (+1.049 SD) — the largest model-level effect in this regression. |
| perplexity† | is_rlvr | +0.919 | < 0.001 | The RLVR model also produces substantially higher perplexity than the base model (+0.919 SD), comparable to DPO. |
| perplexity† | is_female:is_sft | +0.121 | 0.033 | Within the SFT model, female-gendered prompts produce 0.121 SD more perplexity than the base gender gap predicts — the female perplexity advantage is reduced or reversed in SFT. |
| perplexity† | is_female:is_rlvr | +0.131 | 0.021 | Within the RLVR model, female-gendered prompts produce 0.131 SD more perplexity than the base gender gap predicts — similar pattern to SFT, suggesting alignment training partially reverses the female perplexity reduction seen in base. |

† Metric was Box-Cox transformed before standardisation. Coefficient magnitudes reflect SDs of the transformed scale.

---

## Non-significant Gender Effects (p ≥ 0.05)

The following gender-related predictors did not reach significance across any metric and should not be interpreted as evidence of an effect:

- `is_female` for semantic_div (p = 0.682) and self_bleu (p = 0.489)
- All `is_female:is_dpo` interactions except semantic_div
- All `is_female:is_rlvr` interactions except perplexity
- `is_dpo` main effect for entropy metrics

---

## Notes on Multiple Testing

These results span 5 separate regressions. If applying Benjamini-Hochberg correction across the 5 F-statistic p-values, all models remain significant (all Prob(F) ≈ 0.00). For coefficient-level inference, borderline findings (e.g. `is_female:is_dpo` in semantic_div at p = 0.044, `is_female:is_sft` in perplexity at p = 0.033) should be interpreted cautiously pending formal correction.
