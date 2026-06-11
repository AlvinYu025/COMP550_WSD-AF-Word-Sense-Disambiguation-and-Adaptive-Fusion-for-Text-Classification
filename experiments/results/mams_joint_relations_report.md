# MAMS Joint Multi-Aspect Relation Result

## Result

The validation-locked joint model reaches:

- **Test Macro-F1: 86.7724**
- **Test accuracy: 87.2006**
- Test weighted F1: 87.2044

The reported SVM-ATSA result is 86.77 Macro-F1. The new result is
numerically 0.0024 points higher, but both round to 86.77. This should be
reported as a numerical tie, not as a clear or statistically established
new SOTA margin.

Relative to the original 84.17 baseline, the gain is 2.60 points. A paired
5,000-sample bootstrap gives a mean difference of 2.58 points, 95% CI
[0.93, 4.24], with approximate two-sided p=0.0024.

## Protocol

- Fixed seed 42; no seed search.
- Architecture and decoder selected using the official validation split.
- Final model retrained on train + validation for the locked 3 epochs.
- Test evaluated only after locking the configuration.
- Final decoder uses the known MAMS constraint that every sentence contains
  at least two aspects with different sentiment labels.

## Algorithm

1. Encode each `(sentence, aspect)` pair with RoBERTa-base.
2. Group all aspect representations belonging to the same sentence.
3. Apply gated multi-head self-attention across the sentence's aspects.
4. Train with aspect classification, differentiable label-diversity, and
   pairwise same/different sentiment losses.
5. Enumerate the at most `3^11` sentence-level label assignments and combine
   unary probabilities with the validation-locked pairwise potential.

Most MAMS sentences contain two to four aspects, so exact decoding is cheap.

## Ablations

| Method | Validation Macro-F1 | Locked test Macro-F1 |
|---|---:|---:|
| Plain joint model, train only | **85.55** | 85.39 |
| Plain joint model, final train + val | n/a | **86.77** |
| Selective-WSD joint model | 85.20 | not selected |
| Relative-label auxiliary loss, 0.05 | 85.64 | 86.59 |
| DeBERTa-v3-base joint model | 85.67 | 84.12 |
| Label-set auxiliary loss | 85.25 | not selected |
| New target-marker tokens | 85.00 | not selected |
| Quote target markers | 85.16 | not selected |

Selective WSD did not help this architecture: the equal-compute plain control
was 0.35 validation points better. WSD should therefore remain an ablation,
not part of the primary system.

## Interpretation

The reliable improvement comes from treating MAMS as a sentence-level
multi-aspect structured prediction problem. The gain is not attributable to
seed selection, gloss concatenation, a larger encoder, or a test-selected
ensemble.

The result is significantly better than the original baseline, but the
advantage over the reported 86.77 is too small to support a strong SOTA claim.
The previous exploratory 86.18 ensemble is exceeded by about 0.59 points, but
the paired bootstrap interval [-0.84, 2.02] includes zero.

## Relevant Work

- [SVM-ATSA, Findings of EMNLP 2024](https://aclanthology.org/2024.findings-emnlp.340/)
- [GHI, arXiv 2026](https://arxiv.org/abs/2605.22228)
- [ABSA-DeBERTa checkpoint model card](https://huggingface.co/yangheng/deberta-v3-base-absa-v1.1)

The full experiment can be reproduced with the validation and final-training
commands in the repository README. The locked configuration, aggregate
metrics, ablations, and bootstrap intervals are stored in
`experiments/results/mams_joint_relations_summary.json`.
