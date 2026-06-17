# MAMS Relation, WSD, and Scope Experiments

## Conclusion

The stable improvement comes from sentence-level relation attention across
all aspects. It uses one sentiment cross-entropy objective and reaches:

- **Train + validation refit test Macro-F1: 87.5784**
- **Test accuracy: 88.0240**
- Gain over the matched baseline: **+2.67 points**
- Sentence-cluster bootstrap 95% CI: **[+1.26, +4.10]**

This remains the recommended model. WSD pair biases and aspect-to-scope token
pooling improved individual validation runs, but neither survived the locked
train-plus-validation refit. Later structural, backbone, residual-correction,
and class-balance experiments also failed to improve the final result.

## Relation Forward

Each `(sentence, aspect)` pair is encoded by RoBERTa. Aspect representations
from the same sentence are packed into one sequence and passed through
multi-head relation attention before sentiment classification. This directly
models interactions among aspects instead of adding pair, diversity, or
consistency losses.

## WSD Forward

GlossBERT scores WordNet candidates and frozen RoBERTa gloss embeddings form a
soft sense prototype. Version 2 fixes an important preprocessing error:

- no WordNet candidate: zero prototype and zero confidence;
- one candidate: its gloss prototype and confidence 1;
- multiple candidates: confidence-weighted soft prototype.

This increases nonzero WSD coverage from 10,700 to 13,356 of 13,854 aspects.

For aspect pair `(i, j)`, the full model can add two features to each
attention head:

1. pairwise WSD reliability;
2. reliability-weighted gloss-prototype cosine.

The `orthogonal` mode removes the reliability-correlated component of cosine
using statistics estimated from training pairs only. The `semantic-only`
control removes reliability entirely, so any improvement must come from sense
identity rather than confidence.

## Validation Ablation

| Method | Validation Macro-F1 | Locked epoch |
|---|---:|---:|
| Baseline | 84.2300 | 3 |
| Relation attention | 85.0649 | 2 |
| WSD raw v2 | **86.0103** | 5 |
| WSD orthogonal v2 | 85.8023 | 3 |
| WSD semantic-only v2 | 85.4237 | 3 |
| Scope + relation | 85.7882 | 2 |
| Scope + WSD relation | 85.6439 | 2 |

Relative to relation-only, the raw-v2 validation gain is 0.95 points, but its
sentence-cluster bootstrap CI `[-0.56, +2.51]` includes zero. Semantic-only is
0.36 points higher with CI `[-1.10, +1.84]`. Sense identity therefore has a
weak signal, not a statistically established improvement.

## Structural Follow-up

Error slicing showed that the original relation block helped two-aspect and
five-or-more-aspect sentences, but hurt middle aspects and four-aspect
sentences. The first structural change therefore made aspect order explicit:

- `cross` excludes self-attention inside the relation branch while preserving
  the residual path;
- `relative` adds a learned signed aspect-rank bias to every attention head;
- `cross-relative` combines both changes.

A second change kept the independent aspect classifier and initialized a
separate relation correction head to zero. This makes the initial model
exactly equivalent to independent classification and lets relation evidence
change only the final logits.

| Follow-up | Validation Macro-F1 | Best epoch |
|---|---:|---:|
| Original relation | 85.0649 | 2 |
| Cross only | 85.1450 | 2 |
| Relative rank only | 85.2624 | 4 |
| **Cross-relative** | **86.4116** | 3 |
| Residual-logit correction | 85.6212 | 2 |
| Residual-context correction | 84.5375 | 3 |
| Residual-logit + cross-relative | 84.3833 | 5 |
| Square-root class-balanced CE | 85.4029 | 4 |
| RoBERTa-large relation | 84.6390 | 4 |
| DeBERTa-v3-base relation | 85.3621 | 3 |
| DeBERTa-v3-base cross-relative | 84.1307 | 3 |

Cross-relative improves validation by 1.35 points with a sentence-cluster
bootstrap CI of `[+0.0045, +2.6743]`. It repairs part of the
group-size-three, group-size-four, and middle-aspect regressions.
Residual-logit improves validation by 0.56 points, but its CI
`[-0.78, +1.92]` includes zero.

## Final Refit

Each configuration was retrained on train plus validation for its
validation-locked epoch count. Each individual run was fixed before its test
evaluation. However, the structural follow-ups were designed after earlier
test results were already known, so they are exploratory rather than a
pristine single-shot test comparison.

| Method | Test Macro-F1 | Test accuracy |
|---|---:|---:|
| Baseline, 3 epochs | 84.9109 | 85.2545 |
| **Relation attention, 2 epochs** | **87.5784** | **88.0240** |
| WSD raw v2, 5 epochs | 86.1839 | 86.5269 |
| WSD orthogonal v2, 3 epochs | 86.7111 | 87.1257 |
| WSD semantic-only v2, 3 epochs | 85.6397 | 86.0778 |
| Scope + relation, 2 epochs | 85.9127 | 86.4521 |
| Cross-relative, 3 epochs | 86.1157 | 86.6018 |
| Cross-relative, 2,151 updates | 86.7764 | 87.2754 |
| Residual-logit, 2 epochs | 86.2870 | 86.7515 |

Paired sentence-cluster bootstrap differences against relation-only:

| Comparison | Difference | 95% CI |
|---|---:|---:|
| Raw WSD v2 - relation | -1.39 | [-2.60, -0.17] |
| Orthogonal WSD v2 - relation | -0.87 | [-2.03, +0.28] |
| Semantic-only WSD v2 - relation | -1.94 | [-2.98, -0.90] |
| Scope relation - relation | -1.67 | [-2.98, -0.36] |
| Cross-relative update-matched - relation | -0.80 | [-2.01, +0.39] |
| Residual-logit - relation | -1.29 | [-2.43, -0.17] |

## Interpretation

WSD confidence is easier to exploit than sense geometry. In the raw model,
reliability coefficients are much larger than cosine coefficients.
Orthogonalization improves identifiability and gives the best WSD refit, but
it still does not exceed relation-only.

Parameter-free scope pooling also overfits validation. The relation encoder
already receives an aspect-conditioned RoBERTa representation, so an
additional token evidence pooling operation appears redundant on MAMS.

Explicit relative order produces the strongest new validation result, but its
gain is not stable when validation examples are merged into training.
Residual correction, class balancing, RoBERTa-large, and DeBERTa-v3-base do
not solve this transfer problem. The result is therefore not a capacity or
neutral-class-frequency bottleneck.

The negative results narrow the next direction: a further claim should be
validated across multiple public multi-aspect datasets and should learn
relation structure from a transferable supervision signal, rather than tune
more MAMS-specific attention or loss variants.

## Literature Check

As checked on June 14, 2026:

- [SVM-ATSA, Findings of EMNLP 2024](https://aclanthology.org/2024.findings-emnlp.340/)
  reports 86.77 MAMS Macro-F1 under its main preprocessing.
- [GHI, arXiv 2026](https://arxiv.org/abs/2605.22228) reports 84.42 on MAMS;
  its strongest gains are on other ABSA benchmarks.
- [Aspect-to-Scope Multi-view Learning, EMNLP 2023](https://aclanthology.org/2023.findings-emnlp.727/)
  motivates explicit aspect evidence binding.
- [Context Denoising, NAACL 2024](https://aclanthology.org/2024.findings-naacl.194/)
  studies denoising aspect-irrelevant context.

The 87.5784 result is numerically above the directly reported 86.77 value, but
the training and model-selection protocols are not identical. It should be
described as a strong result, not an unqualified external SOTA claim.

## Protocol

- Fixed seed 42; no seed search.
- One supervised loss: sentiment cross-entropy.
- Architecture and epoch counts selected on official MAMS validation only.
- Each configuration was locked before its own test evaluation.
- Structural follow-ups are exploratory because earlier test results were
  already available.
- Bootstrap resamples complete sentences, preserving dependent aspect labels.
- Large caches and checkpoints live outside Git under `WSDAF_STORAGE`.

Aggregate metrics are stored in
`experiments/results/mams_sense_relational_summary.json`.
