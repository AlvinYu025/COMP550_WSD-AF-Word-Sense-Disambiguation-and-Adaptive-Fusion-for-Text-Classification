# MAMS Multi-Aspect Experiments

This directory contains the MAMS-ATSA experiments used to validate relation
attention, WSD pair biases, aspect-to-scope pooling, and structural follow-up
variants.

## Best Result

The recommended model is `mams_sense_relational.py --method relation` with a
two-epoch train-plus-validation refit:

| Model | Test Macro-F1 | Test accuracy |
|---|---:|---:|
| RoBERTa baseline | 84.9109 | 85.2545 |
| Relation attention | **87.5784** | **88.0240** |
| WSD raw v2 | 86.1839 | 86.5269 |
| WSD orthogonal v2 | 86.7111 | 87.1257 |
| Scope + relation | 85.9127 | 86.4521 |
| Cross-relative relation | 86.7764 | 87.2754 |
| Residual-logit relation | 86.2870 | 86.7515 |

The relation-only result is the current recommended result. The structural
follow-ups produced stronger validation runs but did not survive the locked
train-plus-validation refit.

## Setup

Install the shared joint-experiment dependencies and the additional WSD
dependencies:

```bash
pip install -r experiments/requirements-joint.txt
pip install -r experiments/requirements-sense-relational.txt
python -m spacy download en_core_web_sm
```

The runner downloads MAMS XML files, Hugging Face models, NLTK data, and
prototype caches automatically. Keep those artifacts outside Git:

```bash
export WSDAF_STORAGE=/path/to/persistent/storage/wsdaf-sense-relational
```

If `WSDAF_STORAGE` is not set, the script uses `~/private/wsdaf-sense-relational`
when available, otherwise `~/.cache/wsdaf-sense-relational`.

## Reproduce The Recommended Result

First select methods on the official validation split:

```bash
python experiments/mams_sense_relational.py --method baseline
python experiments/mams_sense_relational.py --method relation
python experiments/mams_sense_relational.py --method full --run-tag wsd-raw-v2
python experiments/mams_sense_relational.py \
  --method full \
  --pair-feature-mode semantic-only \
  --run-tag semantic-only-v2
```

Then run the locked train-plus-validation refit:

```bash
python experiments/mams_sense_relational.py \
  --method relation \
  --epochs 2 \
  --train-on-validation \
  --evaluate-test
```

## Structural Follow-Ups

The runner includes two relation-structure controls:

```bash
python experiments/mams_sense_relational.py \
  --method relation \
  --relation-attention-mode cross-relative \
  --run-tag rank-cross-relative

python experiments/mams_sense_relational.py \
  --method relation \
  --relation-output-mode residual-logit \
  --run-tag residual-logit
```

`cross-relative` excludes self-attention inside the relation branch and adds a
learned signed relative-rank attention bias. `residual-logit` keeps the
independent aspect classifier and initializes the relation correction head to
zero, so relation evidence can only add a learned logit correction.

These variants are useful ablations, but neither replaces the relation-only
model under the final refit protocol.

## Reports

- `experiments/results/mams_sense_relational_report.md`: detailed narrative
  report, ablations, bootstrap intervals, and interpretation.
- `experiments/results/mams_sense_relational_summary.json`: compact metrics for
  automated checks.
- `experiments/results/mams_joint_relations_report.md`: earlier joint relation
  experiment with auxiliary pair/diversity objectives.

## Protocol Notes

- Fixed seed 42; no seed search.
- One supervised sentiment cross-entropy objective for
  `mams_sense_relational.py`.
- Architecture and epoch counts are selected on the official MAMS validation
  split before each run's test evaluation.
- Later structural follow-ups are exploratory because earlier MAMS test results
  were already known.
- Checkpoints, model downloads, WSD caches, and generated prototypes are not
  committed.
