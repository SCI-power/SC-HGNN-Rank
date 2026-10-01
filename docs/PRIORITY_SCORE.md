# Candidate priority

The post-lock priority uses the original seven-component weights:

`0.20*model + 0.15*spatial + 0.10*clinical + 0.20*directness + 0.20*external + 0.10*protein_genomic + 0.05*potency`.

`model` is the original `main_axis_freeze_score`, not an OOF prediction or an external score. `external` is the mean of evaluable HPA/GDSC/PRISM/CTRP layers; all other inputs are unchanged. The same formula is recomputed for all 24 retained candidates. Current ranks refer only to the eight retained candidates per cancer, not to new model predictions.

| Cancer | External mean | Coverage | Composite priority |
|---|---:|---:|---:|
| COAD | 0.746151 | 4/4 | 0.807260 |
| LIHC | 0.644568 | 4/4 | 0.710672 |
| STAD | 0.675406 | 2/4 | 0.768062 |

The three selected axes remain first in their respective retained-candidate pools. Protein-stability and cellular experiments are reported separately and do not enter this score.

The original fixed-four priority is retained as `legacy_priority_score`. For STAD its full-precision source value is 0.7005210549374761.

```bash
python src/recompute_priority_scores.py --output outputs/priority_scores
python -m unittest discover -s tests -p test_priority_scores.py -v
```
