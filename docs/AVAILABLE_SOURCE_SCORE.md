# External support among evaluable sources

The revised summary averages the original HPA, GDSC, PRISM and CTRP layer scores
over sources with evaluable matched records:

`available_source_external_score = sum(I_j * S_j) / sum(I_j)`

`I_j` equals one when an evaluable record is present, and zero otherwise.
The source scores and the original compound/target/regulator matching hierarchy
are unchanged. An observed score of zero contributes to both the numerator and
denominator. A missing matched measurement is represented as NA and omitted.
If all four sources are unavailable, the result is NA.

Availability is determined from the source-detail tables, not by testing whether
a score is greater than zero. Drug-response sources require a matched detail
record with a finite response and positive record count. HPA requires a
gene-level evidence record for the target or regulator. The same definition is
applied to all 600 candidates.

| Selected axis | Original fixed-four mean | Available-source mean | Coverage |
|---|---:|---:|---:|
| COAD: Quercetin-CDK1/MYB | 0.7462 | 0.7462 | 4/4 |
| LIHC: 4',5-DHF-AR/HDAC1 | 0.6446 | 0.6446 | 4/4 |
| STAD: Eriodyctiol-HSP90AB1/HOPX | 0.3377 | 0.6754 | 2/4 |

The STAD calculation uses HPA (0.5375) and PRISM (0.8133114640883978).
The GDSC and CTRP detail tables contain no matched records for this axis.
PRISM remains target-class support from NVP-AUY922; averaging does not turn it
into an Eriodyctiol direct-compound measurement.

Report `n_evaluable_databases` or `database_coverage_fraction` alongside the
mean. The mean describes support within observed sources; source coverage
describes how broadly the candidate was represented. A high mean from one
source is not equivalent to the same mean supported by four sources.

## Recompute

```bash
python src/recompute_available_external_scores.py --output outputs/available_source_scores
python -m unittest discover -s tests -p test_available_external_scores.py -v
```

Tables for all 600 candidates, the retained 24, and the selected three are under
`data/supporting/available_source_external_scores/`. Original fixed-four scores
remain in the archived model and result files. Original training inputs,
model weights, spatial coefficients and experimental values are unchanged.

This is a revised external-score definition, not an increase caused by new experimental measurements. The updated component is propagated into the retained 24-axis composite priorities by `src/recompute_priority_scores.py`. Model freezing scores are separate and unchanged.

## Methods wording

External support was summarized across HPA, GDSC, PRISM and CTRP using the
unweighted mean of the available source-level scores. A source was considered
evaluable when a relevant HPA gene-evidence record or a matched drug-response
record with a finite measurement was available. Sources without matched
measurements were coded as missing and excluded from the denominator; observed
zero scores were retained. The number of evaluable databases was reported
alongside each summary score. The same rule was applied to all candidate axes,
with the original fixed-four-source score retained for comparison.
