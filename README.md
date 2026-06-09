# SC-HGNN-Rank

SC-HGNN-Rank is a heterogeneous graph ranker for prioritizing candidate
compound-target-regulon-cell-context axes. The code in this repository is a
review-facing implementation of the model architecture, ranking objective,
input schema, training entry point and candidate-ranking workflow.

The repository is intended to support manuscript review and reproducibility
inspection. It does not contain local raw omics data, private paths, manuscript
build scripts or image-generation assets.

## Repository layout

```text
configs/            Example run configuration
docs/               Model, schema and reproducibility notes
examples/           Small runnable example tables
scripts/            Command-line entry points
src/sc_hgnn_rank/   Model, graph construction, losses and metrics
tests/              Lightweight smoke tests
```

## Install

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Linux and macOS users can activate the environment with
`source .venv/bin/activate`.

## Run a smoke training job

```bash
python scripts/train_model.py --config configs/example_config.json
```

The command writes a trained checkpoint, metrics and ranked candidate table
under `outputs/example_run/`.

For a minimal code-path check without additional test frameworks:

```bash
python scripts/smoke_test.py
python scripts/reviewer_audit.py
```

## Rank candidate axes with a trained checkpoint

```bash
python scripts/rank_candidates.py ^
  --config configs/example_config.json ^
  --checkpoint outputs/example_run/model.pt ^
  --out outputs/example_run/ranked_candidates.csv
```

## Input format

The minimum candidate-axis table requires:

- `cancer`
- `compound`
- `target`
- `regulon`
- `cell_context`

Recommended evidence-score columns are normalized to the range 0-1:

- `drug_evidence`
- `single_cell_regulatory`
- `spatial_recurrence`
- `mapping_consistency`
- `ppi_msi`
- `clinical`
- `lincs_cmap`
- `depmap`
- `non_broad_sensitivity`

See `docs/input_schema.md` and `examples/example_candidate_axes.csv`.

## Claim boundary

SC-HGNN-Rank produces candidate-axis priority scores. A high score supports
mechanism prioritization for follow-up review; it is not direct evidence of
therapeutic efficacy, clinical utility or biochemical binding. External
evidence layers should be interpreted according to their source-specific
coverage and assay boundaries.

## Reproducibility notes

The example data included here are intentionally small so reviewers can inspect
the code path quickly. Full manuscript analyses require the public datasets and
evidence tables described in the manuscript and supplementary materials.
