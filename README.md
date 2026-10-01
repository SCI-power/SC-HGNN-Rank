# SC-HGNN-Rank

Spatial-Context Heterogeneous Graph Neural Network Ranker for prioritizing
compound-target-regulon-cellular-context axes in COAD, LIHC and STAD.

This package contains the study model, processed inputs, fixed splits, saved
predictions, model checkpoints and analysis scripts. The default workflow uses
the original SC-HGNN-Rank architecture and the matched evaluations reported in
the manuscript.

## Installation

Use Python 3.12. Extract the ZIP before running the commands. For a CPU setup on
Windows or Linux:

```bash
python -m venv .venv
```

Activate it with `.venv\Scripts\Activate.ps1` in Windows PowerShell, or
`source .venv/bin/activate` on Linux/macOS. Then install the dependencies:

```bash
python -m pip install -r environment/requirements-cpu.txt
```

For macOS use
`python -m pip install -r requirements.txt`. The tested CUDA 12.6 setup is
specified in `environment/requirements-cuda126.txt`. A GPU is recommended for
full training; all quick checks can run on CPU.

## Quick reproducibility check

Run from the extracted directory:

```bash
python run.py smoke --device cpu --output outputs/quick_check
```

This validates the 600 candidates and 13 fixed folds, rebuilds the graph and
tensors, checks checkpoint inference against saved test predictions, recomputes
external-evidence and priority scores, and trains each of the eight comparators
for three epochs. Success is recorded in `outputs/quick_check/SMOKE_SUCCESS.json`.
Short training outputs are separate from the study results.

The graph contains 960 nodes, 9,576 typed edges and 873 relation-disruption
specifications. The model uses a hidden dimension of 112, two message-passing
layers and dropout of 0.18. The parameter-matched evidence-only MLP has a hidden
width of 196. The saved seeds are 20260807-20260811.

## Reproduce results

```bash
python run.py validate --output outputs/input_check
python run.py prepare --output outputs/rebuilt_inputs
python run.py infer --output outputs/checkpoint_predictions
python run.py scores --output outputs/evidence_scores
python run.py weights --output outputs/score_weight_sensitivity
python run.py summary --output outputs/reference_summary
```

`summary` recomputes the matched comparisons, paired tests, training sensitivity,
candidate uncertainty, TopK results and external-resource matching estimates
from the included formal run records. It does not retrain the model. External
matching confidence intervals use the recorded 5,000 bootstrap replicates.

`weights` repeats the deterministic and 2,000 random integrated-score weight
perturbations without retraining the neural network.

`infer` uses the first saved main-model checkpoint unless `--checkpoint` is
specified. It exports predictions for all 600 candidates and checks the held-out
rows against that checkpoint's recorded predictions. Checkpoint paths are listed
in `checkpoints/index.json`; 65 main-model checkpoints are supplied.

## Full training and additional analyses

```bash
python run.py train --phase all --device auto --output outputs/full_training
python run.py summary --results outputs/full_training --output outputs/new_summary
python run.py plot --output outputs/new_summary
python run.py ablation --mode check --output outputs/ablation_check
python run.py graph-only --mode check --output outputs/graph_only_check
python run.py ablation --mode train --device auto --output outputs/ablation_training
python run.py graph-only --mode train --device auto --output outputs/graph_only_training
python run.py ablation --mode explain --device cpu --output outputs/evidence_dependence
```

Full model evaluation comprises 520 matched fits, 210 additional sensitivity
fits and 78 missing-evidence fits. Thirty baseline fits are reused for the
sensitivity analysis. Module ablation and the graph-only control comprise 260
and 65 additional fits, respectively. Completed formal run records are included
for all of these analyses. The training commands save new outputs rather than
altering the packaged reference results.

Reference source panels can be regenerated with:

```bash
python run.py ablation --mode summary --output outputs/ablation_panels
python run.py graph-only --mode summary --output outputs/graph_only_panels
```

Source-panel filenames retain their analysis identifiers. Their correspondence
to the final manuscript is documented in `docs/FIGURE_SOURCE_MAP.md`.

## Files and inputs

| Directory | Contents |
| --- | --- |
| `src/` | Model, losses, graph/tensor construction, evaluation and scoring |
| `data/model/` | Candidate labels, graph tensors and comparison specifications |
| `data/project/` | Harmonized evidence tables used to rebuild the graph |
| `data/graph/` | Readable node, edge, label and comparison tables |
| `data/supporting/` | Evidence components and retained-candidate priority tables |
| `data/reference_runs/` | Fixed split assignments and source-level evidence |
| `reference_results/model/` | Matched, sensitivity and missing-evidence run records |
| `reference_results/ablation/` | Module-ablation and feature-masking results |
| `reference_results/graph_only/` | Graph-only control results |
| `analyses/` | Module-ablation, explanation and graph-only scripts |
| `checkpoints/index.json` | Paths to the 65 saved main-model checkpoints |
| `environment/` | Requirements and the tested software environment |
| `external_tools/` | Optional upstream spatial-tool adapters |

The executable workflow starts from the supplied processed evidence tables.
Raw sequencing files and bulk third-party database downloads remain at their
original public sources. Input/output formats and spatial adapter instructions
are provided in `docs/INPUT_OUTPUT.md` and `docs/EXTERNAL_TOOLS.md`.

Discovery model scores, matched out-of-fold predictions and retained-candidate
priority scores are separate fields. External means use evaluable sources only,
retain measured zeros and report source coverage. Selected priority scores are
0.807260 (COAD), 0.710672 (LIHC) and 0.768062 (STAD).

Scientific identifiers and legacy class names are retained for checkpoint
compatibility. File paths used for execution are package-relative. See
`TERMS.md` for the author's existing software and data-use terms.

## Explorer and additional data

The companion `SC_HGNN_Rank_Explorer.html` is a standalone file containing its
data and images. Open it directly in a browser to search candidates, inspect
evidence, adjust display weights and export tables. No web server is required.

The companion additional-data archive includes the spatial source crosswalks,
all 49 mapping-weight tables and the supplementary source-data workbook,
including the reported cellular-experiment measurements.
