# Upstream tools and interfaces

The supplied core pipeline is self-contained from harmonized evidence tables.
It does not require RCTD, SPOTlight, Tangram, SCENIC or a database API to retrain
the published candidate graph. Preprocessing precedes the supplied model inputs.

Optional original spatial adapters are in `external_tools/`. They preserve the
source analysis logic, with the machine-specific root replaced by an environment
variable. They have not been rerun on full raw matrices during package testing.

## RCTD and SPOTlight

Set `SC_HGNN_ANALYSIS_READY` to a directory containing
`spatial_mapping/pilot_inputs/spatial_mapping_pilot_manifest.csv`.
The manifest identifies `cancer`, `sample_id`, `reference_dir` and `spatial_dir`.
Bundle matrices use genes x cells/spots, `counts.mtx`, `genes.tsv`, and
`cells.tsv` or `spots.tsv`, with the reference labels and spatial coordinates
expected in the adapter. Read the script's column checks before importing a new
dataset. Matrix cell/spot identifiers must match annotation records.

```powershell
$env:SC_HGNN_ANALYSIS_READY = "./analysis_ready"
$env:SPATIAL_PILOT_CORES = "4"
Rscript external_tools/run_rctd_spotlight_spatial_pilot.R
```

R packages used by the original adapter: Matrix, spacexr, SPOTlight, ggplot2,
data.table. No reconstructed exact R package versions are asserted. Use an
appropriate separate R environment and save `sessionInfo()` for a new run.

## Tangram projection

The Python adapter uses the same bundle manifest and a separate environment with
anndata, scanpy, scipy, numpy, pandas, torch and tangram. Configure the same root:

```powershell
python external_tools/run_tangram_candidate_projection_pilot.py
```

This is the original target/regulatory projection implementation. Its output
should not be relabelled as a new cell-type mapping method or an independently
measured spatial ground truth. See the preserved adapter for normalization and
mapping settings.

## Importing processed evidence

Gene-expression, regulon, spatial, interaction-network and drug-resource outputs
are joined to the harmonized axis table using explicit cancer, gene, compound
and axis identifiers. `docs/data_dictionary.csv` records columns and observed
types/missingness for every supplied model and graph input. Provenance is
retained in the table names, source columns and `docs/source_provenance.json`.
Supplied model inputs are already standardized; applying an upstream method
again and overwriting these scores would create a new analysis, not an exact
reproduction of the frozen model.
