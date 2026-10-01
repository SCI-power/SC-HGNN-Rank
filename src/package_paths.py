from pathlib import Path
import os

ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / "src"
DATA = Path(os.environ.get("SC_HGNN_DATA_DIR", ROOT / "data")).resolve()
INPUTS = DATA / "model"
ARCHIVE = DATA / "reference_runs"
OUTPUT = Path(os.environ.get("SC_HGNN_OUTPUT_DIR", ROOT / "outputs")).resolve()
RESULTS = Path(os.environ.get("SC_HGNN_RESULTS_DIR", ROOT / "reference_results/model")).resolve()
if OUTPUT == DATA or DATA in OUTPUT.parents:
    raise ValueError("Output must not be inside the packaged input data directory.")
