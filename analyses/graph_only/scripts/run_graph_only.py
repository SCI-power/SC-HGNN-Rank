"""Fixed-protocol GraphOnly control for Supplementary Fig. S3L."""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch

SOURCE_DIR = Path(__file__).resolve().parent
BASE = Path(__file__).resolve().parents[3]
ROOT = Path(os.environ.get("SC_HGNN_ANALYSIS_OUTPUT", BASE / "outputs/graph_only")).resolve()
MODEL_RESULTS = Path(os.environ.get("SC_HGNN_RESULTS_DIR", BASE / "reference_results/model")).resolve()
ORIGINAL = BASE / "data/reference_runs/source_original/calibrate_graph_only_hgnn_and_feature_baselines.py"
sys.path.insert(0, str(BASE / "src"))
import reviewer_matched_rerun as core
from run_original_supplement import check_splits
from graph_only_model import GraphOnlySpatialCausalHGNN

SEEDS = [20260807, 20260808, 20260809, 20260810, 20260811]
VARIANT = "GraphOnly-HGNN-simple"
COMPARATORS = ["GCN-simple", "R-GCN-simple", "GAT-simple", "HGT-like-simple"]
CFG = core.TrainConfig()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for data in iter(lambda: handle.read(1048576), b""):
            h.update(data)
    return h.hexdigest()


def dump(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=str, allow_nan=False), encoding="utf-8")


def class_signature(path):
    tree = ast.parse(Path(path).read_text(encoding="utf-8-sig"))
    node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "GraphOnlySpatialCausalHGNN")
    return hashlib.sha256(ast.dump(node, include_attributes=False).encode()).hexdigest()


def make_graph_only(raw, cfg):
    return GraphOnlySpatialCausalHGNN(int(raw["metadata"]["num_node_types"]),
        int(raw["metadata"]["num_edge_types"]), hidden_dim=cfg.hidden_dim,
        num_layers=cfg.num_layers, dropout=cfg.dropout)


def patch_factory():
    original_make, original_forward = core.make_model, core.forward_model

    def make(variant, raw, axis_dim, meta_dim, cfg):
        if variant == VARIANT:
            return make_graph_only(raw, cfg)
        return original_make(variant, raw, axis_dim, meta_dim, cfg)

    def forward(model, variant, data, x, meta):
        if variant == VARIANT:
            return model(data)
        return original_forward(model, variant, data, x, meta)

    core.make_model, core.forward_model = make, forward


def reference_dir(variant, split, seed):
    return MODEL_RESULTS / "main/fits" / f"baseline__{variant}__{split.fold_id}__{seed}"


def prepare():
    torch.set_num_threads(2)
    raw = torch.load(BASE / "data/model/spatial_causal_hgnn_tensor_dataset_2026-05-04.pt",
                     map_location="cpu", weights_only=False)
    labels = pd.read_csv(BASE / "data/model/Table_HGNN3_axis_label_table.csv").reset_index(drop=True)
    decoys = pd.read_csv(BASE / "data/model/Table_HGNN7_negative_axis_decoy_pool.csv")
    splits = pd.read_csv(MODEL_RESULTS / "frozen_splits.csv")
    assert len(labels) == 600 and len(splits) == 13
    assert np.allclose(raw["axis_soft_score"], labels.soft_rank_label)
    checks = check_splits(labels, splits)
    assert class_signature(ORIGINAL) == class_signature(SOURCE_DIR / "graph_only_model.py")
    graph = make_graph_only(raw, CFG)
    assert len(graph.layers) == 2
    assert not hasattr(graph, "axis_encoder") and not hasattr(graph, "meta_encoder")
    graph.eval()
    data = core.add_structural_features(raw)
    with torch.no_grad():
        a = core.forward_model(graph, VARIANT, data, torch.zeros(600, 20), torch.zeros(600, 10))
        b = core.forward_model(graph, VARIANT, data, torch.randn(600, 20), torch.randn(600, 10))
    assert all(torch.equal(x, y) for x, y in zip(a, b))
    assert data["node_struct_features"][raw["axis_node_idx"], 0].eq(0).all()
    parameters = sum(p.numel() for p in graph.parameters())
    refs = []
    files = list((BASE / "data/model").glob("*")) + list((BASE / "src").glob("*.py"))
    files += [MODEL_RESULTS / "frozen_splits.csv", ORIGINAL]
    original_cfg = json.loads((MODEL_RESULTS / "protocol.json").read_text())["config"]
    assert original_cfg == asdict(CFG)
    files += [MODEL_RESULTS / "protocol.json"]
    for comparator in COMPARATORS:
        for seed in SEEDS:
            for _, split in splits.iterrows():
                fit = reference_dir(comparator, split, seed)
                metric = json.loads((fit / "metrics.json").read_text())
                assert metric["objective"] == "simple_cls_reg"
                assert metric["graph_protocol"] == "inductive_edge_masked"
                pred = pd.read_csv(fit / "predictions.csv.gz")
                target = labels.iloc[core.parse_indices(split.test_indices).numpy()].frozen_axis_id
                assert list(pred.frozen_axis_id) == list(target)
                refs.append(metric)
                files += [fit / "metrics.json", fit / "predictions.csv.gz", fit / "history.csv.gz"]
    protocol = {
        "variant": VARIANT, "architecture": "original_GraphOnlySpatialCausalHGNN_class",
        "architecture_ast_sha256": class_signature(ORIGINAL),
        "config": asdict(CFG), "parameter_count": parameters,
        "objective": "simple_cls_reg", "active_loss": "train_weighted_CE + 0.25*train_weighted_smooth_L1",
        "sample_weight_scope": "training_subset_only; same_as_current_GNN_baselines",
        "seeds": SEEDS, "fixed_folds": 13, "planned_new_fits": 65,
        "comparators": COMPARATORS, "reused_comparator_fits": 260,
        "graph_protocol": "entity_edges_masked_in_training_and_validation; full_graph_at_test",
        "extra_edge_dropout": 0.0,
        "direct_axis_or_metapath_vector_input": False,
        "tabular_input_invariance_check": True,
        "graph_contains_prior_weighted_edges": True,
        "scope": "GraphOnly vs standard graph baselines; not a replacement for full SC-HGNN-Rank",
        "primary_statistic": "Spearman difference; seed-average within 13 folds; two-sided Wilcoxon; Holm over 4 comparators",
        "ci": "5000 fold-level percentile bootstrap resamples",
        "test_based_tuning": False,
        "runtime": {"numpy": np.__version__, "pandas": pd.__version__, "torch": torch.__version__,
                    "cuda": torch.version.cuda},
        "source_sha256": {str(p.resolve()): sha(p) for p in sorted(set(files)) if p.is_file()},
    }
    path = ROOT / "protocol.json"
    if path.exists():
        assert json.loads(path.read_text()) == protocol, "Protocol or source changed after locking."
    else:
        dump(path, protocol)
    (ROOT / "tables").mkdir(parents=True, exist_ok=True)
    checks.to_csv(ROOT / "tables/fixed_split_checks.csv", index=False)
    pd.DataFrame(refs).to_csv(ROOT / "tables/reused_standard_gnn_runs.csv", index=False)
    return raw, labels, decoys, splits, protocol


def verify_sources(protocol):
    changed = [p for p, digest in protocol["source_sha256"].items() if sha(p) != digest]
    assert not changed, changed
    return len(protocol["source_sha256"])


def main(limit=None, check_only=False):
    patch_factory()
    raw, labels, decoys, splits, protocol = prepare()
    if check_only:
        print(json.dumps({"checks": "passed", "parameters": protocol["parameter_count"],
                          "source_files_unchanged": verify_sources(protocol)}), flush=True)
        return
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x, meta = torch.empty((600, 0)), torch.empty((600, 0))
    start = time.perf_counter()
    complete = new = 0
    print(json.dumps({"device": str(device), "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU",
                      "new_fits": 65, "objective": "simple_cls_reg"}), flush=True)
    for seed in SEEDS:
        for _, split in splits.iterrows():
            folder = ROOT / "fits" / f"{split.fold_id}__{seed}"
            if (folder / "metrics.json").exists():
                complete += 1
                continue
            folder.mkdir(parents=True, exist_ok=True)
            core.MODEL_DIR = folder
            metric, history, pred, hard = core.train_one(raw, labels, decoys, x, meta, split,
                VARIANT, seed, device, CFG, save_model=True, inductive_graph=True, objective="simple_cls_reg")
            ref = json.loads((reference_dir(COMPARATORS[0], split, seed) / "metrics.json").read_text())
            for key in ["n_train", "n_val", "n_test", "n_train_edges", "n_val_edges", "n_test_edges",
                        "training_top_boost_threshold", "sample_weight_scope", "objective"]:
                assert metric[key] == ref[key], (key, metric[key], ref[key])
            assert metric["parameter_count"] == protocol["parameter_count"]
            for column in ["pairwise_loss", "hard_negative_loss", "listwise_loss", "context_loss"]:
                assert history[column].eq(0).all()
            assert np.allclose(history.train_loss, history.classification_loss + CFG.regression_weight*history.regression_loss)
            for comparator in COMPARATORS:
                rp = pd.read_csv(reference_dir(comparator, split, seed) / "predictions.csv.gz")
                assert list(pred.frozen_axis_id) == list(rp.frozen_axis_id)
            metric["direct_axis_or_metapath_vector_input"] = False
            metric["extra_edge_dropout"] = 0.0
            history.to_csv(folder / "history.csv.gz", index=False)
            pred.to_csv(folder / "predictions.csv.gz", index=False)
            hard.to_csv(folder / "constraint_metadata_unused_in_simple_loss.csv.gz", index=False)
            dump(folder / "metrics.json", metric)
            complete += 1
            new += 1
            dump(ROOT / "progress.json", {"completed": complete, "total": 65})
            print(json.dumps({"completed": complete, "total": 65, "fold": split.fold_id, "seed": seed,
                              "spearman": metric["spearman"], "epochs": metric["epochs"],
                              "elapsed_seconds": round(time.perf_counter()-start, 1)}), flush=True)
            if limit and new >= limit:
                verify_sources(protocol)
                return
    metrics = [json.loads(p.read_text()) for p in sorted((ROOT / "fits").glob("*/metrics.json"))]
    assert len(metrics) == 65
    pd.DataFrame(metrics).to_csv(ROOT / "tables/graph_only_run_metrics.csv", index=False)
    predictions = pd.concat([pd.read_csv(p) for p in sorted((ROOT / "fits").glob("*/predictions.csv.gz"))], ignore_index=True)
    assert len(predictions) == 9000
    assert predictions.groupby("frozen_axis_id").size().eq(15).all()
    predictions.to_csv(ROOT / "tables/graph_only_oof_predictions.csv.gz", index=False)
    dump(ROOT / "progress.json", {"completed": 65, "total": 65, "complete": True,
                                  "source_files_unchanged": verify_sources(protocol)})
    print("COMPLETE: 65 matched GraphOnly fits; existing results unchanged.", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    main(args.limit, args.check_only)
