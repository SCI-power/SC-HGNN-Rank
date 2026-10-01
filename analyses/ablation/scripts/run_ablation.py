"""Matched module ablation for the original SC-HGNN-Rank architecture."""
from __future__ import annotations

import argparse
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

BASE = Path(__file__).resolve().parents[3]
ROOT = Path(os.environ.get("SC_HGNN_ANALYSIS_OUTPUT", BASE / "outputs/ablation")).resolve()
MODEL_RESULTS = Path(os.environ.get("SC_HGNN_RESULTS_DIR", BASE / "reference_results/model")).resolve()
sys.path.insert(0, str(BASE / "src"))
import reviewer_matched_rerun as core
from run_original_supplement import check_splits, architecture_signature

SEEDS = [20260807, 20260808, 20260809, 20260810, 20260811]
CFG = core.TrainConfig()
VARIANT = "SC-HGNN-Rank-full"
MODULES = {
    "without_spatial": {
        "label": "Spatial",
        "features": ["integrated_spatial_recurrence_score", "rctd_spotlight_consistency_score"],
        "edges": ["axis_spatially_recurrent_in_sample", "cell_state_maps_to_spatial_niche",
                  "sample_contains_mapped_spatial_niche", "target_gene_projects_to_spatial_niche"],
    },
    "without_scrna_grn": {
        "label": "scRNA-GRN",
        "features": ["single_cell_regulatory_score"],
        "edges": ["axis_has_tf_regulon", "target_gene_linked_to_cell_state",
                  "tf_regulates_regulon_target_gene", "tf_regulon_active_in_cell_state"],
    },
    "without_ppi_msi": {
        "label": "PPI / MSI",
        "features": ["ppi_msi_local_support_score"],
        "edges": ["axis_supported_by_ppi_msi_module", "target_gene_supported_by_ppi_msi_module"],
    },
    "without_drug_evidence": {
        "label": "Drug / perturbation",
        "features": ["drug_evidence_score", "lincs_cmap_combined_support_score",
                     "depmap_axis_support_score", "non_broad_cancer_sensitivity_score"],
        "edges": ["axis_has_compound", "axis_has_target_gene", "axis_has_non_broad_candidate_gene",
                  "axis_supported_by_depmap_dependency", "axis_supported_by_lincs_reversal",
                  "compound_acts_on_target_gene", "compound_reverses_lincs_signature",
                  "gene_essential_in_depmap_context"],
    },
}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for data in iter(lambda: handle.read(1048576), b""):
            h.update(data)
    return h.hexdigest()


def dump(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, default=str, allow_nan=False), encoding="utf-8")


def load_inputs():
    torch.set_num_threads(2)
    inputs = BASE / "data/model"
    raw = torch.load(inputs / "spatial_causal_hgnn_tensor_dataset_2026-05-04.pt",
                     map_location="cpu", weights_only=False)
    labels = pd.read_csv(inputs / "Table_HGNN3_axis_label_table.csv").reset_index(drop=True)
    decoys = pd.read_csv(inputs / "Table_HGNN7_negative_axis_decoy_pool.csv")
    splits = pd.read_csv(MODEL_RESULTS / "frozen_splits.csv")
    assert len(labels) == 600 and len(splits) == 13 and len(decoys) == 873
    assert np.allclose(raw["axis_soft_score"].numpy(), labels.soft_rank_label)
    assert torch.equal(raw["axis_label"], torch.tensor(labels.hgnn_label_ordinal.values))
    check_splits(labels, splits)
    return raw, labels, decoys, splits


def ablated_features(labels, module):
    changed = labels.copy()
    changed.loc[:, MODULES[module]["features"]] = 0.0
    # Recompute every interaction, including the nonzero-evidence fraction.
    return core.axis_features(changed)


def ablated_graph(raw, module):
    names = raw["maps"]["edge_type_names"]
    drop = MODULES[module]["edges"]
    assert set(drop).issubset(names)
    idx = torch.tensor([names.index(name) for name in drop], dtype=torch.long)
    keep = ~torch.isin(raw["edge_type_idx"], idx)
    result = dict(raw)
    for key in ["edge_src", "edge_dst", "edge_type_idx", "edge_weight", "edge_evidence_score"]:
        result[key] = raw[key][keep].clone()
    result["node_feature_score"] = raw["node_feature_score"].clone()
    # Non-axis nonzero node scores originate from LINCS/non-broad dependency.
    # They must not survive removal of the drug/perturbation evidence module.
    zero_nodes = []
    if module == "without_drug_evidence":
        zero_nodes = [i for name, i in raw["maps"]["node_id_to_index"].items()
                      if name.startswith(("gene::", "lincs_signature::"))]
        result["node_feature_score"][zero_nodes] = 0.0
    result["metadata"] = dict(raw["metadata"], n_edges=int(keep.sum()))
    assert not torch.isin(result["edge_type_idx"], idx).any()
    return result, {"module": module, "n_edges_removed": int((~keep).sum()),
                    "n_edges_remaining": int(keep.sum()), "n_node_scores_cleared": len(zero_nodes)}


def baseline_dir(split, seed):
    return MODEL_RESULTS / "main/fits" / f"baseline__{VARIANT}__{split.fold_id}__{seed}"


def prepare(raw, labels, splits):
    ROOT.mkdir(parents=True, exist_ok=True)
    (ROOT / "tables").mkdir(exist_ok=True)
    files = list((BASE / "data/model").glob("*")) + list((BASE / "src").glob("*.py"))
    files += [MODEL_RESULTS / "frozen_splits.csv", MODEL_RESULTS / "protocol.json"]
    full_rows = []
    for seed in SEEDS:
        for _, split in splits.iterrows():
            fit = baseline_dir(split, seed)
            files.extend([fit / "metrics.json", fit / "predictions.csv.gz",
                          fit / f"{VARIANT}_{split.fold_id}_seed{seed}.pt"])
            metric = json.loads((fit / "metrics.json").read_text())
            assert metric["variant"] == VARIANT and metric["objective"] == "rank_composite"
            assert metric["parameter_count"] == 182719
            pred = pd.read_csv(fit / "predictions.csv.gz")
            ids = labels.iloc[core.parse_indices(split.test_indices).numpy()].frozen_axis_id
            assert set(pred.frozen_axis_id) == set(ids)
            full_rows.append(metric)
    signature = architecture_signature(BASE / "src/train_spatial_causal_hgnn_rank_optimization.py")
    assert signature == architecture_signature(BASE / "data/reference_runs/source_original/train_spatial_causal_hgnn_rank_optimization.py")
    protocol = {
        "architecture": "original_SC-HGNN-Rank", "architecture_ast_sha256": signature,
        "config": asdict(CFG), "seeds": SEEDS, "modules": MODULES,
        "new_fits": 260, "reused_full_model_fits": 65, "parameter_count": 182719,
        "graph_protocol": "entity_edge_masked_training; same_module_removed_in_final_inference",
        "supervision": "full_original_labels_and_train_only_weights_fixed_across_variants",
        "interactions": "recomputed_after_zeroing_removed_base_features",
        "drug_node_scores": "gene_non_broad_scores_and_lincs_signature_scores_zeroed",
        "test_based_tuning": False,
        "primary_metric": "spearman",
        "statistics": "seed_average_within_13_folds; paired_two_sided_Wilcoxon; Holm_4_modules; fold_bootstrap_5000",
        "runtime": {"numpy": np.__version__, "pandas": pd.__version__, "torch": torch.__version__,
                    "cuda": torch.version.cuda},
        "input_sha256": {str(p.resolve()): sha(p) for p in sorted(set(files)) if p.is_file()},
    }
    lock = ROOT / "protocol.json"
    if lock.exists():
        assert json.loads(lock.read_text()) == protocol, "Locked protocol or source file changed."
    else:
        dump(lock, protocol)
    pd.DataFrame(full_rows).to_csv(ROOT / "tables/full_model_reference_runs.csv", index=False)
    check_splits(labels, splits).to_csv(ROOT / "tables/fixed_split_checks.csv", index=False)
    tests = []
    for module in MODULES:
        r, row = ablated_graph(raw, module)
        x, meta, cols = ablated_features(labels, module)
        for col in MODULES[module]["features"]:
            assert torch.all(x[:, cols.index(col)] == 0)
        assert torch.isfinite(x).all() and torch.isfinite(meta).all()
        changed = labels.copy()
        changed.loc[:, MODULES[module]["features"]] = .897123
        x2, meta2, _ = ablated_features(changed, module)
        assert torch.equal(x, x2) and torch.equal(meta, meta2)
        assert torch.equal(r["axis_soft_score"], raw["axis_soft_score"])
        model = core.make_model(VARIANT, r, x.shape[1], meta.shape[1], CFG)
        assert sum(p.numel() for p in model.parameters()) == 182719
        row.update({"zeroed_base_features": ";".join(MODULES[module]["features"]),
                    "removed_edge_types": ";".join(MODULES[module]["edges"]),
                    "masked_feature_invariance": True, "labels_unchanged": True})
        tests.append(row)
    pd.DataFrame(tests).to_csv(ROOT / "tables/module_definitions.csv", index=False)
    return protocol


def verify_sources(protocol):
    changed = [p for p, digest in protocol["input_sha256"].items() if sha(p) != digest]
    assert not changed, changed
    return len(protocol["input_sha256"])


def run(limit=None):
    raw, labels, decoys, splits = load_inputs()
    protocol = prepare(raw, labels, splits)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(json.dumps({"device": str(device), "planned_new_fits": 260,
                      "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU"}), flush=True)
    done = 0
    new = 0
    start = time.perf_counter()
    for module in MODULES:
        r, _ = ablated_graph(raw, module)
        x, meta, _ = ablated_features(labels, module)
        for seed in SEEDS:
            for _, split in splits.iterrows():
                fit = ROOT / "fits" / module / f"{split.fold_id}__{seed}"
                if (fit / "metrics.json").exists():
                    done += 1
                    continue
                fit.mkdir(parents=True, exist_ok=True)
                core.MODEL_DIR = fit
                metric, history, pred, hard = core.train_one(
                    r, labels, decoys, x, meta, split, VARIANT, seed, device, CFG,
                    save_model=True, inductive_graph=True, objective="rank_composite")
                assert metric["parameter_count"] == 182719
                ref = json.loads((baseline_dir(split, seed) / "metrics.json").read_text())
                for field in ["n_train", "n_val", "n_test", "training_top_boost_threshold",
                              "n_pairwise_pairs", "n_hard_pairs", "n_context_pairs"]:
                    assert metric[field] == ref[field], (module, field)
                refpred = pd.read_csv(baseline_dir(split, seed) / "predictions.csv.gz")
                assert list(pred.frozen_axis_id) == list(refpred.frozen_axis_id)
                metric["module"] = module
                history.to_csv(fit / "history.csv.gz", index=False)
                pred.to_csv(fit / "predictions.csv.gz", index=False)
                hard.to_csv(fit / "hard_pairs.csv.gz", index=False)
                dump(fit / "metrics.json", metric)
                done += 1
                new += 1
                print(json.dumps({"completed": done, "total": 260, "module": module,
                                  "fold": split.fold_id, "seed": seed,
                                  "spearman": metric["spearman"], "epochs": metric["epochs"],
                                  "seconds": round(time.perf_counter()-start, 1)}), flush=True)
                if limit and new >= limit:
                    dump(ROOT / "progress.json", {"completed": done, "total": 260,
                         "source_files_unchanged": verify_sources(protocol)})
                    return
    rows = [json.loads(p.read_text()) for p in sorted((ROOT / "fits").glob("*/*/metrics.json"))]
    assert len(rows) == 260
    pd.DataFrame(rows).to_csv(ROOT / "tables/ablation_run_metrics.csv", index=False)
    dump(ROOT / "progress.json", {"completed": 260, "total": 260,
                                 "source_files_unchanged": verify_sources(protocol), "complete": True})
    print("COMPLETE: 260 new ablation fits; original inputs and baseline files unchanged.", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    if args.check_only:
        raw, labels, decoys, splits = load_inputs()
        protocol = prepare(raw, labels, splits)
        print(json.dumps({"checks": "passed", "source_files": verify_sources(protocol)}))
    else:
        run(args.limit)
