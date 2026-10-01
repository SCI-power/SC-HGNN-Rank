"""Held-out score sensitivity from the unchanged 65 full-model checkpoints."""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
import torch

from run_ablation import (ROOT, BASE, SEEDS, CFG, VARIANT, MODULES, core,
                          load_inputs, ablated_features, ablated_graph, baseline_dir, dump)


def main(device_name):
    raw, labels, _, splits = load_inputs()
    device = torch.device(device_name)
    x, meta, _ = core.axis_features(labels)
    x, meta = x.to(device), meta.to(device)
    data = core.move_to_device(core.add_structural_features(raw), device)
    feature_views = {}
    edge_views = {}
    for module in MODULES:
        xx, mm, _ = ablated_features(labels, module)
        feature_views[module] = (xx.to(device), mm.to(device))
        rr, _ = ablated_graph(raw, module)
        edge_views[module] = core.move_to_device(core.add_structural_features(rr), device)
    rows = []
    attention_rows = []
    validations = []
    for seed in SEEDS:
        for _, split in splits.iterrows():
            fit = baseline_dir(split, seed)
            path = fit / f"{VARIANT}_{split.fold_id}_seed{seed}.pt"
            checkpoint = torch.load(path, map_location=device, weights_only=False)
            assert checkpoint["objective"] == "rank_composite"
            model = core.make_model(VARIANT, raw, x.shape[1], meta.shape[1], CFG).to(device)
            model.load_state_dict(checkpoint["state_dict"])
            model.eval()
            test = core.parse_indices(split.test_indices).to(device)
            indices = test.cpu().numpy()
            identity = labels.iloc[indices][["frozen_axis_id", "cancer"]].reset_index(drop=True)
            with torch.no_grad():
                _, full_score, attention = model(data, x, meta, return_attention=True)
                ref = pd.read_csv(fit / "predictions.csv.gz").set_index("frozen_axis_id")
                reference = ref.loc[identity.frozen_axis_id, "predicted_score"].values
                observed = full_score[test].cpu().numpy()
                error = float(np.max(np.abs(observed-reference)))
                assert error < 2e-5, (split.fold_id, seed, error)
                validations.append({"fold_id": split.fold_id, "seed": seed,
                                    "max_checkpoint_reproduction_error": error})
                att = identity.copy()
                att["seed"], att["validation_type"], att["fold_id"] = seed, split.validation_type, split.fold_id
                for j, key in enumerate(["graph_branch", "axis_feature_branch", "metapath_branch"]):
                    att[key] = attention[test, j].cpu().numpy()
                attention_rows.append(att)
                for module in MODULES:
                    for mode in ["input_features", "graph_evidence"]:
                        if mode == "input_features":
                            xx, mm = feature_views[module]
                            _, score = model(data, xx, mm)
                        else:
                            _, score = model(edge_views[module], x, meta)
                        frame = identity.copy()
                        frame["seed"], frame["fold_id"] = seed, split.fold_id
                        frame["validation_type"] = split.validation_type
                        frame["module"], frame["mode"] = module, mode
                        frame["full_score"] = observed
                        frame["masked_score"] = score[test].cpu().numpy()
                        frame["signed_score_change"] = frame.full_score - frame.masked_score
                        frame["absolute_score_change"] = frame.signed_score_change.abs()
                        rows.append(frame)
            print(json.dumps({"explanation_checkpoints": len(validations), "total": 65,
                              "reproduction_error": error}), flush=True)
    result = pd.concat(rows, ignore_index=True)
    assert len(result) == 600 * 15 * 8
    axis = result.groupby(["frozen_axis_id", "cancer", "module", "mode"], as_index=False).agg(
        mean_absolute_change=("absolute_score_change", "mean"),
        mean_signed_change=("signed_score_change", "mean"),
        n_heldout_predictions=("absolute_score_change", "size"))
    assert axis.n_heldout_predictions.eq(15).all()
    summary = axis.groupby(["cancer", "module", "mode"], as_index=False).agg(
        mean_absolute_change=("mean_absolute_change", "mean"),
        median_absolute_change=("mean_absolute_change", "median"),
        mean_signed_change=("mean_signed_change", "mean"),
        n_axes=("frozen_axis_id", "nunique"))
    assert summary.n_axes.eq(200).all()
    table = ROOT / "tables"
    table.mkdir(parents=True, exist_ok=True)
    result.to_csv(table / "Figure4G_heldout_score_changes.csv.gz", index=False)
    axis.to_csv(table / "Figure4G_axis_sensitivity.csv", index=False)
    summary.to_csv(table / "Figure4G_module_sensitivity_summary.csv", index=False)
    pd.concat(attention_rows, ignore_index=True).to_csv(table / "Figure4G_branch_attention.csv.gz", index=False)
    pd.DataFrame(validations).to_csv(table / "Figure4G_checkpoint_checks.csv", index=False)
    dump(ROOT / "explanation_protocol.json", {
        "full_model_checkpoints": 65, "unique_axes": 600, "heldout_predictions_per_axis": 15,
        "scope": "all_200_candidates_per_cancer; no_rank_based_selection",
        "input_features": "zero_module_base_features_and_recompute_interactions;_graph_fixed",
        "graph_evidence": "remove_module_edges_and_source_node_scores;_recompute_degrees;_axis_and_metapath_features_fixed",
        "aggregation": "mean_absolute_score_change_across_15_OOF_evaluations_per_axis_then_200_axes_per_cancer",
        "units": "absolute_model_score_change_on_0_to_1_score_scale",
        "interpretation": "one_group_at_a_time_model_sensitivity;_not_additive_contribution_or_causal_effect",
        "training": "none; original_model_weights_unchanged",
        "max_checkpoint_reproduction_error": max(v["max_checkpoint_reproduction_error"] for v in validations),
        "device": str(device)})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    main(parser.parse_args().device)
