from __future__ import annotations

from package_paths import DATA, INPUTS, ARCHIVE, OUTPUT, CODE

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch


PROJECT_ROOT = DATA / "project"
RUN_DATE = "2026-05-04"

GRAPH_TABLE_DIR = OUTPUT / "preprocess/graph"
TENSOR_DIR = OUTPUT / "preprocess/tensors"
TABLE_DIR = OUTPUT / "preprocess/tensor_maps"
METADATA_DIR = OUTPUT / "preprocess/metadata"

NODE_TABLE = GRAPH_TABLE_DIR / "Table_HGNN1_node_table.csv"
EDGE_TABLE = GRAPH_TABLE_DIR / "Table_HGNN2_edge_table.csv"
LABEL_TABLE = GRAPH_TABLE_DIR / "Table_HGNN3_axis_label_table.csv"
DECOY_TABLE = GRAPH_TABLE_DIR / "Table_HGNN7_negative_axis_decoy_pool.csv"


def ensure_dirs() -> None:
    TENSOR_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    METADATA_DIR.mkdir(parents=True, exist_ok=True)


def text_value(value: Any) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def encode(values: pd.Series, extra_first: str | None = None) -> tuple[dict[str, int], list[str]]:
    unique = sorted({text_value(v) for v in values if text_value(v)})
    if extra_first is not None and extra_first not in unique:
        unique = [extra_first] + unique
    elif extra_first is not None:
        unique = [extra_first] + [v for v in unique if v != extra_first]
    mapping = {v: i for i, v in enumerate(unique)}
    return mapping, unique


def stratified_split(labels: pd.DataFrame, seed: int = 20260504) -> dict[str, list[int]]:
    rng = np.random.default_rng(seed)
    splits = {"train": [], "val": [], "test": []}

    for (_, _), idx in labels.groupby(["cancer", "hgnn_label_ordinal"]).groups.items():
        arr = np.array(list(idx), dtype=int)
        rng.shuffle(arr)
        n = len(arr)
        n_train = max(1, int(round(n * 0.70)))
        n_val = max(1, int(round(n * 0.15)))
        if n_train + n_val >= n:
            n_train = max(1, n - 2)
            n_val = 1
        train = arr[:n_train]
        val = arr[n_train : n_train + n_val]
        test = arr[n_train + n_val :]
        splits["train"].extend(train.tolist())
        splits["val"].extend(val.tolist())
        splits["test"].extend(test.tolist())

    for key in splits:
        splits[key] = sorted(splits[key])
    return splits


def leave_one_cancer_splits(labels: pd.DataFrame) -> dict[str, dict[str, list[int]]]:
    out: dict[str, dict[str, list[int]]] = {}
    for cancer in sorted(labels["cancer"].dropna().unique()):
        test = labels.index[labels["cancer"].eq(cancer)].tolist()
        train = labels.index[~labels["cancer"].eq(cancer)].tolist()
        out[str(cancer)] = {"train": train, "test": test}
    return out


def write_table(df: pd.DataFrame, name: str) -> Path:
    path = TABLE_DIR / f"{name}.csv"
    df.to_csv(path, index=False, encoding="utf-8-sig")
    try:
        df.to_excel(TABLE_DIR / f"{name}.xlsx", index=False)
    except Exception:
        pass
    return path


def main() -> None:
    ensure_dirs()
    nodes = pd.read_csv(NODE_TABLE)
    edges = pd.read_csv(EDGE_TABLE)
    labels = pd.read_csv(LABEL_TABLE)
    decoys = pd.read_csv(DECOY_TABLE) if DECOY_TABLE.exists() else pd.DataFrame()

    nodes = nodes.reset_index(drop=True)
    nodes["node_index"] = np.arange(len(nodes), dtype=int)
    node_id_to_index = dict(zip(nodes["node_id"], nodes["node_index"]))

    node_type_map, node_type_names = encode(nodes["node_type"])
    cancer_map, cancer_names = encode(pd.concat([nodes["cancer"], labels["cancer"]], ignore_index=True), extra_first="__none__")
    edge_type_map, edge_type_names = encode(edges["edge_type"])

    nodes["node_type_idx"] = nodes["node_type"].map(node_type_map).astype(int)
    nodes["node_cancer_idx"] = nodes["cancer"].fillna("").map(lambda x: cancer_map.get(text_value(x), cancer_map["__none__"])).astype(int)
    nodes["node_feature_score"] = pd.to_numeric(nodes["feature_score"], errors="coerce").fillna(0.0).clip(0, 1)

    edges = edges.loc[
        edges["source_node_id"].isin(node_id_to_index) & edges["target_node_id"].isin(node_id_to_index)
    ].copy()
    edges["source_index"] = edges["source_node_id"].map(node_id_to_index).astype(int)
    edges["target_index"] = edges["target_node_id"].map(node_id_to_index).astype(int)
    edges["edge_type_idx"] = edges["edge_type"].map(edge_type_map).astype(int)
    edges["weight"] = pd.to_numeric(edges["weight"], errors="coerce").fillna(0.0).clip(0, 1)
    edges["evidence_score"] = pd.to_numeric(edges["evidence_score"], errors="coerce").fillna(edges["weight"]).clip(0, 1)

    labels = labels.reset_index(drop=True)
    labels["axis_node_id"] = "axis::" + labels["frozen_axis_id"].astype(str)
    labels = labels.loc[labels["axis_node_id"].isin(node_id_to_index)].copy().reset_index(drop=True)
    labels["axis_node_index"] = labels["axis_node_id"].map(node_id_to_index).astype(int)
    labels["axis_cancer_idx"] = labels["cancer"].map(cancer_map).astype(int)
    labels["hgnn_label_ordinal"] = pd.to_numeric(labels["hgnn_label_ordinal"], errors="coerce").fillna(0).astype(int)
    labels["soft_rank_label"] = pd.to_numeric(labels["soft_rank_label"], errors="coerce").fillna(0.0).clip(0, 1)

    splits = stratified_split(labels)
    loco_splits = leave_one_cancer_splits(labels)

    tensor_data = {
        "node_feature_score": torch.tensor(nodes["node_feature_score"].values, dtype=torch.float32).view(-1, 1),
        "node_type_idx": torch.tensor(nodes["node_type_idx"].values, dtype=torch.long),
        "node_cancer_idx": torch.tensor(nodes["node_cancer_idx"].values, dtype=torch.long),
        "edge_src": torch.tensor(edges["source_index"].values, dtype=torch.long),
        "edge_dst": torch.tensor(edges["target_index"].values, dtype=torch.long),
        "edge_type_idx": torch.tensor(edges["edge_type_idx"].values, dtype=torch.long),
        "edge_weight": torch.tensor(edges["weight"].values, dtype=torch.float32),
        "edge_evidence_score": torch.tensor(edges["evidence_score"].values, dtype=torch.float32),
        "axis_node_idx": torch.tensor(labels["axis_node_index"].values, dtype=torch.long),
        "axis_label": torch.tensor(labels["hgnn_label_ordinal"].values, dtype=torch.long),
        "axis_soft_score": torch.tensor(labels["soft_rank_label"].values, dtype=torch.float32),
        "axis_cancer_idx": torch.tensor(labels["axis_cancer_idx"].values, dtype=torch.long),
        "splits": {k: torch.tensor(v, dtype=torch.long) for k, v in splits.items()},
        "leave_one_cancer_splits": {
            cancer: {k: torch.tensor(v, dtype=torch.long) for k, v in split.items()}
            for cancer, split in loco_splits.items()
        },
        "metadata": {
            "run_date": RUN_DATE,
            "n_nodes": int(len(nodes)),
            "n_edges": int(len(edges)),
            "n_axis_labels": int(len(labels)),
            "n_decoys": int(len(decoys)),
            "num_node_types": int(len(node_type_names)),
            "num_edge_types": int(len(edge_type_names)),
            "num_cancers": int(len(cancer_names)),
        },
        "maps": {
            "node_type_names": node_type_names,
            "edge_type_names": edge_type_names,
            "cancer_names": cancer_names,
            "node_id_to_index": node_id_to_index,
        },
    }

    dataset_path = TENSOR_DIR / f"spatial_causal_hgnn_tensor_dataset_{RUN_DATE}.pt"
    torch.save(tensor_data, dataset_path)

    write_table(
        nodes[["node_index", "node_id", "node_type", "node_type_idx", "cancer", "node_cancer_idx", "label", "node_feature_score"]],
        "Table_HGNN_TENSOR1_node_index_map",
    )
    write_table(
        edges[["edge_id", "source_index", "target_index", "edge_type", "edge_type_idx", "weight", "evidence_score", "source_layer", "feature_name", "frozen_axis_id"]],
        "Table_HGNN_TENSOR2_edge_index_map",
    )

    split_rows = []
    for split_name, idxs in splits.items():
        sub = labels.iloc[idxs]
        for (cancer, label), df in sub.groupby(["cancer", "hgnn_integrated_label"]):
            split_rows.append({"split": split_name, "cancer": cancer, "label": label, "n": int(len(df))})
    split_table = pd.DataFrame(split_rows).sort_values(["split", "cancer", "label"])
    write_table(split_table, "Table_HGNN_TENSOR3_train_val_test_split")

    map_table = pd.DataFrame(
        [{"map_name": "node_type", "name": name, "index": idx} for idx, name in enumerate(node_type_names)]
        + [{"map_name": "edge_type", "name": name, "index": idx} for idx, name in enumerate(edge_type_names)]
        + [{"map_name": "cancer", "name": name, "index": idx} for idx, name in enumerate(cancer_names)]
    )
    write_table(map_table, "Table_HGNN_TENSOR4_category_maps")

    manifest = {
        "run_date": RUN_DATE,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "dataset_path": str(dataset_path),
        "n_nodes": int(len(nodes)),
        "n_edges": int(len(edges)),
        "n_axis_labels": int(len(labels)),
        "n_negative_decoys": int(len(decoys)),
        "num_node_types": int(len(node_type_names)),
        "num_edge_types": int(len(edge_type_names)),
        "num_cancers": int(len(cancer_names)),
        "tables": str(TABLE_DIR),
    }
    manifest_path = METADATA_DIR / f"run_manifest_spatial_causal_hgnn_tensor_dataset_{RUN_DATE}.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
