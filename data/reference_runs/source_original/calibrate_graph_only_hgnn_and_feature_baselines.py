from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from scipy.stats import wilcoxon
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from torch import nn
from torch.nn import functional as F

from train_spatial_causal_hgnn_generalization import build_split_configs, parse_indices
from train_spatial_causal_hgnn_model_comparison import (
    AXIS_FEATURE_COLS,
    HGTLikeLayer,
    add_structural_features,
    move_to_device,
    pearson_corr,
    spearman_corr,
)
from train_spatial_causal_hgnn_rank_optimization import INTERACTION_FEATURES, add_axis_interactions
from train_spatial_causal_hgnn_statistical_validation import bootstrap_ci


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUN_DATE = "2026-05-04"
RUN_ID = "spatial_causal_hgnn_graph_only_calibration_2026-05-04"

MODEL_DIR = PROJECT_ROOT / "models" / "spatial_causal_hgnn"
TABLE_DIR = PROJECT_ROOT / "tables" / "spatial_causal_hgnn_graph_only_calibration"
FIG_DIR = PROJECT_ROOT / "figures" / "spatial_causal_hgnn_graph_only_calibration"
METADATA_DIR = PROJECT_ROOT / "metadata"
ARTIFACT_DIR = PROJECT_ROOT / "artifacts" / "experiment" / RUN_ID

DATASET_PATH = MODEL_DIR / f"spatial_causal_hgnn_tensor_dataset_{RUN_DATE}.pt"
LABEL_TABLE = PROJECT_ROOT / "tables" / "spatial_causal_hgnn_graph_inputs" / "Table_HGNN3_axis_label_table.csv"
NODE_TABLE = PROJECT_ROOT / "tables" / "spatial_causal_hgnn_graph_inputs" / "Table_HGNN1_node_table.csv"
EDGE_TABLE = PROJECT_ROOT / "tables" / "spatial_causal_hgnn_graph_inputs" / "Table_HGNN2_edge_table.csv"
EXISTING_METRICS = PROJECT_ROOT / "tables" / "spatial_causal_hgnn_statistical_validation" / "Table_HGNN_STAT1_multiseed_metrics_by_model_split_seed.csv"

SEEDS = [20260504, 20260505, 20260506, 20260507, 20260508]
VALIDATION_ORDER = ["leave_one_cancer", "leave_compound_group", "leave_target_group"]
VALIDATION_LABELS = {
    "leave_one_cancer": "Leave-one-cancer",
    "leave_compound_group": "Leave-compound",
    "leave_target_group": "Leave-target",
}
GRAPH_MODEL = "Spatial-Causal-HGNN-GraphOnly"
CURRENT_HGNN = "Spatial-Causal-HGNN"
BASELINE_MODELS = [
    "Feature-MinimalStructure",
    "Feature-EdgeCount",
    "Feature-FullEvidence",
    CURRENT_HGNN,
    GRAPH_MODEL,
]


def ensure_dirs() -> None:
    for path in [TABLE_DIR, FIG_DIR, METADATA_DIR, ARTIFACT_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def numeric(series: pd.Series, default: float = 0.0) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").fillna(default)


def write_table(df: pd.DataFrame, name: str) -> Path:
    path = TABLE_DIR / f"{name}.csv"
    df.to_csv(path, index=False, encoding="utf-8-sig")
    try:
        df.to_excel(TABLE_DIR / f"{name}.xlsx", index=False)
    except Exception:
        pass
    return path


class GraphOnlySpatialCausalHGNN(nn.Module):
    """Relation-aware graph model with no direct axis evidence-vector input."""

    def __init__(self, num_node_types: int, num_edge_types: int, hidden_dim: int = 112, num_layers: int = 3, dropout: float = 0.22) -> None:
        super().__init__()
        self.node_type_embedding = nn.Embedding(num_node_types, hidden_dim)
        self.feature_projection = nn.Linear(4, hidden_dim)
        self.layers = nn.ModuleList([HGTLikeLayer(hidden_dim, num_edge_types, dropout) for _ in range(num_layers)])
        self.skip = nn.Parameter(torch.tensor(0.55))
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 3),
        )
        self.regressor = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, data: dict[str, torch.Tensor], edge_keep_mask: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        h0 = self.node_type_embedding(data["node_type_idx"]) + self.feature_projection(data["node_struct_features"])
        h = h0
        if edge_keep_mask is None:
            src, dst, et, ew = data["edge_src"], data["edge_dst"], data["edge_type_idx"], data["edge_weight"]
        else:
            src = data["edge_src"][edge_keep_mask]
            dst = data["edge_dst"][edge_keep_mask]
            et = data["edge_type_idx"][edge_keep_mask]
            ew = data["edge_weight"][edge_keep_mask]
        for layer in self.layers:
            h = layer(h, src, dst, et, ew)
        h = self.skip.sigmoid() * h + (1.0 - self.skip.sigmoid()) * h0
        axis_h = h[data["axis_node_idx"]]
        return self.classifier(axis_h), torch.sigmoid(self.regressor(axis_h)).squeeze(-1)


def evaluate_arrays(y_label: np.ndarray, y_label_pred: np.ndarray, y_score: np.ndarray, y_score_pred: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_label, y_label_pred)),
        "macro_f1": float(f1_score(y_label, y_label_pred, average="macro")),
        "score_mae": float(mean_absolute_error(y_score, y_score_pred)),
        "score_mse": float(mean_squared_error(y_score, y_score_pred)),
        "score_r2": float(r2_score(y_score, y_score_pred)) if len(np.unique(y_score)) > 1 else float("nan"),
        "score_pearson": pearson_corr(y_score, y_score_pred),
        "score_spearman": spearman_corr(y_score, y_score_pred),
    }


def sample_weights(labels_df: pd.DataFrame) -> torch.Tensor:
    label = labels_df["hgnn_label_ordinal"].astype(int)
    score = numeric(labels_df["soft_rank_label"])
    base = label.map({0: 0.70, 1: 1.00, 2: 1.18}).fillna(0.85).astype(float)
    boost = np.where(score >= score.quantile(0.80), 1.10, 1.0)
    return torch.tensor((base.values * boost).astype(float), dtype=torch.float32)


def weighted_mean(values: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    return (values * weights).sum() / weights.sum().clamp_min(1e-6)


def prediction_frame(
    labels_df: pd.DataFrame,
    split_row: pd.Series,
    split_name: str,
    idx_np: np.ndarray,
    model_name: str,
    seed: int,
    pred_label: np.ndarray,
    pred_score: np.ndarray,
    prob: np.ndarray | None = None,
) -> pd.DataFrame:
    out = labels_df.iloc[idx_np].copy()
    out["model"] = model_name
    out["seed"] = seed
    out["validation_type"] = split_row["validation_type"]
    out["fold_id"] = split_row["fold_id"]
    out["split"] = split_name
    out["predicted_label"] = pred_label
    out["predicted_soft_score"] = pred_score
    out["absolute_score_error"] = (numeric(out["soft_rank_label"]) - out["predicted_soft_score"]).abs()
    if prob is not None:
        out["prob_unlabeled_or_negative"] = prob[:, 0]
        out["prob_moderate_positive"] = prob[:, 1]
        out["prob_high_positive"] = prob[:, 2]
    return out


def train_graph_only(
    raw: dict[str, Any],
    labels_df: pd.DataFrame,
    split_row: pd.Series,
    device: torch.device,
    seed: int,
    max_epochs: int = 150,
    patience: int = 26,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    data = move_to_device(add_structural_features(raw), device)
    labels = data["axis_label"]
    score = data["axis_soft_score"]
    weights = sample_weights(labels_df).to(device)
    train_idx = parse_indices(split_row["train_indices"]).to(device)
    val_idx = parse_indices(split_row["val_indices"]).to(device)
    test_idx = parse_indices(split_row["test_indices"]).to(device)
    metadata = raw["metadata"]
    model = GraphOnlySpatialCausalHGNN(
        num_node_types=int(metadata["num_node_types"]),
        num_edge_types=int(metadata["num_edge_types"]),
        hidden_dim=112,
        num_layers=3,
        dropout=0.22,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.8e-3, weight_decay=1.8e-4)

    best_state: dict[str, torch.Tensor] | None = None
    best_val_loss = float("inf")
    stale = 0
    history_rows = []
    for epoch in range(1, max_epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        if epoch <= int(max_epochs * 0.70):
            keep = torch.rand(data["edge_weight"].shape[0], device=device) > 0.08
        else:
            keep = None
        logits, pred_score = model(data, edge_keep_mask=keep)
        loss_cls = weighted_mean(F.cross_entropy(logits[train_idx], labels[train_idx], reduction="none"), weights[train_idx])
        loss_reg = weighted_mean(F.smooth_l1_loss(pred_score[train_idx], score[train_idx], reduction="none"), weights[train_idx])
        loss = loss_cls + 0.38 * loss_reg
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        optimizer.step()

        model.eval()
        with torch.no_grad():
            logits_val, pred_val = model(data)
            val_loss = F.cross_entropy(logits_val[val_idx], labels[val_idx]) + 0.38 * F.smooth_l1_loss(pred_val[val_idx], score[val_idx])
            val_spear = spearman_corr(score[val_idx].detach().cpu().numpy(), pred_val[val_idx].detach().cpu().numpy())
        history_rows.append(
            {
                "model": GRAPH_MODEL,
                "seed": seed,
                "validation_type": split_row["validation_type"],
                "fold_id": split_row["fold_id"],
                "epoch": epoch,
                "train_loss": float(loss.detach().cpu()),
                "val_loss": float(val_loss.detach().cpu()),
                "val_score_spearman": val_spear,
            }
        )
        val_loss_value = float(val_loss.detach().cpu())
        if val_loss_value < best_val_loss:
            best_val_loss = val_loss_value
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        if stale >= patience:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        logits, pred_score = model(data)
        prob = torch.softmax(logits, dim=1).detach().cpu().numpy()
        pred_label = logits.argmax(dim=1).detach().cpu().numpy()
        pred_score_np = pred_score.detach().cpu().numpy()

    metrics = []
    preds = []
    for split_name, idx in [("train", train_idx), ("val", val_idx), ("test", test_idx)]:
        idx_np = idx.detach().cpu().numpy()
        y_label = labels[idx].detach().cpu().numpy()
        y_score = score[idx].detach().cpu().numpy()
        row = evaluate_arrays(y_label, pred_label[idx_np], y_score, pred_score_np[idx_np])
        row.update(
            {
                "model": GRAPH_MODEL,
                "seed": seed,
                "validation_type": split_row["validation_type"],
                "fold_id": split_row["fold_id"],
                "heldout_group": split_row["heldout_group"],
                "split": split_name,
                "n": int(len(idx_np)),
                "best_val_loss": best_val_loss,
                "epochs_completed": len(history_rows),
            }
        )
        metrics.append(row)
        preds.append(prediction_frame(labels_df, split_row, split_name, idx_np, GRAPH_MODEL, seed, pred_label[idx_np], pred_score_np[idx_np], prob[idx_np]))
    return pd.DataFrame(metrics), pd.DataFrame(history_rows), pd.concat(preds, ignore_index=True)


def build_feature_matrices(labels_df: pd.DataFrame, raw: dict[str, Any]) -> dict[str, pd.DataFrame]:
    labels = add_axis_interactions(labels_df)
    minimal = pd.get_dummies(labels[["cancer"]].astype(str), prefix="cancer").astype(float)

    edge_names = list(raw["maps"]["edge_type_names"])
    axis_node_idx = raw["axis_node_idx"].numpy()
    node_to_axis_pos = {int(node): pos for pos, node in enumerate(axis_node_idx)}
    counts = np.zeros((len(labels_df), len(edge_names)), dtype=float)
    src = raw["edge_src"].numpy()
    dst = raw["edge_dst"].numpy()
    et = raw["edge_type_idx"].numpy()
    for s, d, e in zip(src, dst, et):
        if int(s) in node_to_axis_pos:
            counts[node_to_axis_pos[int(s)], int(e)] += 1.0
        if int(d) in node_to_axis_pos:
            counts[node_to_axis_pos[int(d)], int(e)] += 1.0
    count_df = pd.DataFrame(counts, columns=[f"edge_count__{x}" for x in edge_names])
    count_df = pd.concat([minimal.reset_index(drop=True), count_df], axis=1)
    count_df["axis_degree_count"] = count_df[[c for c in count_df.columns if c.startswith("edge_count__")]].sum(axis=1)

    full = labels[AXIS_FEATURE_COLS + INTERACTION_FEATURES].apply(pd.to_numeric, errors="coerce").fillna(0.0).clip(0, 1)
    return {
        "Feature-MinimalStructure": minimal.reset_index(drop=True),
        "Feature-EdgeCount": count_df.reset_index(drop=True),
        "Feature-FullEvidence": full.reset_index(drop=True),
    }


def run_feature_baselines(labels_df: pd.DataFrame, feature_mats: dict[str, pd.DataFrame], split_row: pd.Series, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_idx = parse_indices(split_row["train_indices"]).numpy()
    val_idx = parse_indices(split_row["val_indices"]).numpy()
    test_idx = parse_indices(split_row["test_indices"]).numpy()
    y_score = numeric(labels_df["soft_rank_label"]).values.astype(float)
    y_label = labels_df["hgnn_label_ordinal"].astype(int).values
    metric_rows = []
    pred_rows = []
    for model_name, feat in feature_mats.items():
        X = feat.values.astype(float)
        if model_name == "Feature-FullEvidence":
            reg = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
            clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=seed))
        elif model_name == "Feature-EdgeCount":
            reg = GradientBoostingRegressor(n_estimators=160, learning_rate=0.04, max_depth=2, random_state=seed)
            clf = GradientBoostingClassifier(n_estimators=160, learning_rate=0.04, max_depth=2, random_state=seed)
        else:
            reg = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
            clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=seed))
        reg.fit(X[train_idx], y_score[train_idx])
        clf.fit(X[train_idx], y_label[train_idx])
        for split_name, idx in [("train", train_idx), ("val", val_idx), ("test", test_idx)]:
            pred_score = np.clip(reg.predict(X[idx]), 0, 1)
            pred_label = clf.predict(X[idx]).astype(int)
            prob = clf.predict_proba(X[idx])
            if prob.shape[1] != 3:
                full = np.zeros((len(idx), 3), dtype=float)
                for pos, cls in enumerate(clf.classes_):
                    full[:, int(cls)] = prob[:, pos]
                prob = full
            row = evaluate_arrays(y_label[idx], pred_label, y_score[idx], pred_score)
            row.update(
                {
                    "model": model_name,
                    "seed": seed,
                    "validation_type": split_row["validation_type"],
                    "fold_id": split_row["fold_id"],
                    "heldout_group": split_row["heldout_group"],
                    "split": split_name,
                    "n": int(len(idx)),
                }
            )
            metric_rows.append(row)
            pred_rows.append(prediction_frame(labels_df, split_row, split_name, idx, model_name, seed, pred_label, pred_score, prob))
    return pd.DataFrame(metric_rows), pd.concat(pred_rows, ignore_index=True)


def calibration_bins(y_true: np.ndarray, y_pred: np.ndarray, n_bins: int = 10) -> tuple[pd.DataFrame, dict[str, float]]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    bins = np.linspace(0, 1, n_bins + 1)
    ids = np.digitize(y_pred, bins[1:-1], right=False)
    rows = []
    ece = 0.0
    for b in range(n_bins):
        mask = ids == b
        if not np.any(mask):
            rows.append({"bin": b + 1, "bin_low": bins[b], "bin_high": bins[b + 1], "n": 0, "mean_pred": np.nan, "mean_true": np.nan, "abs_gap": np.nan})
            continue
        mean_pred = float(y_pred[mask].mean())
        mean_true = float(y_true[mask].mean())
        gap = abs(mean_pred - mean_true)
        ece += float(mask.mean()) * gap
        rows.append({"bin": b + 1, "bin_low": bins[b], "bin_high": bins[b + 1], "n": int(mask.sum()), "mean_pred": mean_pred, "mean_true": mean_true, "abs_gap": gap})
    metrics = {
        "ece_10bin": ece,
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "mse": float(mean_squared_error(y_true, y_pred)),
        "spearman": spearman_corr(y_true, y_pred),
        "pearson": pearson_corr(y_true, y_pred),
    }
    return pd.DataFrame(rows), metrics


def calibrate_graph_predictions(predictions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows = []
    bin_rows = []
    pred_rows = []
    graph = predictions.loc[predictions["model"].eq(GRAPH_MODEL)].copy()
    for (validation_type, fold_id, seed), df in graph.groupby(["validation_type", "fold_id", "seed"]):
        val = df.loc[df["split"].eq("val")]
        test = df.loc[df["split"].eq("test")]
        if val.empty or test.empty:
            continue
        x_val = numeric(val["predicted_soft_score"]).values
        y_val = numeric(val["soft_rank_label"]).values
        x_test = numeric(test["predicted_soft_score"]).values
        y_test = numeric(test["soft_rank_label"]).values
        methods = {"raw": x_test}
        iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        iso.fit(x_val, y_val)
        methods["isotonic"] = np.clip(iso.predict(x_test), 0, 1)
        linear = Ridge(alpha=0.10)
        linear.fit(x_val.reshape(-1, 1), y_val)
        methods["linear"] = np.clip(linear.predict(x_test.reshape(-1, 1)), 0, 1)
        for method, pred in methods.items():
            bins, metrics = calibration_bins(y_test, pred)
            row = {
                "validation_type": validation_type,
                "fold_id": fold_id,
                "seed": seed,
                "model": GRAPH_MODEL,
                "calibration_method": method,
                "n_test": int(len(test)),
            }
            row.update(metrics)
            rows.append(row)
            bins["validation_type"] = validation_type
            bins["fold_id"] = fold_id
            bins["seed"] = seed
            bins["model"] = GRAPH_MODEL
            bins["calibration_method"] = method
            bin_rows.append(bins)
            cp = test.copy()
            cp["calibration_method"] = method
            cp["calibrated_soft_score"] = pred
            cp["calibrated_abs_error"] = (numeric(cp["soft_rank_label"]) - cp["calibrated_soft_score"]).abs()
            pred_rows.append(cp)
    return pd.DataFrame(rows), pd.concat(bin_rows, ignore_index=True), pd.concat(pred_rows, ignore_index=True)


def summarize_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    test = metrics.loc[metrics["split"].eq("test")].copy()
    rows = []
    for (validation_type, model), df in test.groupby(["validation_type", "model"], sort=False):
        vals = numeric(df["score_spearman"], np.nan).dropna()
        rows.append(
            {
                "validation_type": validation_type,
                "model": model,
                "n_seed_fold_runs": int(len(vals)),
                "mean_score_spearman": float(vals.mean()),
                "sd_score_spearman": float(vals.std(ddof=1)),
                "mean_score_mae": float(numeric(df["score_mae"], np.nan).mean()),
                "sd_score_mae": float(numeric(df["score_mae"], np.nan).std(ddof=1)),
                "mean_macro_f1": float(numeric(df["macro_f1"], np.nan).mean()),
            }
        )
    return pd.DataFrame(rows)


def paired_deltas(metrics: pd.DataFrame) -> pd.DataFrame:
    test = metrics.loc[metrics["split"].eq("test")].copy()
    key = ["validation_type", "fold_id", "seed"]
    graph = test.loc[test["model"].eq(GRAPH_MODEL), key + ["score_spearman"]].rename(columns={"score_spearman": "graph"})
    rows = []
    for ref in ["Feature-MinimalStructure", "Feature-EdgeCount", "Feature-FullEvidence", CURRENT_HGNN]:
        base = test.loc[test["model"].eq(ref), key + ["score_spearman"]].rename(columns={"score_spearman": "reference"})
        for validation_type in VALIDATION_ORDER:
            merged = graph.loc[graph["validation_type"].eq(validation_type)].merge(base.loc[base["validation_type"].eq(validation_type)], on=key, how="inner")
            diffs = numeric(merged["graph"], np.nan) - numeric(merged["reference"], np.nan)
            diffs = diffs.dropna()
            if len(diffs) and np.any(np.abs(diffs.values) > 1e-12):
                try:
                    p = float(wilcoxon(diffs, zero_method="wilcox", alternative="greater").pvalue)
                except ValueError:
                    p = float("nan")
            else:
                p = float("nan")
            ci_low, ci_high = bootstrap_ci(diffs.values, seed=20261101 + len(rows))
            rows.append(
                {
                    "validation_type": validation_type,
                    "reference_model": ref,
                    "n_pairs": int(len(diffs)),
                    "mean_delta_spearman": float(diffs.mean()) if len(diffs) else float("nan"),
                    "bootstrap_ci95_low": ci_low,
                    "bootstrap_ci95_high": ci_high,
                    "wilcoxon_p_greater": p,
                }
            )
        merged_all = graph.merge(base, on=key, how="inner")
        diffs = numeric(merged_all["graph"], np.nan) - numeric(merged_all["reference"], np.nan)
        diffs = diffs.dropna()
        ci_low, ci_high = bootstrap_ci(diffs.values, seed=20261201 + len(rows))
        try:
            p = float(wilcoxon(diffs, zero_method="wilcox", alternative="greater").pvalue) if len(diffs) and np.any(np.abs(diffs.values) > 1e-12) else float("nan")
        except ValueError:
            p = float("nan")
        rows.append(
            {
                "validation_type": "all_strict_splits",
                "reference_model": ref,
                "n_pairs": int(len(diffs)),
                "mean_delta_spearman": float(diffs.mean()) if len(diffs) else float("nan"),
                "bootstrap_ci95_low": ci_low,
                "bootstrap_ci95_high": ci_high,
                "wilcoxon_p_greater": p,
            }
        )
    return pd.DataFrame(rows)


def summarize_calibration(calibration: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (validation_type, method), df in calibration.groupby(["validation_type", "calibration_method"]):
        rows.append(
            {
                "validation_type": validation_type,
                "calibration_method": method,
                "n_runs": int(len(df)),
                "mean_ece_10bin": float(numeric(df["ece_10bin"], np.nan).mean()),
                "sd_ece_10bin": float(numeric(df["ece_10bin"], np.nan).std(ddof=1)),
                "mean_mae": float(numeric(df["mae"], np.nan).mean()),
                "mean_mse": float(numeric(df["mse"], np.nan).mean()),
                "mean_spearman": float(numeric(df["spearman"], np.nan).mean()),
            }
        )
    return pd.DataFrame(rows)


def make_figures(summary: pd.DataFrame, delta: pd.DataFrame, cal_summary: pd.DataFrame, cal_bins: pd.DataFrame) -> list[dict[str, str]]:
    import matplotlib.pyplot as plt

    figures = []
    colors = {
        "Feature-MinimalStructure": "#8A9199",
        "Feature-EdgeCount": "#626A73",
        "Feature-FullEvidence": "#A9793A",
        CURRENT_HGNN: "#4F73C7",
        GRAPH_MODEL: "#B93A3A",
    }
    fig, ax = plt.subplots(figsize=(10.2, 4.8))
    x = np.arange(len(VALIDATION_ORDER))
    width = 0.15
    offsets = np.linspace(-2, 2, len(BASELINE_MODELS)) * width
    for offset, model in zip(offsets, BASELINE_MODELS):
        sub = summary.loc[summary["model"].eq(model)].set_index("validation_type").reindex(VALIDATION_ORDER)
        ax.bar(
            x + offset,
            numeric(sub["mean_score_spearman"], np.nan),
            yerr=numeric(sub["sd_score_spearman"], 0.0),
            width=width,
            color=colors[model],
            alpha=0.88,
            capsize=3,
            label=model,
        )
    ax.set_xticks(x)
    ax.set_xticklabels([VALIDATION_LABELS[v] for v in VALIDATION_ORDER], rotation=10, ha="right")
    ax.set_ylabel("Test Spearman, mean +/- SD")
    ax.set_title("Leakage-controlled graph-only HGNN versus feature baselines")
    ax.set_ylim(0.0, 1.0)
    ax.grid(axis="y", alpha=0.22)
    ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.25), fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    for suffix in ["png", "pdf", "svg"]:
        fig.savefig(FIG_DIR / f"Fig_GRAPHONLY1_spearman_vs_feature_baselines.{suffix}", dpi=360 if suffix == "png" else None, bbox_inches="tight")
    plt.close(fig)
    figures.append({"figure": "Fig_GRAPHONLY1_spearman_vs_feature_baselines", "png": str(FIG_DIR / "Fig_GRAPHONLY1_spearman_vs_feature_baselines.png")})

    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    d = delta.loc[delta["validation_type"].eq("all_strict_splits")].copy()
    ax.bar(np.arange(len(d)), numeric(d["mean_delta_spearman"], np.nan), color=[colors.get(x, "#777777") for x in d["reference_model"]], alpha=0.88)
    low = numeric(d["mean_delta_spearman"], np.nan) - numeric(d["bootstrap_ci95_low"], np.nan)
    high = numeric(d["bootstrap_ci95_high"], np.nan) - numeric(d["mean_delta_spearman"], np.nan)
    ax.errorbar(np.arange(len(d)), numeric(d["mean_delta_spearman"], np.nan), yerr=np.vstack([low, high]), fmt="none", ecolor="#222222", capsize=4)
    ax.axhline(0, color="#222222", lw=1)
    ax.set_xticks(np.arange(len(d)))
    ax.set_xticklabels(d["reference_model"], rotation=20, ha="right")
    ax.set_ylabel("Delta Spearman")
    ax.set_title("Graph-only HGNN paired improvement over controls")
    ax.grid(axis="y", alpha=0.22)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    for suffix in ["png", "pdf", "svg"]:
        fig.savefig(FIG_DIR / f"Fig_GRAPHONLY2_paired_delta.{suffix}", dpi=360 if suffix == "png" else None, bbox_inches="tight")
    plt.close(fig)
    figures.append({"figure": "Fig_GRAPHONLY2_paired_delta", "png": str(FIG_DIR / "Fig_GRAPHONLY2_paired_delta.png")})

    fig, ax = plt.subplots(figsize=(8.0, 4.6))
    methods = ["raw", "linear", "isotonic"]
    method_colors = {"raw": "#B93A3A", "linear": "#5BAE96", "isotonic": "#4F73C7"}
    x = np.arange(len(VALIDATION_ORDER))
    width = 0.24
    for i, method in enumerate(methods):
        sub = cal_summary.loc[cal_summary["calibration_method"].eq(method)].set_index("validation_type").reindex(VALIDATION_ORDER)
        ax.bar(x + (i - 1) * width, numeric(sub["mean_ece_10bin"], np.nan), width=width, color=method_colors[method], alpha=0.88, label=method)
    ax.set_xticks(x)
    ax.set_xticklabels([VALIDATION_LABELS[v] for v in VALIDATION_ORDER], rotation=10, ha="right")
    ax.set_ylabel("Continuous score ECE, 10 bins")
    ax.set_title("Graph-only HGNN post-hoc calibration")
    ax.grid(axis="y", alpha=0.22)
    ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.14), fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    for suffix in ["png", "pdf", "svg"]:
        fig.savefig(FIG_DIR / f"Fig_GRAPHONLY3_calibration_ece.{suffix}", dpi=360 if suffix == "png" else None, bbox_inches="tight")
    plt.close(fig)
    figures.append({"figure": "Fig_GRAPHONLY3_calibration_ece", "png": str(FIG_DIR / "Fig_GRAPHONLY3_calibration_ece.png")})

    fig, ax = plt.subplots(figsize=(5.8, 5.4))
    bins = cal_bins.loc[cal_bins["calibration_method"].isin(["raw", "isotonic"])].copy()
    for method, color in [("raw", "#B93A3A"), ("isotonic", "#4F73C7")]:
        sub = bins.loc[bins["calibration_method"].eq(method)].groupby("bin")[["mean_pred", "mean_true"]].mean().dropna()
        ax.plot(sub["mean_pred"], sub["mean_true"], marker="o", color=color, label=method)
    ax.plot([0, 1], [0, 1], color="#222222", lw=1, ls="--")
    ax.set_xlabel("Mean predicted score")
    ax.set_ylabel("Mean true soft label")
    ax.set_title("Graph-only reliability diagram")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.22)
    ax.legend(frameon=False, loc="upper left")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    for suffix in ["png", "pdf", "svg"]:
        fig.savefig(FIG_DIR / f"Fig_GRAPHONLY4_reliability_diagram.{suffix}", dpi=360 if suffix == "png" else None, bbox_inches="tight")
    plt.close(fig)
    figures.append({"figure": "Fig_GRAPHONLY4_reliability_diagram", "png": str(FIG_DIR / "Fig_GRAPHONLY4_reliability_diagram.png")})
    return figures


def write_report(summary: pd.DataFrame, delta: pd.DataFrame, cal_summary: pd.DataFrame, figures: list[dict[str, str]], device_name: str, start_time: str, end_time: str) -> Path:
    report = PROJECT_ROOT / f"SPATIAL_CAUSAL_HGNN_GRAPH_ONLY_CALIBRATION_REPORT_{RUN_DATE}.md"
    lines = [
        f"# Spatial-Causal-HGNN graph-only calibration report ({RUN_DATE})",
        "",
        "## Run Contract",
        "",
        f"- Device: `{device_name}`",
        f"- Seeds: `{';'.join(map(str, SEEDS))}`",
        f"- Started: `{start_time}`",
        f"- Finished: `{end_time}`",
        "- No pairwise/listwise ranking loss was used.",
        "- The graph-only model does not consume direct axis evidence-vector features.",
        "- Leakage-controlled feature baselines use cancer-only or unweighted edge-count features; full evidence feature-only is retained as an upper-bound leakage audit.",
        "",
        "## Model Comparison",
        "",
        summary.sort_values(["validation_type", "model"]).to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Paired Delta For Graph-Only HGNN",
        "",
        delta.loc[delta["validation_type"].eq("all_strict_splits")].to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Calibration Summary",
        "",
        cal_summary.sort_values(["validation_type", "calibration_method"]).to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Interpretation",
        "",
        "- This analysis tests whether graph message passing helps when direct evidence-vector leakage is blocked.",
        "- If graph-only HGNN beats minimal/edge-count controls but not full evidence feature-only, the correct conclusion is that GNN adds value through relational evidence propagation, while the current soft label is still linearly reconstructable from its source evidence features.",
        "- Manuscript claims should therefore avoid saying the graph model outperforms all feature-only baselines on the same evidence-derived label. The stronger claim is graph-constrained prioritization, path interpretability, and leakage-controlled relational learning.",
        "",
        "## Outputs",
        "",
        f"- Tables: `{TABLE_DIR}`",
        f"- Figures: `{FIG_DIR}`",
    ]
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def main() -> None:
    ensure_dirs()
    start_time = datetime.now().isoformat(timespec="seconds")
    raw = torch.load(DATASET_PATH, map_location="cpu", weights_only=False)
    labels = pd.read_csv(LABEL_TABLE)
    feature_mats = build_feature_matrices(labels, raw)
    split_configs = build_split_configs(labels)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"

    metric_parts = []
    history_parts = []
    pred_parts = []
    for seed in SEEDS:
        for _, split_row in split_configs.iterrows():
            graph_metrics, graph_history, graph_predictions = train_graph_only(raw, labels, split_row, device, seed)
            feat_metrics, feat_predictions = run_feature_baselines(labels, feature_mats, split_row, seed)
            metric_parts.extend([graph_metrics, feat_metrics])
            history_parts.append(graph_history)
            pred_parts.extend([graph_predictions, feat_predictions])

    new_metrics = pd.concat(metric_parts, ignore_index=True)
    history = pd.concat(history_parts, ignore_index=True)
    predictions = pd.concat(pred_parts, ignore_index=True)
    existing = pd.read_csv(EXISTING_METRICS)
    existing = existing.loc[existing["model"].eq(CURRENT_HGNN)].copy()
    combined_metrics = pd.concat([existing, new_metrics], ignore_index=True, sort=False)
    summary = summarize_metrics(combined_metrics)
    delta = paired_deltas(combined_metrics)
    calibration, cal_bins, calibrated_predictions = calibrate_graph_predictions(predictions)
    cal_summary = summarize_calibration(calibration)

    write_table(new_metrics, "Table_GRAPHONLY1_new_metrics_by_split_seed")
    write_table(history, "Table_GRAPHONLY2_training_history")
    write_table(predictions, "Table_GRAPHONLY3_predictions_all_splits")
    write_table(summary, "Table_GRAPHONLY4_model_comparison_summary")
    write_table(delta, "Table_GRAPHONLY5_paired_delta")
    write_table(calibration, "Table_GRAPHONLY6_calibration_metrics")
    write_table(cal_summary, "Table_GRAPHONLY7_calibration_summary")
    write_table(cal_bins, "Table_GRAPHONLY8_reliability_bins")
    write_table(calibrated_predictions, "Table_GRAPHONLY9_calibrated_test_predictions")
    feature_catalog = pd.DataFrame(
        [{"feature_set": name, "n_features": int(df.shape[1]), "features": ";".join(map(str, df.columns[:80]))} for name, df in feature_mats.items()]
        + [
            {"feature_set": "graph_only_hgnn", "n_features": 0, "features": "No direct axis evidence vector; graph uses node_struct_features, relation types, edge weights, and message passing."}
        ]
    )
    write_table(feature_catalog, "Table_GRAPHONLY10_feature_set_contract")
    figures = make_figures(summary, delta, cal_summary, cal_bins)
    end_time = datetime.now().isoformat(timespec="seconds")
    report = write_report(summary, delta, cal_summary, figures, device_name, start_time, end_time)
    manifest = {
        "run_date": RUN_DATE,
        "created_at": end_time,
        "run_id": RUN_ID,
        "device": device_name,
        "seeds": SEEDS,
        "dataset_path": str(DATASET_PATH),
        "label_table": str(LABEL_TABLE),
        "n_split_configs": int(len(split_configs)),
        "n_new_metric_rows": int(len(new_metrics)),
        "n_prediction_rows": int(len(predictions)),
        "tables": str(TABLE_DIR),
        "figures": str(FIG_DIR),
        "report": str(report),
    }
    manifest_path = METADATA_DIR / f"run_manifest_spatial_causal_hgnn_graph_only_calibration_{RUN_DATE}.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
