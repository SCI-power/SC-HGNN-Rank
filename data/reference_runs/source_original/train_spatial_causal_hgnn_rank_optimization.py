from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from scipy.stats import wilcoxon
from torch import nn
from torch.nn import functional as F

from train_spatial_causal_hgnn_generalization import build_split_configs, mean_by_cancer_baseline, parse_indices
from train_spatial_causal_hgnn_model_comparison import (
    AXIS_FEATURE_COLS,
    HGTLikeLayer,
    add_structural_features,
    move_to_device,
    pearson_corr,
    spearman_corr,
)
from train_spatial_causal_hgnn_statistical_validation import bootstrap_ci, summarize_topk, topk_metrics


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUN_DATE = "2026-05-04"
RUN_ID = "spatial_causal_hgnn_rank_optimization_2026-05-04"

MODEL_DIR = PROJECT_ROOT / "models" / "spatial_causal_hgnn"
TABLE_DIR = PROJECT_ROOT / "tables" / "spatial_causal_hgnn_rank_optimization"
FIG_DIR = PROJECT_ROOT / "figures" / "spatial_causal_hgnn_rank_optimization"
METADATA_DIR = PROJECT_ROOT / "metadata"
ARTIFACT_DIR = PROJECT_ROOT / "artifacts" / "experiment" / RUN_ID

DATASET_PATH = MODEL_DIR / f"spatial_causal_hgnn_tensor_dataset_{RUN_DATE}.pt"
LABEL_TABLE = PROJECT_ROOT / "tables" / "spatial_causal_hgnn_graph_inputs" / "Table_HGNN3_axis_label_table.csv"
DECOY_TABLE = PROJECT_ROOT / "tables" / "spatial_causal_hgnn_graph_inputs" / "Table_HGNN7_negative_axis_decoy_pool.csv"
EXISTING_METRICS = PROJECT_ROOT / "tables" / "spatial_causal_hgnn_statistical_validation" / "Table_HGNN_STAT1_multiseed_metrics_by_model_split_seed.csv"
EXISTING_PREDICTIONS = PROJECT_ROOT / "tables" / "spatial_causal_hgnn_statistical_validation" / "Table_HGNN_STAT12_test_predictions_multiseed.csv"

PROPOSED_MODEL = "Spatial-Causal-HGNN-Rank"
REFERENCE_MODEL = "Spatial-Causal-HGNN"
BASELINE_MODELS = ["GCN", "R-GCN", "GAT", "HGT-like"]
VALIDATION_ORDER = ["leave_one_cancer", "leave_compound_group", "leave_target_group"]
VALIDATION_LABELS = {
    "leave_one_cancer": "Leave-one-cancer",
    "leave_compound_group": "Leave-compound",
    "leave_target_group": "Leave-target",
}
SEEDS = [20260504, 20260505, 20260506, 20260507, 20260508]
TOPK_COMPARE_MODELS = ["GCN", "R-GCN", "GAT", "HGT-like", REFERENCE_MODEL, PROPOSED_MODEL]


INTERACTION_FEATURES = [
    "metapath_spatial_grn",
    "metapath_spatial_deconv",
    "metapath_drug_grn",
    "metapath_drug_spatial",
    "metapath_drug_context",
    "metapath_ppi_clinical",
    "metapath_lincs_depmap",
    "metapath_full_chain",
    "metapath_external_mean",
    "evidence_nonzero_fraction",
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


def add_axis_interactions(labels: pd.DataFrame) -> pd.DataFrame:
    frame = labels.copy()
    for col in AXIS_FEATURE_COLS:
        if col not in frame.columns:
            frame[col] = 0.0
        frame[col] = numeric(frame[col]).clip(0, 1)

    c = frame
    eps = 1e-6
    c["metapath_spatial_grn"] = c["integrated_spatial_recurrence_score"] * c["single_cell_regulatory_score"]
    c["metapath_spatial_deconv"] = c["integrated_spatial_recurrence_score"] * c["rctd_spotlight_consistency_score"]
    c["metapath_drug_grn"] = c["drug_evidence_score"] * c["single_cell_regulatory_score"]
    c["metapath_drug_spatial"] = c["drug_evidence_score"] * c["integrated_spatial_recurrence_score"]
    c["metapath_drug_context"] = c["drug_evidence_score"] * c["cancer_context_specificity_score"]
    c["metapath_ppi_clinical"] = c["ppi_msi_local_support_score"] * c["same_cancer_clinical_prognosis_score"]
    c["metapath_lincs_depmap"] = c["lincs_cmap_combined_support_score"] * c["depmap_axis_support_score"]
    c["metapath_full_chain"] = np.power(
        (
            c["drug_evidence_score"].clip(eps, 1)
            * c["single_cell_regulatory_score"].clip(eps, 1)
            * c["integrated_spatial_recurrence_score"].clip(eps, 1)
            * c["cancer_context_specificity_score"].clip(eps, 1)
        ),
        0.25,
    )
    c["metapath_external_mean"] = c[
        [
            "lincs_cmap_combined_support_score",
            "depmap_axis_support_score",
            "non_broad_cancer_sensitivity_score",
            "same_cancer_clinical_prognosis_score",
        ]
    ].mean(axis=1)
    c["evidence_nonzero_fraction"] = (c[AXIS_FEATURE_COLS] > 0.0).mean(axis=1)
    for col in INTERACTION_FEATURES:
        c[col] = numeric(c[col]).clip(0, 1)
    return c


def make_axis_feature_tensors(labels: pd.DataFrame) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
    frame = add_axis_interactions(labels)
    feature_cols = AXIS_FEATURE_COLS + INTERACTION_FEATURES
    axis_features = frame[feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).clip(0, 1)
    meta_features = frame[INTERACTION_FEATURES].apply(pd.to_numeric, errors="coerce").fillna(0.0).clip(0, 1)
    return (
        torch.tensor(axis_features.values, dtype=torch.float32),
        torch.tensor(meta_features.values, dtype=torch.float32),
        feature_cols,
    )


class RankOptimizedSpatialCausalHGNN(nn.Module):
    def __init__(
        self,
        num_node_types: int,
        num_edge_types: int,
        axis_dim: int,
        meta_dim: int,
        hidden_dim: int = 112,
        num_layers: int = 2,
        dropout: float = 0.18,
    ) -> None:
        super().__init__()
        self.node_type_embedding = nn.Embedding(num_node_types, hidden_dim)
        self.feature_projection = nn.Linear(4, hidden_dim)
        self.layers = nn.ModuleList([HGTLikeLayer(hidden_dim, num_edge_types, dropout) for _ in range(num_layers)])
        self.axis_encoder = nn.Sequential(
            nn.Linear(axis_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.meta_encoder = nn.Sequential(
            nn.Linear(meta_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.path_attention = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 3),
        )
        self.fusion = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Linear(hidden_dim, 3)
        self.regressor = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(
        self,
        data: dict[str, torch.Tensor],
        axis_features: torch.Tensor,
        meta_features: torch.Tensor,
        return_attention: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.node_type_embedding(data["node_type_idx"]) + self.feature_projection(data["node_struct_features"])
        for layer in self.layers:
            h = layer(h, data["edge_src"], data["edge_dst"], data["edge_type_idx"], data["edge_weight"])
        graph_h = h[data["axis_node_idx"]]
        axis_h = self.axis_encoder(axis_features)
        meta_h = self.meta_encoder(meta_features)
        attn = torch.softmax(self.path_attention(torch.cat([graph_h, axis_h, meta_h], dim=1)), dim=1)
        fused = (
            attn[:, 0:1] * graph_h
            + attn[:, 1:2] * axis_h
            + attn[:, 2:3] * meta_h
        )
        fused = self.fusion(fused)
        logits = self.classifier(fused)
        score = torch.sigmoid(self.regressor(fused)).squeeze(-1)
        if return_attention:
            return logits, score, attn
        return logits, score


def weighted_mean(values: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    return (values * weights).sum() / weights.sum().clamp_min(1e-6)


def sample_weight_vector(labels_df: pd.DataFrame) -> torch.Tensor:
    label = numeric(labels_df["hgnn_label_ordinal"]).astype(int)
    score = numeric(labels_df["soft_rank_label"])
    nz = (labels_df[AXIS_FEATURE_COLS].apply(pd.to_numeric, errors="coerce").fillna(0.0) > 0).mean(axis=1)
    base = label.map({0: 0.62, 1: 1.00, 2: 1.25}).fillna(0.75).astype(float)
    confidence = 0.75 + 0.25 * nz.astype(float)
    top_boost = np.where(score >= score.quantile(0.80), 1.12, 1.0)
    return torch.tensor((base.values * confidence.values * top_boost).astype(float), dtype=torch.float32)


def build_pairwise_pairs(
    labels_df: pd.DataFrame,
    train_idx: np.ndarray,
    seed: int,
    min_delta: float = 0.045,
    max_pairs: int = 4096,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    rng = np.random.default_rng(seed)
    train = labels_df.iloc[train_idx].copy()
    rows: list[tuple[int, int, float]] = []
    for _, df in train.groupby("cancer"):
        idx = df.index.to_numpy(dtype=int)
        score = numeric(df["soft_rank_label"]).to_numpy(dtype=float)
        for a_pos, i in enumerate(idx):
            delta = score[a_pos] - score
            good = np.where(delta >= min_delta)[0]
            for j_pos in good:
                rows.append((int(i), int(idx[j_pos]), float(min(2.0, 0.6 + delta[j_pos] * 4.0))))
    if len(rows) > max_pairs:
        weights = np.array([r[2] for r in rows], dtype=float)
        prob = weights / weights.sum()
        take = rng.choice(len(rows), size=max_pairs, replace=False, p=prob)
        rows = [rows[i] for i in take]
    if not rows:
        return torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.float32)
    pos, neg, weight = zip(*rows)
    return torch.tensor(pos, dtype=torch.long), torch.tensor(neg, dtype=torch.long), torch.tensor(weight, dtype=torch.float32)


def build_decoy_guided_pairs(
    labels_df: pd.DataFrame,
    decoys: pd.DataFrame,
    train_idx: np.ndarray,
    max_pairs: int = 2048,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, pd.DataFrame]:
    if decoys.empty:
        audit = pd.DataFrame([{"negative_reason": "no_decoy_table", "n_pairs": 0}])
        return torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.float32), audit

    train_set = set(map(int, train_idx))
    axis_to_idx = dict(zip(labels_df["frozen_axis_id"].astype(str), labels_df.index.astype(int)))
    rows: list[tuple[int, int, float, str]] = []
    labels = labels_df.copy()
    labels["_score"] = numeric(labels["soft_rank_label"])
    labels["_label"] = numeric(labels["hgnn_label_ordinal"]).astype(int)

    for _, decoy in decoys.iterrows():
        source = str(decoy.get("source_positive_axis_id", ""))
        if source not in axis_to_idx:
            continue
        src_idx = int(axis_to_idx[source])
        if src_idx not in train_set or int(labels.at[src_idx, "_label"]) < 1:
            continue
        source_row = labels.loc[src_idx]
        cancer = source_row["cancer"]
        reason = str(decoy.get("negative_reason", "unknown"))
        pool = labels.loc[labels.index.isin(train_set) & labels["cancer"].eq(cancer) & (labels.index != src_idx)].copy()
        if pool.empty:
            continue
        if reason == "same_compound_wrong_target":
            cand = pool.loc[
                pool["compound_master_id"].astype(str).eq(str(source_row["compound_master_id"]))
                & ~pool["target_gene_symbol"].astype(str).eq(str(source_row["target_gene_symbol"]))
            ]
        elif reason == "wrong_compound_same_context":
            cand = pool.loc[
                pool["target_gene_symbol"].astype(str).eq(str(source_row["target_gene_symbol"]))
                & pool["dominant_spatial_celltype"].astype(str).eq(str(source_row["dominant_spatial_celltype"]))
                & ~pool["compound_master_id"].astype(str).eq(str(source_row["compound_master_id"]))
            ]
        elif reason == "same_target_wrong_cell_state":
            cand = pool.loc[
                pool["target_gene_symbol"].astype(str).eq(str(source_row["target_gene_symbol"]))
                & ~pool["dominant_spatial_celltype"].astype(str).eq(str(source_row["dominant_spatial_celltype"]))
            ]
        else:
            cand = pool
        cand = cand.loc[cand["_score"] < float(source_row["_score"]) - 0.02]
        if cand.empty:
            continue
        target_idx = int(cand.sort_values("_score").index[0])
        weight = 1.4 if int(labels.at[target_idx, "_label"]) == 0 else 1.1
        rows.append((src_idx, target_idx, weight, reason))

    if len(rows) > max_pairs:
        rows = rows[:max_pairs]
    audit = (
        pd.DataFrame(rows, columns=["positive_index", "negative_index", "pair_weight", "negative_reason"])
        .groupby("negative_reason", dropna=False)
        .size()
        .reset_index(name="n_pairs")
    ) if rows else pd.DataFrame([{"negative_reason": "no_observed_decoy_pairs", "n_pairs": 0}])
    if not rows:
        return torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.float32), audit
    pos, neg, weight, _reason = zip(*rows)
    return torch.tensor(pos, dtype=torch.long), torch.tensor(neg, dtype=torch.long), torch.tensor(weight, dtype=torch.float32), audit


def build_context_pairs(labels_df: pd.DataFrame, train_idx: np.ndarray, max_pairs: int = 2048) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    train = labels_df.iloc[train_idx].copy()
    train["_score"] = numeric(train["soft_rank_label"])
    rows: list[tuple[int, int, float]] = []
    for _, df in train.groupby(["compound_master_id", "target_gene_symbol"], dropna=False):
        if df["cancer"].nunique() < 2:
            continue
        idx = df.index.to_numpy(dtype=int)
        score = df["_score"].to_numpy(dtype=float)
        for a in range(len(idx)):
            for b in range(a + 1, len(idx)):
                delta = float(score[a] - score[b])
                if abs(delta) < 0.04:
                    continue
                if delta > 0:
                    rows.append((int(idx[a]), int(idx[b]), abs(delta)))
                else:
                    rows.append((int(idx[b]), int(idx[a]), abs(delta)))
    rows = sorted(rows, key=lambda x: x[2], reverse=True)[:max_pairs]
    if not rows:
        return torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.float32)
    a, b, w = zip(*rows)
    return torch.tensor(a, dtype=torch.long), torch.tensor(b, dtype=torch.long), torch.tensor(w, dtype=torch.float32)


def build_listwise_groups(labels_df: pd.DataFrame, train_idx: np.ndarray) -> list[torch.Tensor]:
    groups = []
    train = labels_df.iloc[train_idx].copy()
    for _, df in train.groupby("cancer"):
        idx = torch.tensor(df.index.to_numpy(dtype=int), dtype=torch.long)
        if len(idx) > 2:
            groups.append(idx)
    return groups


def pairwise_rank_loss(pred_score: torch.Tensor, pos: torch.Tensor, neg: torch.Tensor, weight: torch.Tensor, temperature: float = 0.070) -> torch.Tensor:
    if pos.numel() == 0:
        return pred_score.new_tensor(0.0)
    diff = pred_score[pos] - pred_score[neg]
    loss = F.softplus(-diff / temperature)
    return weighted_mean(loss, weight)


def listwise_rank_loss(pred_score: torch.Tensor, true_score: torch.Tensor, groups: list[torch.Tensor], temperature: float = 0.080) -> torch.Tensor:
    losses = []
    for idx in groups:
        target = torch.softmax(true_score[idx] / temperature, dim=0)
        log_pred = torch.log_softmax(pred_score[idx] / temperature, dim=0)
        losses.append(-(target * log_pred).sum())
    if not losses:
        return pred_score.new_tensor(0.0)
    return torch.stack(losses).mean()


def evaluate_predictions(
    logits: torch.Tensor,
    pred_score: torch.Tensor,
    labels: torch.Tensor,
    score: torch.Tensor,
    idx: torch.Tensor,
    split: str,
) -> dict[str, Any]:
    y_true = labels[idx].detach().cpu().numpy()
    y_pred = logits[idx].argmax(dim=1).detach().cpu().numpy()
    score_true = score[idx].detach().cpu().numpy()
    score_pred = pred_score[idx].detach().cpu().numpy()
    from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, r2_score

    return {
        "split": split,
        "n": int(len(idx)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "score_mae": float(mean_absolute_error(score_true, score_pred)),
        "score_r2": float(r2_score(score_true, score_pred)) if len(np.unique(score_true)) > 1 else float("nan"),
        "score_pearson": pearson_corr(score_true, score_pred),
        "score_spearman": spearman_corr(score_true, score_pred),
    }


def train_one_rank_model(
    raw: dict[str, Any],
    labels_df: pd.DataFrame,
    decoys: pd.DataFrame,
    axis_features_cpu: torch.Tensor,
    meta_features_cpu: torch.Tensor,
    split_row: pd.Series,
    device: torch.device,
    seed: int,
    max_epochs: int = 160,
    patience: int = 28,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    data = move_to_device(add_structural_features(raw), device)
    axis_features = axis_features_cpu.to(device)
    meta_features = meta_features_cpu.to(device)
    weights = sample_weight_vector(labels_df).to(device)
    metadata = raw["metadata"]

    model = RankOptimizedSpatialCausalHGNN(
        num_node_types=int(metadata["num_node_types"]),
        num_edge_types=int(metadata["num_edge_types"]),
        axis_dim=axis_features.shape[1],
        meta_dim=meta_features.shape[1],
        hidden_dim=112,
        num_layers=2,
        dropout=0.18,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.6e-3, weight_decay=1.5e-4)

    train_idx = parse_indices(split_row["train_indices"]).to(device)
    val_idx = parse_indices(split_row["val_indices"]).to(device)
    test_idx = parse_indices(split_row["test_indices"]).to(device)
    train_np = parse_indices(split_row["train_indices"]).numpy()
    labels = data["axis_label"]
    soft_score = data["axis_soft_score"]

    pair_pos, pair_neg, pair_w = build_pairwise_pairs(labels_df, train_np, seed)
    hard_pos, hard_neg, hard_w, hard_audit = build_decoy_guided_pairs(labels_df, decoys, train_np)
    ctx_pos, ctx_neg, ctx_w = build_context_pairs(labels_df, train_np)
    list_groups = build_listwise_groups(labels_df, train_np)
    pair_pos, pair_neg, pair_w = pair_pos.to(device), pair_neg.to(device), pair_w.to(device)
    hard_pos, hard_neg, hard_w = hard_pos.to(device), hard_neg.to(device), hard_w.to(device)
    ctx_pos, ctx_neg, ctx_w = ctx_pos.to(device), ctx_neg.to(device), ctx_w.to(device)
    list_groups = [g.to(device) for g in list_groups]

    best_state: dict[str, torch.Tensor] | None = None
    best_val_loss = float("inf")
    stale = 0
    history_rows: list[dict[str, Any]] = []
    for epoch in range(1, max_epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits, pred_score = model(data, axis_features, meta_features)
        loss_cls_vec = F.cross_entropy(logits[train_idx], labels[train_idx], reduction="none")
        loss_cls = weighted_mean(loss_cls_vec, weights[train_idx])
        loss_reg_vec = F.smooth_l1_loss(pred_score[train_idx], soft_score[train_idx], reduction="none")
        loss_reg = weighted_mean(loss_reg_vec, weights[train_idx])
        loss_pair = pairwise_rank_loss(pred_score, pair_pos, pair_neg, pair_w)
        loss_hard = pairwise_rank_loss(pred_score, hard_pos, hard_neg, hard_w)
        loss_list = listwise_rank_loss(pred_score, soft_score, list_groups)
        loss_context = pairwise_rank_loss(pred_score, ctx_pos, ctx_neg, ctx_w, temperature=0.090)
        loss = loss_cls + 0.25 * loss_reg + 0.55 * loss_pair + 0.35 * loss_hard + 0.20 * loss_list + 0.12 * loss_context
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        optimizer.step()

        model.eval()
        with torch.no_grad():
            logits_eval, score_eval = model(data, axis_features, meta_features)
            val_loss_cls = F.cross_entropy(logits_eval[val_idx], labels[val_idx])
            val_loss_reg = F.smooth_l1_loss(score_eval[val_idx], soft_score[val_idx])
            val_rank = pairwise_rank_loss(score_eval, pair_pos, pair_neg, pair_w)
            val_loss = val_loss_cls + 0.25 * val_loss_reg + 0.20 * val_rank
            val_metrics = evaluate_predictions(logits_eval, score_eval, labels, soft_score, val_idx, "val")

        history_rows.append(
            {
                "model": PROPOSED_MODEL,
                "seed": seed,
                "validation_type": split_row["validation_type"],
                "fold_id": split_row["fold_id"],
                "epoch": epoch,
                "train_loss": float(loss.detach().cpu()),
                "loss_cls": float(loss_cls.detach().cpu()),
                "loss_reg": float(loss_reg.detach().cpu()),
                "loss_pairwise": float(loss_pair.detach().cpu()),
                "loss_hard_negative": float(loss_hard.detach().cpu()),
                "loss_listwise": float(loss_list.detach().cpu()),
                "loss_context": float(loss_context.detach().cpu()),
                "val_loss": float(val_loss.detach().cpu()),
                "val_macro_f1": float(val_metrics["macro_f1"]),
                "val_score_spearman": float(val_metrics["score_spearman"]),
                "n_pairwise_pairs": int(pair_pos.numel()),
                "n_hard_negative_pairs": int(hard_pos.numel()),
                "n_context_pairs": int(ctx_pos.numel()),
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
        logits, pred_score, attn = model(data, axis_features, meta_features, return_attention=True)
        pred_prob = torch.softmax(logits, dim=1).detach().cpu().numpy()
        attn_np = attn.detach().cpu().numpy()

    metric_rows: list[dict[str, Any]] = []
    for split_name, idx in [("train", train_idx), ("val", val_idx), ("test", test_idx)]:
        row = evaluate_predictions(logits, pred_score, labels, soft_score, idx, split_name)
        row.update(
            {
                "model": PROPOSED_MODEL,
                "ablation_variant": "rank_optimized_full",
                "seed": seed,
                "validation_type": split_row["validation_type"],
                "fold_id": split_row["fold_id"],
                "heldout_group": split_row["heldout_group"],
                "best_val_loss": best_val_loss,
                "epochs_completed": len(history_rows),
                "n_pairwise_pairs": int(pair_pos.numel()),
                "n_hard_negative_pairs": int(hard_pos.numel()),
                "n_context_pairs": int(ctx_pos.numel()),
            }
        )
        metric_rows.append(row)

    baseline = mean_by_cancer_baseline(
        labels_df,
        parse_indices(split_row["train_indices"]).numpy(),
        parse_indices(split_row["test_indices"]).numpy(),
    )
    for row in metric_rows:
        if row["split"] == "test":
            row.update(baseline)

    test_np = parse_indices(split_row["test_indices"]).numpy()
    predictions = labels_df.iloc[test_np].copy()
    predictions["model"] = PROPOSED_MODEL
    predictions["ablation_variant"] = "rank_optimized_full"
    predictions["seed"] = seed
    predictions["validation_type"] = split_row["validation_type"]
    predictions["fold_id"] = split_row["fold_id"]
    predictions["predicted_label"] = logits.argmax(dim=1).detach().cpu().numpy()[test_np]
    predictions["prob_unlabeled_or_negative"] = pred_prob[test_np, 0]
    predictions["prob_moderate_positive"] = pred_prob[test_np, 1]
    predictions["prob_high_positive"] = pred_prob[test_np, 2]
    predictions["predicted_soft_score"] = pred_score.detach().cpu().numpy()[test_np]
    predictions["absolute_score_error"] = (numeric(predictions["soft_rank_label"]) - predictions["predicted_soft_score"]).abs()
    predictions["path_attention_graph"] = attn_np[test_np, 0]
    predictions["path_attention_axis_features"] = attn_np[test_np, 1]
    predictions["path_attention_metapath_features"] = attn_np[test_np, 2]

    hard_audit = hard_audit.copy()
    hard_audit["seed"] = seed
    hard_audit["validation_type"] = split_row["validation_type"]
    hard_audit["fold_id"] = split_row["fold_id"]
    return pd.DataFrame(metric_rows), pd.DataFrame(history_rows), predictions, hard_audit


def summarize_multiseed(metrics: pd.DataFrame) -> pd.DataFrame:
    test = metrics.loc[metrics["split"].eq("test")].copy()
    rows: list[dict[str, Any]] = []
    for (validation_type, model), df in test.groupby(["validation_type", "model"], sort=False):
        vals = numeric(df["score_spearman"], np.nan).dropna()
        rows.append(
            {
                "validation_type": validation_type,
                "model": model,
                "n_seed_fold_runs": int(len(vals)),
                "n_seeds": int(df["seed"].nunique()),
                "n_folds": int(df["fold_id"].nunique()),
                "mean_accuracy": float(numeric(df["accuracy"], np.nan).mean()),
                "mean_macro_f1": float(numeric(df["macro_f1"], np.nan).mean()),
                "mean_score_mae": float(numeric(df["score_mae"], np.nan).mean()),
                "sd_score_mae": float(numeric(df["score_mae"], np.nan).std(ddof=1)),
                "mean_score_spearman": float(vals.mean()),
                "sd_score_spearman": float(vals.std(ddof=1)),
                "median_score_spearman": float(vals.median()),
            }
        )
    return pd.DataFrame(rows)


def paired_delta_table(combined_metrics: pd.DataFrame) -> pd.DataFrame:
    test = combined_metrics.loc[combined_metrics["split"].eq("test")].copy()
    key = ["validation_type", "fold_id", "seed"]
    rows: list[dict[str, Any]] = []
    prop = test.loc[test["model"].eq(PROPOSED_MODEL), key + ["score_spearman"]].rename(columns={"score_spearman": "proposed"})
    for validation_type, dfv in test.groupby("validation_type"):
        prop_v = prop.loc[prop["validation_type"].eq(validation_type)]
        means = dfv.groupby("model")["score_spearman"].mean().sort_values(ascending=False)
        candidate_refs = [REFERENCE_MODEL] + [m for m in BASELINE_MODELS if m in means.index]
        best_graph = means.loc[[m for m in BASELINE_MODELS if m in means.index]].idxmax()
        for ref in candidate_refs + ["best_standard_graph"]:
            ref_model = best_graph if ref == "best_standard_graph" else ref
            base = dfv.loc[dfv["model"].eq(ref_model), key + ["score_spearman"]].rename(columns={"score_spearman": "reference"})
            merged = prop_v.merge(base, on=key, how="inner")
            diffs = numeric(merged["proposed"], np.nan) - numeric(merged["reference"], np.nan)
            diffs = diffs.dropna()
            if len(diffs) and np.any(np.abs(diffs.values) > 1e-12):
                try:
                    p_two = float(wilcoxon(diffs, zero_method="wilcox", alternative="two-sided").pvalue)
                    p_greater = float(wilcoxon(diffs, zero_method="wilcox", alternative="greater").pvalue)
                except ValueError:
                    p_two = float("nan")
                    p_greater = float("nan")
            else:
                p_two = float("nan")
                p_greater = float("nan")
            ci_low, ci_high = bootstrap_ci(diffs.values, seed=20260801 + len(rows))
            rows.append(
                {
                    "validation_type": validation_type,
                    "reference_model": ref_model,
                    "reference_role": ref,
                    "n_paired_seed_fold_runs": int(len(diffs)),
                    "mean_delta_spearman": float(diffs.mean()) if len(diffs) else float("nan"),
                    "sd_delta_spearman": float(diffs.std(ddof=1)) if len(diffs) > 1 else float("nan"),
                    "median_delta_spearman": float(diffs.median()) if len(diffs) else float("nan"),
                    "bootstrap_ci95_low": ci_low,
                    "bootstrap_ci95_high": ci_high,
                    "wilcoxon_p_two_sided": p_two,
                    "wilcoxon_p_greater": p_greater,
                }
            )

    all_rows = []
    for validation_type, dfv in test.groupby("validation_type"):
        means = dfv.groupby("model")["score_spearman"].mean().sort_values(ascending=False)
        best_graph = means.loc[[m for m in BASELINE_MODELS if m in means.index]].idxmax()
        p = prop.loc[prop["validation_type"].eq(validation_type)]
        for ref_role, ref_model in [("current_hgnn", REFERENCE_MODEL), ("best_standard_graph", best_graph)]:
            b = dfv.loc[dfv["model"].eq(ref_model), key + ["score_spearman"]].rename(columns={"score_spearman": "reference"})
            all_rows.append(p.merge(b, on=key, how="inner").assign(reference_model=ref_model, reference_role=ref_role))
    if all_rows:
        all_df = pd.concat(all_rows, ignore_index=True)
        for ref_role, df in all_df.groupby("reference_role"):
            diffs = numeric(df["proposed"], np.nan) - numeric(df["reference"], np.nan)
            diffs = diffs.dropna()
            try:
                p_two = float(wilcoxon(diffs, zero_method="wilcox", alternative="two-sided").pvalue)
                p_greater = float(wilcoxon(diffs, zero_method="wilcox", alternative="greater").pvalue)
            except ValueError:
                p_two = float("nan")
                p_greater = float("nan")
            ci_low, ci_high = bootstrap_ci(diffs.values, seed=20260831 + len(rows))
            rows.append(
                {
                    "validation_type": "all_strict_splits",
                    "reference_model": ";".join(sorted(df["reference_model"].unique())),
                    "reference_role": ref_role,
                    "n_paired_seed_fold_runs": int(len(diffs)),
                    "mean_delta_spearman": float(diffs.mean()),
                    "sd_delta_spearman": float(diffs.std(ddof=1)),
                    "median_delta_spearman": float(diffs.median()),
                    "bootstrap_ci95_low": ci_low,
                    "bootstrap_ci95_high": ci_high,
                    "wilcoxon_p_two_sided": p_two,
                    "wilcoxon_p_greater": p_greater,
                }
            )
    return pd.DataFrame(rows)


def topk_delta_table(topk_summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    rank = topk_summary.loc[topk_summary["model"].eq(PROPOSED_MODEL)].copy()
    refs = topk_summary.loc[topk_summary["model"].isin([REFERENCE_MODEL] + BASELINE_MODELS)].copy()
    for _, row in rank.iterrows():
        same = refs.loc[
            refs["validation_type"].eq(row["validation_type"])
            & refs["flag"].eq(row["flag"])
            & refs["k"].eq(row["k"])
        ].copy()
        if same.empty:
            continue
        ref_hgnn = same.loc[same["model"].eq(REFERENCE_MODEL)]
        best = same.sort_values("mean_ndcg", ascending=False).iloc[0]
        for role, ref in [("current_hgnn", ref_hgnn.iloc[0] if not ref_hgnn.empty else None), ("best_available_reference", best)]:
            if ref is None:
                continue
            rows.append(
                {
                    "validation_type": row["validation_type"],
                    "flag": row["flag"],
                    "k": int(row["k"]),
                    "reference_role": role,
                    "reference_model": ref["model"],
                    "rank_model_mean_hit_rate": row["mean_hit_rate"],
                    "reference_mean_hit_rate": ref["mean_hit_rate"],
                    "delta_hit_rate": row["mean_hit_rate"] - ref["mean_hit_rate"],
                    "rank_model_mean_enrichment": row["mean_enrichment"],
                    "reference_mean_enrichment": ref["mean_enrichment"],
                    "delta_enrichment": row["mean_enrichment"] - ref["mean_enrichment"],
                    "rank_model_mean_ndcg": row["mean_ndcg"],
                    "reference_mean_ndcg": ref["mean_ndcg"],
                    "delta_ndcg": row["mean_ndcg"] - ref["mean_ndcg"],
                }
            )
    return pd.DataFrame(rows)


def make_figures(
    summary: pd.DataFrame,
    deltas: pd.DataFrame,
    topk_summary: pd.DataFrame,
    predictions: pd.DataFrame,
    history: pd.DataFrame,
) -> list[dict[str, str]]:
    import matplotlib.pyplot as plt

    figures: list[dict[str, str]] = []
    colors = {
        "best_standard_graph": "#6D737A",
        REFERENCE_MODEL: "#4F73C7",
        PROPOSED_MODEL: "#B93A3A",
    }

    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    plot_rows = []
    for validation in VALIDATION_ORDER:
        sub = summary.loc[summary["validation_type"].eq(validation)]
        best_graph_mean = sub.loc[sub["model"].isin(BASELINE_MODELS)].sort_values("mean_score_spearman", ascending=False).head(1)
        if not best_graph_mean.empty:
            r = best_graph_mean.iloc[0].to_dict()
            r["model"] = "best_standard_graph"
            plot_rows.append(r)
        for model in [REFERENCE_MODEL, PROPOSED_MODEL]:
            tmp = sub.loc[sub["model"].eq(model)]
            if not tmp.empty:
                plot_rows.append(tmp.iloc[0].to_dict())
    plot = pd.DataFrame(plot_rows)
    x = np.arange(len(VALIDATION_ORDER))
    width = 0.23
    for i, model in enumerate(["best_standard_graph", REFERENCE_MODEL, PROPOSED_MODEL]):
        sub = plot.loc[plot["model"].eq(model)].set_index("validation_type").reindex(VALIDATION_ORDER)
        ax.bar(
            x + (i - 1) * width,
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
    ax.set_title("Rank-optimized HGNN versus current model and graph baselines")
    ax.set_ylim(0.70, 1.0)
    ax.grid(axis="y", alpha=0.22)
    ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.15), fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    for suffix in ["png", "pdf", "svg"]:
        fig.savefig(FIG_DIR / f"Fig_RANK1_strict_split_spearman.{suffix}", dpi=360 if suffix == "png" else None, bbox_inches="tight")
    plt.close(fig)
    figures.append({"figure": "Fig_RANK1_strict_split_spearman", "png": str(FIG_DIR / "Fig_RANK1_strict_split_spearman.png")})

    fig, ax = plt.subplots(figsize=(8.0, 4.6))
    d = deltas.loc[
        deltas["reference_role"].isin(["current_hgnn", "best_standard_graph"])
        & deltas["validation_type"].isin(VALIDATION_ORDER)
    ].copy()
    width = 0.34
    for i, ref_role in enumerate(["current_hgnn", "best_standard_graph"]):
        sub = d.loc[d["reference_role"].eq(ref_role)].set_index("validation_type").reindex(VALIDATION_ORDER)
        y = numeric(sub["mean_delta_spearman"], np.nan).values
        low = y - numeric(sub["bootstrap_ci95_low"], np.nan).values
        high = numeric(sub["bootstrap_ci95_high"], np.nan).values - y
        ax.bar(x + (i - 0.5) * width, y, width=width, color=["#4F73C7", "#6D737A"][i], alpha=0.88, label=f"vs {ref_role}")
        ax.errorbar(x + (i - 0.5) * width, y, yerr=np.vstack([low, high]), fmt="none", ecolor="#222222", capsize=3, lw=1.0)
    ax.axhline(0, color="#222222", lw=1)
    ax.set_xticks(x)
    ax.set_xticklabels([VALIDATION_LABELS[v] for v in VALIDATION_ORDER], rotation=10, ha="right")
    ax.set_ylabel("Delta Spearman")
    ax.set_title("Paired rank-model improvement with bootstrap 95% CI")
    ax.grid(axis="y", alpha=0.22)
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.14), ncol=2, fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    for suffix in ["png", "pdf", "svg"]:
        fig.savefig(FIG_DIR / f"Fig_RANK2_paired_delta.{suffix}", dpi=360 if suffix == "png" else None, bbox_inches="tight")
    plt.close(fig)
    figures.append({"figure": "Fig_RANK2_paired_delta", "png": str(FIG_DIR / "Fig_RANK2_paired_delta.png")})

    ext_flags = ["publication_core", "lincs_reversal", "depmap_sensitivity", "clinical_risk", "spatial_recurrence"]
    ext = topk_summary.loc[
        topk_summary["model"].eq(PROPOSED_MODEL)
        & topk_summary["k"].eq(20)
        & topk_summary["flag"].isin(ext_flags)
    ].copy()
    mat = ext.pivot_table(index="flag", columns="validation_type", values="mean_enrichment", aggfunc="mean").reindex(ext_flags).reindex(columns=VALIDATION_ORDER)
    fig, ax = plt.subplots(figsize=(7.8, 4.8))
    values = mat.values.astype(float)
    vmax = max(2.0, float(np.nanmax(values)) if np.isfinite(values).any() else 2.0)
    im = ax.imshow(np.nan_to_num(values, nan=0.0), cmap="YlGnBu", vmin=0, vmax=vmax)
    ax.set_xticks(np.arange(len(mat.columns)))
    ax.set_xticklabels([VALIDATION_LABELS[v] for v in mat.columns], rotation=12, ha="right")
    ax.set_yticks(np.arange(len(mat.index)))
    ax.set_yticklabels([v.replace("_", " ") for v in mat.index])
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            val = mat.iloc[i, j]
            ax.text(j, i, "NA" if pd.isna(val) else f"{val:.2f}x", ha="center", va="center", fontsize=8)
    ax.set_title("Rank-optimized HGNN Top20 enrichment")
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Top20 enrichment")
    fig.tight_layout()
    for suffix in ["png", "pdf", "svg"]:
        fig.savefig(FIG_DIR / f"Fig_RANK3_top20_enrichment_heatmap.{suffix}", dpi=360 if suffix == "png" else None, bbox_inches="tight")
    plt.close(fig)
    figures.append({"figure": "Fig_RANK3_top20_enrichment_heatmap", "png": str(FIG_DIR / "Fig_RANK3_top20_enrichment_heatmap.png")})

    attn_cols = ["path_attention_graph", "path_attention_axis_features", "path_attention_metapath_features"]
    attn = predictions.groupby("validation_type")[attn_cols].mean().reindex(VALIDATION_ORDER)
    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    bottom = np.zeros(len(attn))
    labels = ["graph", "axis features", "metapath interactions"]
    cols = ["#4F73C7", "#5BAE96", "#B93A3A"]
    for col, label, color in zip(attn_cols, labels, cols):
        vals = numeric(attn[col], 0.0).values
        ax.bar(np.arange(len(attn)), vals, bottom=bottom, color=color, alpha=0.88, label=label)
        bottom += vals
    ax.set_xticks(np.arange(len(attn)))
    ax.set_xticklabels([VALIDATION_LABELS[v] for v in attn.index], rotation=10, ha="right")
    ax.set_ylabel("Mean path attention weight")
    ax.set_ylim(0, 1.0)
    ax.set_title("Learned path-level fusion weights")
    ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.16), fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.18)
    fig.tight_layout()
    for suffix in ["png", "pdf", "svg"]:
        fig.savefig(FIG_DIR / f"Fig_RANK4_path_attention.{suffix}", dpi=360 if suffix == "png" else None, bbox_inches="tight")
    plt.close(fig)
    figures.append({"figure": "Fig_RANK4_path_attention", "png": str(FIG_DIR / "Fig_RANK4_path_attention.png")})

    last = history.sort_values("epoch").groupby(["validation_type", "fold_id", "seed"]).tail(1)
    fig, ax = plt.subplots(figsize=(7.8, 4.4))
    loss_cols = ["loss_cls", "loss_reg", "loss_pairwise", "loss_hard_negative", "loss_listwise", "loss_context"]
    means = last[loss_cols].apply(pd.to_numeric, errors="coerce").mean().sort_values(ascending=False)
    ax.bar(np.arange(len(means)), means.values, color="#6D737A", alpha=0.88)
    ax.set_xticks(np.arange(len(means)))
    ax.set_xticklabels([x.replace("loss_", "") for x in means.index], rotation=25, ha="right")
    ax.set_ylabel("Mean final loss component")
    ax.set_title("Rank-optimized training objective components")
    ax.grid(axis="y", alpha=0.22)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    for suffix in ["png", "pdf", "svg"]:
        fig.savefig(FIG_DIR / f"Fig_RANK5_loss_components.{suffix}", dpi=360 if suffix == "png" else None, bbox_inches="tight")
    plt.close(fig)
    figures.append({"figure": "Fig_RANK5_loss_components", "png": str(FIG_DIR / "Fig_RANK5_loss_components.png")})
    return figures


def write_report(
    summary: pd.DataFrame,
    deltas: pd.DataFrame,
    topk_delta: pd.DataFrame,
    figures: list[dict[str, str]],
    device_name: str,
    start_time: str,
    end_time: str,
) -> Path:
    report = PROJECT_ROOT / f"SPATIAL_CAUSAL_HGNN_RANK_OPTIMIZATION_REPORT_{RUN_DATE}.md"
    rank_summary = summary.loc[summary["model"].eq(PROPOSED_MODEL)].sort_values("validation_type")
    key_deltas = deltas.loc[
        deltas["validation_type"].eq("all_strict_splits")
        & deltas["reference_role"].isin(["current_hgnn", "best_standard_graph"])
    ]
    top20 = topk_delta.loc[
        topk_delta["k"].eq(20)
        & topk_delta["reference_role"].eq("current_hgnn")
        & topk_delta["flag"].isin(["publication_core", "clinical_risk", "spatial_recurrence", "lincs_reversal", "depmap_sensitivity"])
    ].copy()
    lines = [
        f"# Spatial-Causal-HGNN-Rank optimization report ({RUN_DATE})",
        "",
        "## Run Contract",
        "",
        f"- Device: `{device_name}`",
        f"- Seeds: `{';'.join(map(str, SEEDS))}`",
        f"- Started: `{start_time}`",
        f"- Finished: `{end_time}`",
        "- Model: `Spatial-Causal-HGNN-Rank`.",
        "- Objective: weighted classification + soft-score regression + pairwise ranking + decoy-guided hard-negative ranking + listwise ranking + cancer-context ranking.",
        "",
        "## Rank Model Mean +/- SD",
        "",
        rank_summary.to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Paired Delta",
        "",
        key_deltas.to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Top20 Delta Versus Current HGNN",
        "",
        top20.sort_values(["flag", "validation_type"]).to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Interpretation Guardrails",
        "",
        "- This upgrade optimizes mechanism-axis ranking, not direct clinical efficacy.",
        "- The ranking objective intentionally targets TopK prioritization; accuracy is not the primary endpoint.",
        "- Hard negatives are decoy-guided observed-axis contrasts, not wet-lab negative labels.",
        "- If a validation split shows smaller gains, report it transparently and prioritize TopK/external enrichment interpretation.",
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
    decoys = pd.read_csv(DECOY_TABLE) if DECOY_TABLE.exists() else pd.DataFrame()
    axis_features, meta_features, feature_cols = make_axis_feature_tensors(labels)
    split_configs = build_split_configs(labels)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"

    metrics_all: list[pd.DataFrame] = []
    history_all: list[pd.DataFrame] = []
    predictions_all: list[pd.DataFrame] = []
    hard_audits: list[pd.DataFrame] = []
    for seed in SEEDS:
        for _, split_row in split_configs.iterrows():
            metrics, history, predictions, hard_audit = train_one_rank_model(
                raw,
                labels,
                decoys,
                axis_features,
                meta_features,
                split_row,
                device,
                seed,
            )
            metrics_all.append(metrics)
            history_all.append(history)
            predictions_all.append(predictions)
            hard_audits.append(hard_audit)

    metrics = pd.concat(metrics_all, ignore_index=True)
    history = pd.concat(history_all, ignore_index=True)
    predictions = pd.concat(predictions_all, ignore_index=True)
    hard_audit = pd.concat(hard_audits, ignore_index=True)
    rank_summary = summarize_multiseed(metrics)

    existing_metrics = pd.read_csv(EXISTING_METRICS)
    existing_metrics = existing_metrics.loc[existing_metrics["model"].isin([REFERENCE_MODEL] + BASELINE_MODELS)].copy()
    combined_metrics = pd.concat([existing_metrics, metrics], ignore_index=True, sort=False)
    combined_summary = summarize_multiseed(combined_metrics)
    deltas = paired_delta_table(combined_metrics)

    existing_predictions = pd.read_csv(EXISTING_PREDICTIONS)
    existing_predictions = existing_predictions.loc[existing_predictions["model"].isin([REFERENCE_MODEL] + BASELINE_MODELS)].copy()
    combined_predictions = pd.concat([existing_predictions, predictions], ignore_index=True, sort=False)
    topk_long = topk_metrics(combined_predictions, [
        "flag_publication_core",
        "flag_lincs_reversal",
        "flag_depmap_sensitivity",
        "flag_clinical_risk",
        "flag_spatial_recurrence",
        "flag_non_broad_opportunity",
    ])
    topk_summary = summarize_topk(topk_long)
    topk_delta = topk_delta_table(topk_summary)

    design = pd.DataFrame(
        [
            {"component": "model_name", "value": PROPOSED_MODEL},
            {"component": "base_graph", "value": "HGT-like relation-aware message passing"},
            {"component": "axis_feature_count", "value": str(len(feature_cols))},
            {"component": "interaction_features", "value": ";".join(INTERACTION_FEATURES)},
            {"component": "loss", "value": "CE + 0.25*reg + 0.55*pairwise + 0.35*hard_negative + 0.20*listwise + 0.12*context"},
            {"component": "hard_negative_source", "value": "observed-axis contrasts guided by HGNN decoy table"},
            {"component": "primary_metric", "value": "strict split score_spearman and TopK enrichment"},
        ]
    )

    write_table(metrics, "Table_RANK1_metrics_by_split_seed")
    write_table(rank_summary, "Table_RANK2_rank_model_multiseed_summary")
    write_table(combined_summary, "Table_RANK3_combined_summary_vs_existing")
    write_table(deltas, "Table_RANK4_paired_delta_vs_existing")
    write_table(predictions, "Table_RANK5_test_predictions")
    write_table(history, "Table_RANK6_training_history")
    write_table(hard_audit, "Table_RANK7_decoy_guided_hard_negative_audit")
    write_table(topk_long, "Table_RANK8_topk_metrics_long")
    write_table(topk_summary, "Table_RANK9_topk_summary")
    write_table(topk_delta, "Table_RANK10_topk_delta_vs_existing")
    write_table(design, "Table_RANK11_model_design_contract")

    figures = make_figures(combined_summary, deltas, topk_summary, predictions, history)
    end_time = datetime.now().isoformat(timespec="seconds")
    report = write_report(combined_summary, deltas, topk_delta, figures, device_name, start_time, end_time)
    manifest = {
        "run_date": RUN_DATE,
        "created_at": end_time,
        "run_id": RUN_ID,
        "device": device_name,
        "seeds": SEEDS,
        "dataset_path": str(DATASET_PATH),
        "label_table": str(LABEL_TABLE),
        "decoy_table": str(DECOY_TABLE),
        "n_split_configs": int(len(split_configs)),
        "n_metric_rows": int(len(metrics)),
        "n_prediction_rows": int(len(predictions)),
        "tables": str(TABLE_DIR),
        "figures": str(FIG_DIR),
        "report": str(report),
    }
    manifest_path = METADATA_DIR / f"run_manifest_spatial_causal_hgnn_rank_optimization_{RUN_DATE}.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
