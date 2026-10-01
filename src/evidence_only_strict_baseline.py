from __future__ import annotations

from package_paths import DATA, INPUTS, ARCHIVE, OUTPUT, CODE

import argparse
import json
import sys
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from scipy.stats import wilcoxon
from torch import nn


OLD_RUN_ROOT = ARCHIVE
OLD_CODE_DIR = CODE
if str(OLD_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(OLD_CODE_DIR))

import reviewer_matched_rerun as core  # noqa: E402


RUN_ROOT = OUTPUT / "evidence_only"
TABLE_DIR = RUN_ROOT / "tables"
MODEL_DIR = RUN_ROOT / "models"
LOG_DIR = RUN_ROOT / "logs"

VARIANTS = ["EvidenceOnly-112-rank", "EvidenceOnly-PM-rank"]
REFERENCE_VARIANT = "SC-HGNN-Rank-full"


class EvidenceOnlyRank(nn.Module):
    """Axis-evidence ranker with the SC-HGNN ranking heads but no graph message passing."""

    def __init__(self, axis_dim: int, meta_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
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
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2),
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

    def forward(self, axis_features: torch.Tensor, meta_features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        axis_h = self.axis_encoder(axis_features)
        meta_h = self.meta_encoder(meta_features)
        attention = torch.softmax(self.path_attention(torch.cat([axis_h, meta_h], dim=1)), dim=1)
        fused = attention[:, 0:1] * axis_h + attention[:, 1:2] * meta_h
        fused = self.fusion(fused)
        logits = self.classifier(fused)
        score = torch.sigmoid(self.regressor(fused)).squeeze(-1)
        return logits, score


def parameter_count(model: nn.Module) -> int:
    return int(sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad))


def choose_parameter_matched_width(raw: dict[str, Any], axis_dim: int, meta_dim: int, cfg: core.TrainConfig) -> tuple[int, int, int]:
    reference = core.RankOptimizedSpatialCausalHGNN(
        int(raw["metadata"]["num_node_types"]),
        int(raw["metadata"]["num_edge_types"]),
        axis_dim,
        meta_dim,
        hidden_dim=cfg.hidden_dim,
        num_layers=cfg.num_layers,
        dropout=cfg.dropout,
    )
    reference_n = parameter_count(reference)
    candidates = []
    for width in range(96, 385, 2):
        model = EvidenceOnlyRank(axis_dim, meta_dim, width, cfg.dropout)
        candidates.append((abs(parameter_count(model) - reference_n), width, parameter_count(model)))
    _, width, matched_n = min(candidates)
    return width, matched_n, reference_n


def bootstrap_ci(delta: np.ndarray, seed: int, n_boot: int = 5000) -> tuple[float, float]:
    if not len(delta):
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    sampled = rng.choice(delta, size=(n_boot, len(delta)), replace=True).mean(axis=1)
    low, high = np.percentile(sampled, [2.5, 97.5])
    return float(low), float(high)


def paired_tests(metrics: pd.DataFrame) -> pd.DataFrame:
    keys = ["seed", "validation_type", "fold_id"]
    reference = metrics[metrics["variant"].eq(REFERENCE_VARIANT)]
    rows: list[dict[str, Any]] = []
    for comparator in VARIANTS:
        merged = reference[keys + ["spearman", "ndcg20", "mae"]].merge(
            metrics[metrics["variant"].eq(comparator)][keys + ["spearman", "ndcg20", "mae"]],
            on=keys,
            suffixes=("_graph", "_evidence"),
            validate="one_to_one",
        )
        for split_family in ["all", "leave_one_cancer", "leave_compound_group", "leave_target_group"]:
            frame = merged if split_family == "all" else merged[merged["validation_type"].eq(split_family)]
            for metric in ["spearman", "ndcg20", "mae"]:
                graph = pd.to_numeric(frame[f"{metric}_graph"], errors="coerce")
                evidence = pd.to_numeric(frame[f"{metric}_evidence"], errors="coerce")
                delta = (evidence - graph if metric == "mae" else graph - evidence).dropna().to_numpy(dtype=float)
                if len(delta) and np.any(np.abs(delta) > 1e-12):
                    p_value = float(wilcoxon(delta, alternative="two-sided", zero_method="wilcox").pvalue)
                else:
                    p_value = float("nan")
                low, high = bootstrap_ci(delta, seed=20260807 + len(rows))
                rows.append(
                    {
                        "graph_model": REFERENCE_VARIANT,
                        "evidence_only_model": comparator,
                        "validation_type": split_family,
                        "metric": metric,
                        "positive_delta_favors_graph": True,
                        "n_pairs": len(delta),
                        "mean_delta": float(np.mean(delta)) if len(delta) else np.nan,
                        "median_delta": float(np.median(delta)) if len(delta) else np.nan,
                        "bootstrap_ci95_low": low,
                        "bootstrap_ci95_high": high,
                        "wilcoxon_p_two_sided": p_value,
                    }
                )
    return pd.DataFrame(rows)


def save(frame: pd.DataFrame, name: str) -> None:
    frame.to_csv(TABLE_DIR / f"{name}.csv", index=False, encoding="utf-8-sig")
    try:
        frame.to_excel(TABLE_DIR / f"{name}.xlsx", index=False)
    except Exception:
        pass




