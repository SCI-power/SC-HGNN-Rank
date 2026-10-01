from __future__ import annotations

import torch
from torch import nn
from train_spatial_causal_hgnn_model_comparison import HGTLikeLayer


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
