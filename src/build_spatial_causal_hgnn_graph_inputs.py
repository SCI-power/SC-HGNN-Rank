from __future__ import annotations

from package_paths import DATA, INPUTS, ARCHIVE, OUTPUT, CODE

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = DATA / "project"
RUN_DATE = "2026-05-04"

TABLE_DIR = OUTPUT / "preprocess/graph"
FIG_DIR = OUTPUT / "preprocess/figures"
METADATA_DIR = OUTPUT / "preprocess/metadata"

BASE_AXIS_TABLE = PROJECT_ROOT / "tables" / "integrated_axis_scoring_with_ppi_msi" / "Table_IS1_integrated_top200_axis_scores.csv"
LINCS_TABLE = PROJECT_ROOT / "tables" / "lincs_cmap_combined_validation" / "Table_LINCS_COMBINED1_axis_level_best_release_support.csv"
DEPMAP_AXIS_TABLE = PROJECT_ROOT / "tables" / "depmap_dependency_validation" / "Table_DEPMAP1_axis_level_dependency_support.csv"
DEPMAP_GENE_TABLE = PROJECT_ROOT / "tables" / "depmap_dependency_validation" / "Table_DEPMAP2_axis_gene_role_dependency_support.csv"
NON_BROAD_TABLE = PROJECT_ROOT / "tables" / "non_broad_target_discovery" / "Table_NONBROAD1_axis_level_best_non_broad_candidates.csv"
SPATIAL_SAMPLE_TABLE = PROJECT_ROOT / "tables" / "candidate_axis_convergence_validation" / "Table_CV4_main_top50_sample_spatial_recurrence.csv"
GRAPH_SCHEMA_TABLE = PROJECT_ROOT / "tables" / "full_design_execution_control" / "Table_FD3_hgnn_graph_schema.csv"


def ensure_dirs() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    METADATA_DIR.mkdir(parents=True, exist_ok=True)


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def text_value(value: Any) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def num_value(value: Any, default: float = 0.0) -> float:
    val = pd.to_numeric(value, errors="coerce")
    if pd.isna(val):
        return default
    return float(np.clip(val, 0.0, 1.0))


def safe_native_id(*parts: Any) -> str:
    cleaned = [text_value(p).replace("|", "_").replace(" ", "_") for p in parts if text_value(p)]
    return "::".join(cleaned) if cleaned else "unknown"


def node_id(node_type: str, native_id: str) -> str:
    return f"{node_type}::{native_id}"


def merge_unique_columns(base: pd.DataFrame, extra: pd.DataFrame, key: str, columns: list[str]) -> pd.DataFrame:
    if extra.empty:
        return base
    keep = [key] + [c for c in columns if c in extra.columns and c not in base.columns]
    if len(keep) <= 1:
        return base
    reduced = extra[keep].drop_duplicates(key)
    return base.merge(reduced, on=key, how="left")


def join_unique(values: pd.Series) -> str:
    cleaned = [text_value(v) for v in values if text_value(v)]
    if not cleaned:
        return ""
    return ";".join(sorted(set(cleaned)))


def deduplicate_edges(edge_df: pd.DataFrame) -> pd.DataFrame:
    if edge_df.empty:
        return edge_df
    keys = ["source_node_id", "target_node_id", "edge_type", "cancer", "frozen_axis_id", "candidate_axis_id"]
    agg: dict[str, Any] = {}
    for col in edge_df.columns:
        if col in keys or col == "edge_id":
            continue
        if col in {"weight", "evidence_score"}:
            agg[col] = "max"
        else:
            agg[col] = join_unique
    out = edge_df.groupby(keys, dropna=False, as_index=False).agg(agg)
    out.insert(0, "edge_id", [f"E{i + 1:08d}" for i in range(len(out))])
    ordered_cols = ["edge_id"] + [c for c in edge_df.columns if c != "edge_id" and c in out.columns]
    return out[ordered_cols].sort_values(["edge_type", "cancer", "source_node_id", "target_node_id"]).reset_index(drop=True)


def enrich_axis_table(axis: pd.DataFrame) -> pd.DataFrame:
    lincs = read_csv(LINCS_TABLE)
    depmap_axis = read_csv(DEPMAP_AXIS_TABLE)
    non_broad = read_csv(NON_BROAD_TABLE)

    axis = merge_unique_columns(
        axis,
        lincs,
        "frozen_axis_id",
        [
            "lincs_cmap_combined_support_score",
            "lincs_cmap_combined_support_tier",
            "lincs_best_release",
            "lincs_axis_support_status",
            "lincs_release_consistency",
            "lincs_best_pert_id",
            "lincs_best_pert_iname",
            "lincs_best_n_signatures",
        ],
    )
    axis = merge_unique_columns(
        axis,
        depmap_axis,
        "frozen_axis_id",
        [
            "depmap_axis_support_status",
            "depmap_axis_support_score",
            "depmap_axis_support_tier",
            "depmap_best_gene_symbol",
            "depmap_best_gene_role",
            "depmap_best_broad_dependency_flag",
            "depmap_best_effect_strength_score",
            "depmap_best_dependency_probability_score",
            "depmap_best_cellline_consistency_score",
            "depmap_best_expression_gate_score",
            "depmap_best_selectivity_score",
        ],
    )
    axis = merge_unique_columns(
        axis,
        non_broad,
        "frozen_axis_id",
        [
            "best_non_broad_gene_symbol",
            "best_non_broad_gene_role",
            "best_non_broad_gene_score",
            "non_broad_cancer_sensitivity_score",
            "underexplored_opportunity_score",
            "under_validation_gap_score",
            "candidate_discovery_class",
            "existing_evidence_gap_class",
            "axis_n_non_broad_tested_genes",
            "axis_n_nb1_nb2_non_broad_genes",
        ],
    )
    return axis


class GraphBuilder:
    def __init__(self) -> None:
        self.nodes: dict[str, dict[str, Any]] = {}
        self.edges: list[dict[str, Any]] = []

    def add_node(
        self,
        node_type: str,
        native_id: str,
        label: str,
        cancer: str = "",
        source_layer: str = "",
        feature_score: float | None = None,
        feature_tier: str = "",
    ) -> str:
        nid = node_id(node_type, native_id)
        if nid not in self.nodes:
            self.nodes[nid] = {
                "node_id": nid,
                "node_type": node_type,
                "native_id": native_id,
                "label": label,
                "cancer": cancer,
                "source_layer": source_layer,
                "feature_score": feature_score if feature_score is not None else np.nan,
                "feature_tier": feature_tier,
            }
        else:
            current = self.nodes[nid]
            if not current.get("cancer") and cancer:
                current["cancer"] = cancer
            if not current.get("source_layer") and source_layer:
                current["source_layer"] = source_layer
            if pd.isna(current.get("feature_score")) and feature_score is not None:
                current["feature_score"] = feature_score
            if not current.get("feature_tier") and feature_tier:
                current["feature_tier"] = feature_tier
        return nid

    def add_edge(
        self,
        source_node_id: str,
        target_node_id: str,
        edge_type: str,
        cancer: str,
        frozen_axis_id: str = "",
        candidate_axis_id: str = "",
        weight: float = 1.0,
        evidence_score: float | None = None,
        source_layer: str = "",
        feature_name: str = "",
        evidence_tier: str = "",
        edge_role: str = "model_feature",
        extra: dict[str, Any] | None = None,
    ) -> None:
        row = {
            "edge_id": f"E{len(self.edges) + 1:08d}",
            "source_node_id": source_node_id,
            "target_node_id": target_node_id,
            "edge_type": edge_type,
            "cancer": cancer,
            "frozen_axis_id": frozen_axis_id,
            "candidate_axis_id": candidate_axis_id,
            "weight": num_value(weight, default=1.0),
            "evidence_score": np.nan if evidence_score is None else num_value(evidence_score),
            "source_layer": source_layer,
            "feature_name": feature_name,
            "evidence_tier": evidence_tier,
            "edge_role": edge_role,
        }
        if extra:
            row.update(extra)
        self.edges.append(row)


def build_graph(axis: pd.DataFrame, spatial_samples: pd.DataFrame, depmap_gene: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    graph = GraphBuilder()

    for row in axis.itertuples(index=False):
        cancer = text_value(getattr(row, "cancer", ""))
        frozen_axis_id = text_value(getattr(row, "frozen_axis_id", ""))
        candidate_axis_id = text_value(getattr(row, "candidate_axis_id", ""))
        compound_id = text_value(getattr(row, "compound_master_id", ""))
        compound_name = text_value(getattr(row, "compound_representative_name", compound_id))
        target_gene = text_value(getattr(row, "target_gene_symbol", ""))
        tf_gene = text_value(getattr(row, "tf_gene_symbol", ""))
        regulon_target = text_value(getattr(row, "regulon_target_gene_symbol", target_gene))
        celltype = text_value(getattr(row, "dominant_spatial_celltype", "unknown_cell_state"))
        niche = text_value(getattr(row, "niche_class", celltype)) or celltype
        axis_label = text_value(getattr(row, "axis_display_label", "")) or f"{compound_name} | {target_gene}/{tf_gene}"

        axis_node = graph.add_node(
            "axis",
            frozen_axis_id,
            axis_label,
            cancer,
            "integrated_axis",
            num_value(getattr(row, "integrated_axis_score_with_ppi_msi", np.nan), default=np.nan),
            text_value(getattr(row, "integrated_evidence_tier", "")),
        )
        compound_node = graph.add_node("compound", compound_id, compound_name, "", "tcm_drug_evidence")
        target_node = graph.add_node("gene", target_gene, target_gene, "", "target_gene")
        tf_node = graph.add_node("tf_regulon", tf_gene, tf_gene, "", "single_cell_grn")
        regulon_target_node = graph.add_node("gene", regulon_target, regulon_target, "", "single_cell_grn")
        cell_node = graph.add_node("cell_state", safe_native_id(cancer, celltype), f"{cancer} {celltype}", cancer, "single_cell_annotation")
        niche_node = graph.add_node("spatial_niche", safe_native_id(cancer, niche), f"{cancer} {niche}", cancer, "spatial_transcriptomics")
        cancer_node = graph.add_node("cancer_context", cancer, cancer, cancer, "study_context")
        clinical_node = graph.add_node("clinical_endpoint", safe_native_id(cancer, "survival"), f"{cancer} survival", cancer, "clinical_prognosis")
        ppi_node = graph.add_node("ppi_msi_module", safe_native_id(cancer, "candidate_module"), f"{cancer} PPI/MSI module", cancer, "ppi_msi")
        depmap_node = graph.add_node("depmap_context", safe_native_id(cancer), f"{cancer} DepMap context", cancer, "depmap")

        drug_score = num_value(getattr(row, "drug_evidence_score", np.nan), default=0.0)
        scrna_score = num_value(getattr(row, "single_cell_regulatory_score", np.nan), default=0.0)
        spatial_score = num_value(getattr(row, "integrated_spatial_recurrence_score", getattr(row, "exact_spatial_recurrence_score", np.nan)), default=0.0)
        deconv_score = num_value(getattr(row, "rctd_spotlight_consistency_score", np.nan), default=0.0)
        clinical_score = num_value(getattr(row, "same_cancer_clinical_prognosis_score", np.nan), default=0.0)
        ppi_score = num_value(getattr(row, "ppi_msi_local_support_score", np.nan), default=0.0)
        context_score = num_value(getattr(row, "cancer_context_specificity_score", np.nan), default=0.0)

        graph.add_edge(axis_node, compound_node, "axis_has_compound", cancer, frozen_axis_id, candidate_axis_id, 1.0, 1.0, "axis_composition")
        graph.add_edge(axis_node, target_node, "axis_has_target_gene", cancer, frozen_axis_id, candidate_axis_id, drug_score, drug_score, "axis_composition", "drug_evidence_score")
        graph.add_edge(axis_node, tf_node, "axis_has_tf_regulon", cancer, frozen_axis_id, candidate_axis_id, scrna_score, scrna_score, "single_cell_grn", "single_cell_regulatory_score")
        graph.add_edge(compound_node, target_node, "compound_acts_on_target_gene", cancer, frozen_axis_id, candidate_axis_id, drug_score, drug_score, "drug_target_evidence", "drug_evidence_score", text_value(getattr(row, "drug_evidence_class", "")))
        graph.add_edge(tf_node, regulon_target_node, "tf_regulates_regulon_target_gene", cancer, frozen_axis_id, candidate_axis_id, scrna_score, scrna_score, "single_cell_grn", "single_cell_regulatory_score")
        graph.add_edge(target_node, cell_node, "target_gene_linked_to_cell_state", cancer, frozen_axis_id, candidate_axis_id, scrna_score, scrna_score, "single_cell_annotation_grn", "single_cell_regulatory_score")
        graph.add_edge(tf_node, cell_node, "tf_regulon_active_in_cell_state", cancer, frozen_axis_id, candidate_axis_id, scrna_score, scrna_score, "single_cell_grn", "single_cell_regulatory_score")
        graph.add_edge(cell_node, niche_node, "cell_state_maps_to_spatial_niche", cancer, frozen_axis_id, candidate_axis_id, deconv_score, deconv_score, "rctd_spotlight", "rctd_spotlight_consistency_score", text_value(getattr(row, "rctd_spotlight_consistency_class", "")))
        graph.add_edge(target_node, niche_node, "target_gene_projects_to_spatial_niche", cancer, frozen_axis_id, candidate_axis_id, spatial_score, spatial_score, "tangram_spatial_projection", "integrated_spatial_recurrence_score")
        graph.add_edge(axis_node, clinical_node, "axis_associated_with_clinical_endpoint", cancer, frozen_axis_id, candidate_axis_id, clinical_score, clinical_score, "clinical_prognosis", "same_cancer_clinical_prognosis_score", text_value(getattr(row, "clinical_prognosis_support_class", "")))
        graph.add_edge(target_node, ppi_node, "target_gene_supported_by_ppi_msi_module", cancer, frozen_axis_id, candidate_axis_id, ppi_score, ppi_score, "ppi_msi", "ppi_msi_local_support_score", text_value(getattr(row, "ppi_msi_local_support_tier", "")))
        graph.add_edge(axis_node, ppi_node, "axis_supported_by_ppi_msi_module", cancer, frozen_axis_id, candidate_axis_id, ppi_score, ppi_score, "ppi_msi", "ppi_msi_local_support_score", text_value(getattr(row, "ppi_msi_local_support_tier", "")))
        graph.add_edge(axis_node, cancer_node, "axis_observed_in_cancer_context", cancer, frozen_axis_id, candidate_axis_id, context_score, context_score, "cancer_context", "cancer_context_specificity_score")

        lincs_score = num_value(getattr(row, "lincs_cmap_combined_support_score", np.nan), default=np.nan)
        pert_id = text_value(getattr(row, "lincs_best_pert_id", ""))
        if pert_id and not pd.isna(lincs_score):
            pert_name = text_value(getattr(row, "lincs_best_pert_iname", "")) or pert_id
            lincs_node = graph.add_node("lincs_signature", pert_id, pert_name, "", "lincs_cmap", lincs_score, text_value(getattr(row, "lincs_cmap_combined_support_tier", "")))
            graph.add_edge(compound_node, lincs_node, "compound_reverses_lincs_signature", cancer, frozen_axis_id, candidate_axis_id, lincs_score, lincs_score, "lincs_cmap", "lincs_cmap_combined_support_score", text_value(getattr(row, "lincs_cmap_combined_support_tier", "")))
            graph.add_edge(axis_node, lincs_node, "axis_supported_by_lincs_reversal", cancer, frozen_axis_id, candidate_axis_id, lincs_score, lincs_score, "lincs_cmap", "lincs_cmap_combined_support_score", text_value(getattr(row, "lincs_cmap_combined_support_tier", "")))

        depmap_axis_score = num_value(getattr(row, "depmap_axis_support_score", np.nan), default=np.nan)
        if not pd.isna(depmap_axis_score):
            graph.add_edge(axis_node, depmap_node, "axis_supported_by_depmap_dependency", cancer, frozen_axis_id, candidate_axis_id, depmap_axis_score, depmap_axis_score, "depmap", "depmap_axis_support_score", text_value(getattr(row, "depmap_axis_support_tier", "")))

        nb_gene = text_value(getattr(row, "best_non_broad_gene_symbol", ""))
        nb_score = num_value(getattr(row, "non_broad_cancer_sensitivity_score", np.nan), default=np.nan)
        if nb_gene and not pd.isna(nb_score):
            nb_gene_node = graph.add_node("gene", nb_gene, nb_gene, "", "non_broad_depmap_extension", nb_score, text_value(getattr(row, "candidate_discovery_class", "")))
            graph.add_edge(axis_node, nb_gene_node, "axis_has_non_broad_candidate_gene", cancer, frozen_axis_id, candidate_axis_id, nb_score, nb_score, "non_broad_target_discovery", "non_broad_cancer_sensitivity_score", text_value(getattr(row, "candidate_discovery_class", "")), edge_role="supplementary_extension")

    if not depmap_gene.empty:
        for row in depmap_gene.itertuples(index=False):
            cancer = text_value(getattr(row, "cancer", ""))
            gene = text_value(getattr(row, "depmap_mapped_gene_symbol", "")) or text_value(getattr(row, "requested_gene_symbol", ""))
            if not gene:
                continue
            gene_node = graph.add_node("gene", gene, gene, "", "depmap")
            depmap_node = graph.add_node("depmap_context", safe_native_id(cancer), f"{cancer} DepMap context", cancer, "depmap")
            graph.add_edge(
                gene_node,
                depmap_node,
                "gene_essential_in_depmap_context",
                cancer,
                text_value(getattr(row, "frozen_axis_id", "")),
                text_value(getattr(row, "candidate_axis_id", "")),
                num_value(getattr(row, "depmap_gene_support_score", np.nan), default=0.0),
                num_value(getattr(row, "depmap_gene_support_score", np.nan), default=0.0),
                "depmap",
                "depmap_gene_support_score",
                text_value(getattr(row, "depmap_gene_support_tier", "")),
                extra={
                    "gene_role": text_value(getattr(row, "gene_role", "")),
                    "broad_dependency_flag": text_value(getattr(row, "broad_dependency_flag", "")),
                },
            )

    if not spatial_samples.empty:
        axis_map = axis[["candidate_axis_id", "frozen_axis_id"]].drop_duplicates()
        sample_df = spatial_samples.merge(axis_map, on="candidate_axis_id", how="left", suffixes=("", "_axis"))
        sample_df = sample_df.loc[sample_df["frozen_axis_id"].notna()].copy()
        for row in sample_df.itertuples(index=False):
            cancer = text_value(getattr(row, "cancer", ""))
            frozen_axis_id = text_value(getattr(row, "frozen_axis_id", ""))
            candidate_axis_id = text_value(getattr(row, "candidate_axis_id", ""))
            sample_id = text_value(getattr(row, "sample_id", ""))
            celltype = text_value(getattr(row, "dominant_spatial_celltype", "unknown_cell_state"))
            axis_node = node_id("axis", frozen_axis_id)
            sample_node = graph.add_node("spatial_sample", sample_id, text_value(getattr(row, "sample_label", sample_id)), cancer, "spatial_transcriptomics")
            niche_node = graph.add_node("spatial_niche", safe_native_id(cancer, celltype), f"{cancer} {celltype}", cancer, "spatial_transcriptomics")
            fixed_abs = num_value(getattr(row, "fixed_celltype_abs_spearman", np.nan), default=0.0)
            deconv = num_value(getattr(row, "deconv_spearman", np.nan), default=0.0)
            graph.add_edge(axis_node, sample_node, "axis_spatially_recurrent_in_sample", cancer, frozen_axis_id, candidate_axis_id, fixed_abs, fixed_abs, "tangram_spatial_projection", "fixed_celltype_abs_spearman", edge_role="spatial_validation")
            graph.add_edge(sample_node, niche_node, "sample_contains_mapped_spatial_niche", cancer, frozen_axis_id, candidate_axis_id, deconv, deconv, "rctd_spotlight", "deconv_spearman", text_value(getattr(row, "deconv_stable", "")), edge_role="spatial_validation")

    node_df = pd.DataFrame(graph.nodes.values()).sort_values(["node_type", "node_id"]).reset_index(drop=True)
    edge_df = deduplicate_edges(pd.DataFrame(graph.edges))
    return node_df, edge_df


def build_axis_labels(axis: pd.DataFrame) -> pd.DataFrame:
    label_map = {
        "positive_high_confidence": 2,
        "positive_moderate_confidence": 1,
        "reserve_or_unlabeled": 0,
    }
    label = axis.copy()
    label["hgnn_label_ordinal"] = label["hgnn_integrated_label"].map(label_map).fillna(0).astype(int)
    label["hgnn_binary_positive"] = np.where(label["hgnn_integrated_label"].str.startswith("positive", na=False), 1, np.nan)
    label["soft_rank_label"] = pd.to_numeric(label.get("integrated_axis_score_with_ppi_msi"), errors="coerce")
    keep = [
        "frozen_axis_id",
        "candidate_axis_id",
        "cancer",
        "compound_master_id",
        "compound_representative_name",
        "target_gene_symbol",
        "tf_gene_symbol",
        "regulon_target_gene_symbol",
        "dominant_spatial_celltype",
        "integrated_rank_within_cancer",
        "integrated_axis_score_with_ppi_msi",
        "hgnn_integrated_label",
        "hgnn_label_ordinal",
        "hgnn_binary_positive",
        "soft_rank_label",
        "integrated_evidence_tier",
        "publication_use_role",
        "same_cancer_clinical_prognosis_score",
        "integrated_spatial_recurrence_score",
        "rctd_spotlight_consistency_score",
        "drug_evidence_score",
        "single_cell_regulatory_score",
        "ppi_msi_local_support_score",
        "lincs_cmap_combined_support_score",
        "depmap_axis_support_score",
        "non_broad_cancer_sensitivity_score",
        "cancer_context_specificity_score",
    ]
    return label[[c for c in keep if c in label.columns]].sort_values(["cancer", "integrated_rank_within_cancer"])


def build_negative_decoys(axis: pd.DataFrame, seed: int = 20260504) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    positives = axis.loc[axis["hgnn_integrated_label"].str.startswith("positive", na=False)].copy()
    existing = set(
        zip(
            axis["cancer"].astype(str),
            axis["compound_master_id"].astype(str),
            axis["target_gene_symbol"].astype(str),
            axis["tf_gene_symbol"].astype(str),
            axis["dominant_spatial_celltype"].astype(str),
        )
    )
    rows: list[dict[str, Any]] = []

    for row in positives.itertuples(index=False):
        cancer = text_value(getattr(row, "cancer", ""))
        pool = axis.loc[axis["cancer"].eq(cancer)].copy()
        if len(pool) < 2:
            continue

        for reason in ["same_compound_wrong_target", "same_target_wrong_cell_state", "wrong_compound_same_context"]:
            for _ in range(25):
                donor = pool.iloc[int(rng.integers(0, len(pool)))]
                compound_id = text_value(getattr(row, "compound_master_id", ""))
                compound_name = text_value(getattr(row, "compound_representative_name", ""))
                target = text_value(getattr(row, "target_gene_symbol", ""))
                tf = text_value(getattr(row, "tf_gene_symbol", ""))
                celltype = text_value(getattr(row, "dominant_spatial_celltype", ""))

                if reason == "same_compound_wrong_target":
                    target = text_value(donor.get("target_gene_symbol", ""))
                    tf = text_value(donor.get("tf_gene_symbol", ""))
                elif reason == "same_target_wrong_cell_state":
                    celltype = text_value(donor.get("dominant_spatial_celltype", ""))
                elif reason == "wrong_compound_same_context":
                    compound_id = text_value(donor.get("compound_master_id", ""))
                    compound_name = text_value(donor.get("compound_representative_name", ""))

                key = (cancer, compound_id, target, tf, celltype)
                if key in existing:
                    continue

                rows.append(
                    {
                        "decoy_axis_id": f"NEG_{len(rows) + 1:06d}",
                        "source_positive_axis_id": text_value(getattr(row, "frozen_axis_id", "")),
                        "cancer": cancer,
                        "compound_master_id": compound_id,
                        "compound_representative_name": compound_name,
                        "target_gene_symbol": target,
                        "tf_gene_symbol": tf,
                        "dominant_spatial_celltype": celltype,
                        "negative_reason": reason,
                        "hgnn_integrated_label": "negative_decoy_candidate",
                        "hgnn_label_ordinal": 0,
                        "hgnn_binary_positive": 0,
                        "sampling_note": "Degree-aware replacement should be refined before final model training; this table is the first hard-negative candidate pool.",
                    }
                )
                existing.add(key)
                break

    return pd.DataFrame(rows)


def build_missingness(axis: pd.DataFrame) -> pd.DataFrame:
    feature_cols = [
        "same_cancer_clinical_prognosis_score",
        "integrated_spatial_recurrence_score",
        "rctd_spotlight_consistency_score",
        "drug_evidence_score",
        "single_cell_regulatory_score",
        "ppi_msi_local_support_score",
        "cancer_context_specificity_score",
        "evidence_robustness_penalty_adjusted_score",
        "lincs_cmap_combined_support_score",
        "depmap_axis_support_score",
        "non_broad_cancer_sensitivity_score",
        "underexplored_opportunity_score",
    ]
    rows = []
    for col in feature_cols:
        if col not in axis.columns:
            rows.append({"feature": col, "status": "missing_column"})
            continue
        values = pd.to_numeric(axis[col], errors="coerce")
        rows.append(
            {
                "feature": col,
                "status": "available",
                "n_rows": len(axis),
                "n_missing": int(values.isna().sum()),
                "missing_fraction": float(values.isna().mean()),
                "n_zero": int(values.fillna(0).eq(0).sum()),
                "zero_fraction_including_missing_as_zero": float(values.fillna(0).eq(0).mean()),
                "min": float(values.min()) if values.notna().any() else np.nan,
                "median": float(values.median()) if values.notna().any() else np.nan,
                "mean": float(values.mean()) if values.notna().any() else np.nan,
                "max": float(values.max()) if values.notna().any() else np.nan,
            }
        )
    return pd.DataFrame(rows)


def build_summary(nodes: pd.DataFrame, edges: pd.DataFrame, labels: pd.DataFrame, decoys: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for node_type, count in nodes["node_type"].value_counts().sort_index().items():
        rows.append({"summary_section": "nodes_by_type", "item": node_type, "value": int(count)})
    for edge_type, count in edges["edge_type"].value_counts().sort_index().items():
        rows.append({"summary_section": "edges_by_type", "item": edge_type, "value": int(count)})
    for label, count in labels["hgnn_integrated_label"].value_counts().sort_index().items():
        rows.append({"summary_section": "axis_labels", "item": label, "value": int(count)})
    for cancer, count in labels["cancer"].value_counts().sort_index().items():
        rows.append({"summary_section": "axes_by_cancer", "item": cancer, "value": int(count)})
    if not decoys.empty:
        for reason, count in decoys["negative_reason"].value_counts().sort_index().items():
            rows.append({"summary_section": "negative_decoys_by_reason", "item": reason, "value": int(count)})
    rows.append({"summary_section": "graph_total", "item": "nodes", "value": int(len(nodes))})
    rows.append({"summary_section": "graph_total", "item": "edges", "value": int(len(edges))})
    rows.append({"summary_section": "graph_total", "item": "axis_labels", "value": int(len(labels))})
    rows.append({"summary_section": "graph_total", "item": "negative_decoys", "value": int(len(decoys))})
    return pd.DataFrame(rows)


def build_label_distribution(labels: pd.DataFrame, decoys: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (cancer, label), df in labels.groupby(["cancer", "hgnn_integrated_label"], dropna=False):
        rows.append(
            {
                "source": "observed_axis_pool",
                "cancer": cancer,
                "label": label,
                "n": int(len(df)),
                "mean_soft_rank_label": float(pd.to_numeric(df["soft_rank_label"], errors="coerce").mean()),
                "min_soft_rank_label": float(pd.to_numeric(df["soft_rank_label"], errors="coerce").min()),
                "max_soft_rank_label": float(pd.to_numeric(df["soft_rank_label"], errors="coerce").max()),
            }
        )
    if not decoys.empty:
        for (cancer, reason), df in decoys.groupby(["cancer", "negative_reason"], dropna=False):
            rows.append(
                {
                    "source": "negative_decoy_pool",
                    "cancer": cancer,
                    "label": reason,
                    "n": int(len(df)),
                    "mean_soft_rank_label": 0.0,
                    "min_soft_rank_label": 0.0,
                    "max_soft_rank_label": 0.0,
                }
            )
    return pd.DataFrame(rows).sort_values(["source", "cancer", "label"])


def build_integrity_audit(nodes: pd.DataFrame, edges: pd.DataFrame, labels: pd.DataFrame, decoys: pd.DataFrame) -> pd.DataFrame:
    node_ids = set(nodes["node_id"].astype(str))
    source_missing = ~edges["source_node_id"].astype(str).isin(node_ids)
    target_missing = ~edges["target_node_id"].astype(str).isin(node_ids)
    weight = pd.to_numeric(edges["weight"], errors="coerce")
    duplicate_nodes = nodes["node_id"].duplicated().sum()
    duplicate_edges = edges.duplicated(
        ["source_node_id", "target_node_id", "edge_type", "frozen_axis_id", "candidate_axis_id"]
    ).sum()
    rows = [
        {"check": "node_id_unique", "status": "pass" if duplicate_nodes == 0 else "fail", "value": int(duplicate_nodes), "note": "Number of duplicated node ids."},
        {"check": "edge_source_nodes_exist", "status": "pass" if int(source_missing.sum()) == 0 else "fail", "value": int(source_missing.sum()), "note": "Edges whose source node is absent from node table."},
        {"check": "edge_target_nodes_exist", "status": "pass" if int(target_missing.sum()) == 0 else "fail", "value": int(target_missing.sum()), "note": "Edges whose target node is absent from node table."},
        {"check": "edge_weight_not_missing", "status": "pass" if int(weight.isna().sum()) == 0 else "fail", "value": int(weight.isna().sum()), "note": "Missing edge weights."},
        {"check": "edge_weight_in_0_1", "status": "pass" if int(((weight < 0) | (weight > 1)).sum()) == 0 else "fail", "value": int(((weight < 0) | (weight > 1)).sum()), "note": "Weights outside [0,1]."},
        {"check": "duplicate_edge_records", "status": "warn" if int(duplicate_edges) > 0 else "pass", "value": int(duplicate_edges), "note": "Exact duplicate source-target-type-axis records; may be acceptable for multi-evidence edges but should be reviewed."},
        {"check": "axis_label_count", "status": "pass" if len(labels) == 600 else "warn", "value": int(len(labels)), "note": "Expected 600 Top200 axes across 3 cancers."},
        {"check": "negative_decoy_count", "status": "pass" if len(decoys) > 0 else "warn", "value": int(len(decoys)), "note": "Initial hard-negative candidate count before degree-aware refinement."},
    ]
    return pd.DataFrame(rows)


def build_feature_dictionary() -> pd.DataFrame:
    rows = [
        ("drug_evidence_score", "compound-target edge", "compound_acts_on_target_gene", "TCM edge score, BindingDB, PubMed, evidence tier"),
        ("single_cell_regulatory_score", "single-cell GRN edge", "target_gene_linked_to_cell_state; tf_regulon_active_in_cell_state", "scRNA regulon specificity and cell-state linkage"),
        ("rctd_spotlight_consistency_score", "cell-state spatial mapping edge", "cell_state_maps_to_spatial_niche", "RCTD and SPOTlight concordance"),
        ("integrated_spatial_recurrence_score", "target spatial projection edge", "target_gene_projects_to_spatial_niche", "Tangram and spatial recurrence support"),
        ("same_cancer_clinical_prognosis_score", "clinical endpoint edge", "axis_associated_with_clinical_endpoint", "same-cancer TCGA/GEO prognosis evidence"),
        ("ppi_msi_local_support_score", "PPI/MSI module edge", "axis_supported_by_ppi_msi_module; target_gene_supported_by_ppi_msi_module", "STRING, BioGRID, OmniPath local network support"),
        ("lincs_cmap_combined_support_score", "LINCS perturbation edge", "compound_reverses_lincs_signature", "best GSE70138/GSE92742 perturbation reversal"),
        ("depmap_axis_support_score", "DepMap context edge", "axis_supported_by_depmap_dependency", "functional dependency support with broad-essential caution"),
        ("non_broad_cancer_sensitivity_score", "supplementary non-broad edge", "axis_has_non_broad_candidate_gene", "non-broad cancer-sensitive extension"),
        ("cancer_context_specificity_score", "cancer context edge", "axis_observed_in_cancer_context", "cancer-specific over other cancer contexts"),
    ]
    return pd.DataFrame(rows, columns=["feature_name", "model_role", "edge_type", "definition"])


def write_table(df: pd.DataFrame, name: str) -> Path:
    path = TABLE_DIR / f"{name}.csv"
    df.to_csv(path, index=False, encoding="utf-8-sig")
    try:
        df.to_excel(TABLE_DIR / f"{name}.xlsx", index=False)
    except Exception:
        pass
    return path


def make_schema_figure(summary: pd.DataFrame) -> list[dict[str, str]]:
    import matplotlib.pyplot as plt

    figures = []
    node_counts = summary.loc[summary["summary_section"].eq("nodes_by_type")].copy()
    edge_counts = summary.loc[summary["summary_section"].eq("edges_by_type")].copy()

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2))
    for ax, data, title, color in [
        (axes[0], node_counts, "Node types in Spatial-Causal-HGNN graph", "#4C78A8"),
        (axes[1], edge_counts.head(18), "Top edge types in Spatial-Causal-HGNN graph", "#54A24B"),
    ]:
        data = data.sort_values("value", ascending=True)
        ax.barh(data["item"], data["value"], color=color, alpha=0.86)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Count")
        ax.grid(axis="x", alpha=0.25, linewidth=0.7)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(axis="y", labelsize=7)
    fig.tight_layout()
    for suffix in ["png", "pdf", "svg"]:
        path = FIG_DIR / f"Fig_HGNN_graph_input_summary.{suffix}"
        fig.savefig(path, dpi=360 if suffix == "png" else None, bbox_inches="tight")
    plt.close(fig)
    figures.append({"figure": "Fig_HGNN_graph_input_summary", "png": str(FIG_DIR / "Fig_HGNN_graph_input_summary.png")})
    return figures


def write_report(summary: pd.DataFrame, missing: pd.DataFrame, figures: list[dict[str, str]]) -> Path:
    report = OUTPUT / f"SPATIAL_CAUSAL_HGNN_GRAPH_INPUT_REPORT_{RUN_DATE}.md"
    lines = [
        f"# Spatial-Causal-HGNN graph input report ({RUN_DATE})",
        "",
        "## Purpose",
        "",
        "This run converts the current TCM, single-cell GRN, spatial transcriptomics, clinical, PPI/MSI, LINCS/CMap, DepMap, and non-broad target layers into graph-ready nodes, edges, labels, and quality-control tables.",
        "",
        "The output is model input preparation, not final model training.",
        "",
        "## Graph Summary",
        "",
        summary.to_markdown(index=False),
        "",
        "## Feature Missingness",
        "",
        missing.to_markdown(index=False, floatfmt=".3f"),
        "",
        "## Key Interpretation",
        "",
        "- Single-cell information is encoded as target/regulon-to-cell-state edges.",
        "- Spatial transcriptomics is encoded as cell-state-to-spatial-niche and target-to-spatial-niche edges.",
        "- Clinical, PPI/MSI, LINCS/CMap, and DepMap layers are encoded as validation/context edges.",
        "- Current weighted scores remain soft labels and baselines; the HGNN should be evaluated against these baselines and ablations.",
        "",
        "## Outputs",
        "",
        f"- Node table: `{TABLE_DIR / 'Table_HGNN1_node_table.csv'}`",
        f"- Edge table: `{TABLE_DIR / 'Table_HGNN2_edge_table.csv'}`",
        f"- Axis label table: `{TABLE_DIR / 'Table_HGNN3_axis_label_table.csv'}`",
        f"- Summary table: `{TABLE_DIR / 'Table_HGNN4_node_edge_summary.csv'}`",
        f"- Missingness table: `{TABLE_DIR / 'Table_HGNN5_feature_missingness.csv'}`",
        f"- Label distribution table: `{TABLE_DIR / 'Table_HGNN6_label_distribution.csv'}`",
        f"- Negative decoy candidate pool: `{TABLE_DIR / 'Table_HGNN7_negative_axis_decoy_pool.csv'}`",
        f"- Feature dictionary: `{TABLE_DIR / 'Table_HGNN8_feature_dictionary.csv'}`",
        f"- Graph integrity audit: `{TABLE_DIR / 'Table_HGNN10_graph_integrity_audit.csv'}`",
        f"- Figures: `{FIG_DIR}`",
        "",
    ]
    report.write_text("\n".join(lines), encoding="utf-8")
    return report


def main() -> None:
    ensure_dirs()
    axis = read_csv(BASE_AXIS_TABLE)
    if axis.empty:
        raise FileNotFoundError(BASE_AXIS_TABLE)
    axis = enrich_axis_table(axis)
    spatial_samples = read_csv(SPATIAL_SAMPLE_TABLE)
    depmap_gene = read_csv(DEPMAP_GENE_TABLE)

    nodes, edges = build_graph(axis, spatial_samples, depmap_gene)
    labels = build_axis_labels(axis)
    decoys = build_negative_decoys(axis)
    missing = build_missingness(axis)
    summary = build_summary(nodes, edges, labels, decoys)
    label_distribution = build_label_distribution(labels, decoys)
    integrity = build_integrity_audit(nodes, edges, labels, decoys)
    feature_dictionary = build_feature_dictionary()
    schema = read_csv(GRAPH_SCHEMA_TABLE)

    node_path = write_table(nodes, "Table_HGNN1_node_table")
    edge_path = write_table(edges, "Table_HGNN2_edge_table")
    label_path = write_table(labels, "Table_HGNN3_axis_label_table")
    summary_path = write_table(summary, "Table_HGNN4_node_edge_summary")
    missing_path = write_table(missing, "Table_HGNN5_feature_missingness")
    label_distribution_path = write_table(label_distribution, "Table_HGNN6_label_distribution")
    decoy_path = write_table(decoys, "Table_HGNN7_negative_axis_decoy_pool")
    feature_dictionary_path = write_table(feature_dictionary, "Table_HGNN8_feature_dictionary")
    if not schema.empty:
        schema_path = write_table(schema, "Table_HGNN9_original_design_graph_schema")
    else:
        schema_path = Path("")
    integrity_path = write_table(integrity, "Table_HGNN10_graph_integrity_audit")

    figures = make_schema_figure(summary)
    report = write_report(summary, missing, figures)

    catalog = pd.DataFrame(
        [
            {"table_id": "Table_HGNN1", "description": "Spatial-Causal-HGNN node table.", "project_csv": str(node_path)},
            {"table_id": "Table_HGNN2", "description": "Spatial-Causal-HGNN edge table.", "project_csv": str(edge_path)},
            {"table_id": "Table_HGNN3", "description": "Axis labels and soft rank labels.", "project_csv": str(label_path)},
            {"table_id": "Table_HGNN4", "description": "Node, edge, label, and decoy summary.", "project_csv": str(summary_path)},
            {"table_id": "Table_HGNN5", "description": "Feature missingness and score distribution audit.", "project_csv": str(missing_path)},
            {"table_id": "Table_HGNN6", "description": "Observed label and negative-decoy distribution by cancer.", "project_csv": str(label_distribution_path)},
            {"table_id": "Table_HGNN7", "description": "Initial hard-negative decoy candidate pool.", "project_csv": str(decoy_path)},
            {"table_id": "Table_HGNN8", "description": "Feature-to-edge dictionary for graph model inputs.", "project_csv": str(feature_dictionary_path)},
            {"table_id": "Table_HGNN9", "description": "Original graph schema from full design control.", "project_csv": str(schema_path) if str(schema_path) else ""},
            {"table_id": "Table_HGNN10", "description": "Graph integrity audit.", "project_csv": str(integrity_path)},
        ]
    )
    catalog_path = write_table(catalog, "table_catalog")

    manifest = {
        "run_date": RUN_DATE,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "n_nodes": int(len(nodes)),
        "n_edges": int(len(edges)),
        "n_axis_labels": int(len(labels)),
        "n_negative_decoys": int(len(decoys)),
        "tables": str(TABLE_DIR),
        "figures": figures,
        "catalog": str(catalog_path),
        "report_path": str(report),
    }
    manifest_path = METADATA_DIR / f"run_manifest_spatial_causal_hgnn_graph_inputs_{RUN_DATE}.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
