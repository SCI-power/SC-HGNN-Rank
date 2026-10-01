from __future__ import annotations

import json
import os
from pathlib import Path

import anndata as ad
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import tangram as tg
import torch
from scipy.io import mmread
from scipy import sparse


ANALYSIS_READY = Path(os.environ["SC_HGNN_ANALYSIS_READY"]).resolve()
INPUT_DIR = ANALYSIS_READY / "spatial_mapping" / "pilot_inputs"
RCTD_DIR = ANALYSIS_READY / "spatial_mapping" / "pilot_results" / "rctd_spotlight"
OUT_DIR = ANALYSIS_READY / "spatial_mapping" / "pilot_results" / "tangram_candidate_projection"
AXES_PATH = ANALYSIS_READY / "tcm_regulon_final_mechanism_axes" / "publication_flagship_mechanism_axes_top30.csv"


def parse_samples() -> set[str] | None:
    raw = os.environ.get("TANGRAM_PILOT_SAMPLES", "").strip()
    if not raw:
        return None
    return {x.strip() for x in raw.split(",") if x.strip()}


def read_bundle(dir_path: Path, item_label: str) -> ad.AnnData:
    matrix = mmread(dir_path / "counts.mtx").tocsr().T
    genes = pd.read_csv(dir_path / "genes.tsv", sep="\t", header=None)[0].astype(str).tolist()
    items = pd.read_csv(dir_path / f"{item_label}s.tsv", sep="\t", header=None)[0].astype(str).tolist()
    var = pd.DataFrame(index=pd.Index(genes, name="gene"))
    obs = pd.DataFrame(index=pd.Index(items, name=f"{item_label}_id"))
    adata = ad.AnnData(X=matrix, obs=obs, var=var)
    adata.var_names_make_unique()
    return adata


def load_axes() -> tuple[pd.DataFrame, list[str]]:
    axes = pd.read_csv(AXES_PATH)
    gene_cols = ["target_gene_symbol", "tf_gene_symbol", "regulon_target_gene_symbol"]
    genes: list[str] = []
    for col in gene_cols:
        genes.extend(axes[col].dropna().astype(str).str.strip().tolist())
    genes = sorted({g for g in genes if g and g.lower() != "nan"})
    return axes, genes


def load_training_genes(sample_out: Path, candidate_genes: list[str]) -> list[str]:
    marker_path = sample_out / "spotlight_marker_genes.csv"
    genes = list(candidate_genes)
    if marker_path.exists():
        markers = pd.read_csv(marker_path)
        if "gene" in markers.columns:
            genes.extend(markers["gene"].dropna().astype(str).tolist())
    return sorted({g.strip() for g in genes if g.strip()})


def zscore_frame(df: pd.DataFrame) -> pd.DataFrame:
    z = df.copy()
    for col in z.columns:
        values = z[col].to_numpy(dtype=float)
        sd = np.nanstd(values)
        if sd == 0 or not np.isfinite(sd):
            z[col] = 0.0
        else:
            z[col] = (values - np.nanmean(values)) / sd
    return z


def safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 3 or np.nanstd(a) == 0 or np.nanstd(b) == 0:
        return float("nan")
    return float(pd.Series(a).corr(pd.Series(b), method="spearman"))


def summarize_gene_spatial(projected: pd.DataFrame, rctd_weights: pd.DataFrame) -> pd.DataFrame:
    common = projected.index.intersection(rctd_weights.index)
    out = []
    for gene in projected.columns:
        vals = projected.loc[common, gene].astype(float).to_numpy()
        corrs = {
            ct: safe_corr(vals, rctd_weights.loc[common, ct].astype(float).to_numpy())
            for ct in rctd_weights.columns
        }
        valid_corrs = {k: v for k, v in corrs.items() if np.isfinite(v)}
        top_ct = max(valid_corrs, key=lambda k: abs(valid_corrs[k])) if valid_corrs else ""
        cutoff = np.nanquantile(vals, 0.9) if len(vals) else np.nan
        out.append(
            {
                "gene": gene,
                "n_spots": int(len(vals)),
                "mean_projected": float(np.nanmean(vals)) if len(vals) else np.nan,
                "max_projected": float(np.nanmax(vals)) if len(vals) else np.nan,
                "top10_mean_projected": float(np.nanmean(vals[vals >= cutoff])) if len(vals) else np.nan,
                "top_correlated_rctd_celltype": top_ct,
                "top_abs_spearman": float(abs(valid_corrs[top_ct])) if top_ct else np.nan,
                "top_signed_spearman": float(valid_corrs[top_ct]) if top_ct else np.nan,
            }
        )
    return pd.DataFrame(out)


def summarize_axes(axes: pd.DataFrame, projected: pd.DataFrame, rctd_weights: pd.DataFrame) -> pd.DataFrame:
    projected_z = zscore_frame(projected)
    common = projected_z.index.intersection(rctd_weights.index)
    rows = []
    for _, row in axes.iterrows():
        genes = [
            str(row.get("target_gene_symbol", "")).strip(),
            str(row.get("tf_gene_symbol", "")).strip(),
            str(row.get("regulon_target_gene_symbol", "")).strip(),
        ]
        genes = [g for g in dict.fromkeys(genes) if g and g.lower() != "nan" and g in projected_z.columns]
        if genes:
            axis_score = projected_z.loc[common, genes].mean(axis=1).to_numpy(dtype=float)
        else:
            axis_score = np.repeat(np.nan, len(common))
        corrs = {
            ct: safe_corr(axis_score, rctd_weights.loc[common, ct].astype(float).to_numpy())
            for ct in rctd_weights.columns
        }
        valid_corrs = {k: v for k, v in corrs.items() if np.isfinite(v)}
        top_ct = max(valid_corrs, key=lambda k: abs(valid_corrs[k])) if valid_corrs else ""
        rows.append(
            {
                "compound_master_id": row.get("compound_master_id", ""),
                "compound_representative_name": row.get("compound_representative_name", ""),
                "target_gene_symbol": row.get("target_gene_symbol", ""),
                "tf_gene_symbol": row.get("tf_gene_symbol", ""),
                "regulon_target_gene_symbol": row.get("regulon_target_gene_symbol", ""),
                "genes_projected": ";".join(genes),
                "n_genes_projected": len(genes),
                "axis_validation_score": row.get("axis_validation_score", np.nan),
                "best_validation_tier": row.get("best_validation_tier", ""),
                "axis_spatial_mean_z": float(np.nanmean(axis_score)) if len(genes) else np.nan,
                "axis_spatial_top10_mean_z": float(np.nanmean(axis_score[axis_score >= np.nanquantile(axis_score, 0.9)])) if len(genes) else np.nan,
                "top_correlated_rctd_celltype": top_ct,
                "top_abs_spearman": float(abs(valid_corrs[top_ct])) if top_ct else np.nan,
                "top_signed_spearman": float(valid_corrs[top_ct]) if top_ct else np.nan,
            }
        )
    return pd.DataFrame(rows)


def plot_projected_genes(projected: pd.DataFrame, coords: pd.DataFrame, genes: list[str], out_png: Path, title: str) -> None:
    genes = [g for g in genes if g in projected.columns][:12]
    if not genes:
        return
    coords = coords.set_index("spot_id").reindex(projected.index)
    ncols = 4
    nrows = int(np.ceil(len(genes) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 3 * nrows), squeeze=False)
    for ax, gene in zip(axes.ravel(), genes):
        vals = projected[gene].astype(float).to_numpy()
        sca = ax.scatter(coords["x"], coords["y"], c=vals, s=4, cmap="magma", linewidths=0)
        ax.set_title(gene, fontsize=9)
        ax.set_aspect("equal")
        ax.invert_yaxis()
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(sca, ax=ax, fraction=0.046, pad=0.02)
    for ax in axes.ravel()[len(genes) :]:
        ax.axis("off")
    fig.suptitle(title, fontsize=11, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_png, dpi=300)
    plt.close(fig)


def run_one(row: pd.Series, axes: pd.DataFrame, candidate_genes: list[str], epochs: int, device: str) -> dict:
    sample_id = row["sample_id"]
    sample_out = OUT_DIR / sample_id
    sample_out.mkdir(parents=True, exist_ok=True)
    print(f"[sample] {sample_id}", flush=True)
    summary_path = sample_out / "tangram_summary.json"
    force = os.environ.get("TANGRAM_FORCE", "0").strip() == "1"
    expected_outputs = [
        sample_out / "tangram_cell_to_space_map.h5ad",
        sample_out / "tangram_projected_candidate_genes.h5ad",
        sample_out / "tangram_projected_candidate_genes.csv.gz",
        sample_out / "tangram_candidate_gene_spatial_summary.csv",
        sample_out / "tangram_axis_spatial_summary.csv",
        summary_path,
    ]
    if not force and all(path.exists() for path in expected_outputs):
        print(f"[Tangram] {sample_id} using existing projection", flush=True)
        return json.loads(summary_path.read_text(encoding="utf-8-sig"))

    adata_sc = read_bundle(Path(row["reference_dir"]), "cell")
    adata_sp = read_bundle(Path(row["spatial_dir"]), "spot")
    labels = pd.read_csv(row["reference_labels_csv"])
    labels = labels.set_index("reference_cell_id").reindex(adata_sc.obs_names)
    adata_sc.obs["celltype"] = labels["reference_celltype"].astype(str).to_numpy()
    coords = pd.read_csv(row["spatial_coords_csv"])
    coords = coords.set_index("spot_id").reindex(adata_sp.obs_names).reset_index()
    adata_sp.obs["x"] = coords["x"].to_numpy()
    adata_sp.obs["y"] = coords["y"].to_numpy()

    training_genes = load_training_genes(RCTD_DIR / sample_id, candidate_genes)
    training_genes_lower = sorted({g.lower() for g in training_genes})
    candidate_lower_to_original = {g.lower(): g for g in candidate_genes}

    adata_sc.var_names = [g.lower() for g in adata_sc.var_names]
    adata_sp.var_names = [g.lower() for g in adata_sp.var_names]
    adata_sc.var_names_make_unique()
    adata_sp.var_names_make_unique()
    tg.pp_adatas(adata_sc, adata_sp, genes=training_genes_lower, gene_to_lowercase=False)
    if len(adata_sc.uns["training_genes"]) < 50:
      raise RuntimeError(f"{sample_id}: too few Tangram training genes: {len(adata_sc.uns['training_genes'])}")

    torch.manual_seed(20260503)
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.manual_seed_all(20260503)
    adata_map = tg.map_cells_to_space(
        adata_sc,
        adata_sp,
        mode="cells",
        device=device,
        learning_rate=0.1,
        num_epochs=epochs,
        scale=True,
        random_state=20260503,
        verbose=True,
        density_prior="rna_count_based",
    )
    adata_map.write_h5ad(sample_out / "tangram_cell_to_space_map.h5ad", compression="gzip")

    projection_genes_lower = [g for g in candidate_lower_to_original if g in adata_sc.var_names]
    adata_sc_proj = adata_sc[:, projection_genes_lower].copy()
    adata_ge = tg.project_genes(adata_map, adata_sc_proj, scale=True)
    adata_ge.write_h5ad(sample_out / "tangram_projected_candidate_genes.h5ad", compression="gzip")
    projected = pd.DataFrame(
        np.asarray(adata_ge.X),
        index=adata_ge.obs_names.astype(str),
        columns=[candidate_lower_to_original.get(g, g.upper()) for g in adata_ge.var_names.astype(str)],
    )
    projected.to_csv(sample_out / "tangram_projected_candidate_genes.csv.gz", index_label="spot_id")

    rctd = pd.read_csv(RCTD_DIR / sample_id / "rctd_weights.csv.gz").set_index("spot_id")
    gene_summary = summarize_gene_spatial(projected, rctd)
    gene_summary.insert(0, "sample_id", sample_id)
    gene_summary.insert(1, "cancer", row["cancer"])
    gene_summary.to_csv(sample_out / "tangram_candidate_gene_spatial_summary.csv", index=False, encoding="utf-8-sig")

    axis_summary = summarize_axes(axes, projected, rctd)
    axis_summary.insert(0, "sample_id", sample_id)
    axis_summary.insert(1, "cancer", row["cancer"])
    axis_summary.to_csv(sample_out / "tangram_axis_spatial_summary.csv", index=False, encoding="utf-8-sig")

    top_genes = gene_summary.sort_values("top10_mean_projected", ascending=False)["gene"].head(12).tolist()
    plot_projected_genes(projected, coords, top_genes, sample_out / "tangram_projected_gene_maps_top12.png", f"{sample_id} Tangram candidate genes")

    summary = {
        "sample_id": sample_id,
        "cancer": row["cancer"],
        "n_sc_cells": int(adata_sc.n_obs),
        "n_spots": int(adata_sp.n_obs),
        "n_training_genes": int(len(adata_sc.uns["training_genes"])),
        "n_projected_candidate_genes": int(projected.shape[1]),
        "epochs": int(epochs),
        "device": device,
        "output_dir": str(sample_out),
    }
    (sample_out / "tangram_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(INPUT_DIR / "spatial_mapping_pilot_manifest.csv")
    selected = parse_samples()
    if selected:
        manifest = manifest[manifest["cancer"].isin(selected) | manifest["sample_id"].isin(selected)].copy()
    if manifest.empty:
        raise RuntimeError("No samples selected for Tangram")
    axes, candidate_genes = load_axes()
    epochs = int(os.environ.get("TANGRAM_EPOCHS", "300"))
    requested_device = os.environ.get("TANGRAM_DEVICE", "cuda:0" if torch.cuda.is_available() else "cpu")
    if requested_device.startswith("cuda") and not torch.cuda.is_available():
        requested_device = "cpu"
    summaries = []
    for _, row in manifest.iterrows():
        summaries.append(run_one(row, axes, candidate_genes, epochs, requested_device))
    summary_df = pd.DataFrame(summaries)
    summary_df.to_csv(OUT_DIR / "tangram_candidate_projection_summary.csv", index=False, encoding="utf-8-sig")
    print(summary_df.to_string(index=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
