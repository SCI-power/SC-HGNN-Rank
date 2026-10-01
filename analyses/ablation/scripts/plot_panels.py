"""Separate editable Figure 4E, 4F and 4G panels with traceable source tables."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator, FormatStrFormatter
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from run_ablation import ROOT, BASE, MODEL_RESULTS, MODULES

TABLES = ROOT / "tables"
FIGURES = ROOT / "figures"
LEGENDS = ROOT / "legends"
MM = 1 / 25.4
COLORS = ["#356C98", "#8C629B", "#C18B38", "#BC5367"]
MODELS = ["SC-HGNN-Rank-full", "S_integrated"]
CANCERS = ["COAD", "LIHC", "STAD"]
plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial"],
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8.5,
    "xtick.labelsize": 7.5, "ytick.labelsize": 8, "legend.fontsize": 7.5,
    "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": .65, "xtick.major.width": .6, "ytick.major.width": .6,
    "xtick.major.size": 2.6, "ytick.major.size": 2.6,
    "text.color": "#20242B", "axes.labelcolor": "#20242B",
    "axes.edgecolor": "#565C64", "savefig.facecolor": "white",
    "figure.facecolor": "white", "legend.frameon": False,
})


def header(fig, letter, title):
    fig.text(.015, .96, letter, fontsize=13, fontweight="bold", ha="left", va="top")
    fig.text(.057, .958, title, fontsize=10, fontweight="bold", ha="left", va="top")


def export(fig, stem):
    FIGURES.mkdir(parents=True, exist_ok=True)
    for ext in ["pdf", "svg", "png"]:
        fig.savefig(FIGURES / f"{stem}.{ext}", dpi=600)
    fig.savefig(FIGURES / f"{stem}_preview.png", dpi=180)
    plt.close(fig)


def legend(letter, text):
    LEGENDS.mkdir(parents=True, exist_ok=True)
    (LEGENDS / f"Figure4{letter}_legend_EN.txt").write_text(text+"\n", encoding="utf-8")


def holm(values):
    p = np.asarray(values, float)
    order = np.argsort(p)
    adjusted = np.minimum(1, np.maximum.accumulate(p[order] * np.arange(len(p), 0, -1)))
    result = np.empty_like(p)
    result[order] = adjusted
    return result


def stars(p):
    return "***" if p < .001 else "**" if p < .01 else "*" if p < .05 else "ns"


def ablation_statistics():
    full = pd.read_csv(TABLES / "full_model_reference_runs.csv")
    runs = pd.read_csv(TABLES / "ablation_run_metrics.csv")
    assert len(full) == 65 and len(runs) == 260
    keys = ["validation_type", "fold_id", "seed"]
    metrics = ["spearman", "ndcg20", "mae"]
    pairs = runs.merge(full[keys+metrics], on=keys, suffixes=("_ablated", "_full"), validate="many_to_one")
    for metric in metrics:
        pairs[f"delta_{metric}"] = pairs[f"{metric}_full"] - pairs[f"{metric}_ablated"]
    pairs.to_csv(TABLES / "Figure4E_seed_fold_paired_values.csv", index=False)
    cols = [f"delta_{m}" for m in metrics] + [f"{m}_{s}" for m in metrics for s in ["ablated", "full"]]
    folds = pairs.groupby(["module", "validation_type", "fold_id"], as_index=False)[cols].mean()
    folds.to_csv(TABLES / "Figure4E_seed_averaged_fold_values.csv", index=False)
    rows = []
    for metric in metrics:
        for i, module in enumerate(MODULES):
            g = folds[folds.module.eq(module)]
            v = g[f"delta_{metric}"].to_numpy()
            assert len(v) == 13
            rng = np.random.default_rng(20260923+i)
            boots = rng.choice(v, (5000, len(v)), replace=True).mean(axis=1)
            lo, hi = np.quantile(boots, [.025, .975])
            p = float(wilcoxon(v, alternative="two-sided").pvalue) if np.any(v) else 1.
            rows.append({"module": module, "metric": metric, "n_folds": len(v), "n_seeds": 5,
                         "mean_full": g[f"{metric}_full"].mean(), "mean_ablated": g[f"{metric}_ablated"].mean(),
                         "mean_delta": v.mean(), "ci95_low": lo, "ci95_high": hi,
                         "wilcoxon_p_two_sided": p})
    summary = pd.DataFrame(rows)
    summary["p_holm_4_modules"] = summary.groupby("metric")["wilcoxon_p_two_sided"].transform(holm)
    summary["significance"] = summary.p_holm_4_modules.map(stars)
    summary.to_csv(TABLES / "Figure4E_ablation_summary.csv", index=False)
    by_design = folds.groupby(["module", "validation_type"], as_index=False)[cols].agg(["mean", "std"])
    by_design.to_csv(TABLES / "Figure4E_descriptive_by_design.csv")
    return folds, summary


def panel_e():
    folds, summary = ablation_statistics()
    fig = plt.figure(figsize=(180*MM, 76*MM))
    header(fig, "E", "Module ablation under held-out splits")
    axes = [fig.add_axes([.225, .24, .325, .53]), fig.add_axes([.635, .24, .325, .53])]
    for ax, metric, title in zip(axes, ["spearman", "ndcg20"], ["Spearman", "NDCG@20"]):
        s = summary[summary.metric.eq(metric)].set_index("module")
        values = folds[f"delta_{metric}"]
        low, high = min(0., values.min()), max(0., values.max())
        span = max(high-low, .01)
        ax.set_xlim(low-span*.10, high+span*.22)
        for i, (module, color) in enumerate(zip(MODULES, COLORS)):
            v = folds[folds.module.eq(module)][f"delta_{metric}"].to_numpy()
            yy = i + np.random.default_rng(192+i).uniform(-.16, .16, len(v))
            ax.scatter(v, yy, s=10, color=color, alpha=.23, edgecolors="none", zorder=2)
            row = s.loc[module]
            ax.errorbar(row.mean_delta, i, xerr=[[row.mean_delta-row.ci95_low], [row.ci95_high-row.mean_delta]],
                        fmt="o", color=color, ms=5.2, lw=1.8, capsize=2.8, zorder=4,
                        markeredgecolor="white", markeredgewidth=.5)
            ax.text(.99, i, row.significance, transform=ax.get_yaxis_transform(), ha="right", va="center",
                    fontsize=8, fontweight="bold" if row.significance != "ns" else "normal")
        ax.axvline(0, color="#737A83", lw=.8, ls=(0, (3, 3)), zorder=1)
        ax.set_ylim(3.5, -.5)
        ax.set_yticks(range(4))
        ax.set_title(title, loc="left", fontweight="bold", pad=9)
        ax.grid(axis="x", color="#E7E9EC", lw=.5)
        ax.set_axisbelow(True)
        ax.xaxis.set_major_locator(MaxNLocator(5, steps=[1, 2, 5, 10]))
        ax.xaxis.set_major_formatter(FormatStrFormatter("%.2f"))
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="y", length=0, pad=8)
        ax.set_xlabel("Full model - ablated model", labelpad=6)
    axes[0].set_yticklabels([MODULES[m]["label"] for m in MODULES])
    axes[1].set_yticklabels([])
    fig.text(.225, .085, "13 fold means; 5 seeds per fold", fontsize=7.2, color="#5A626D")
    fig.text(.635, .085, "Mean with 95% bootstrap CI", fontsize=7.2, color="#5A626D")
    export(fig, "Figure4E_module_ablation")
    legend("E", "(E) Matched module-ablation analysis of the original SC-HGNN-Rank architecture. "
           "Spatial, scRNA-GRN, PPI/MSI or drug/perturbation evidence was removed from its base-feature "
           "channels and corresponding graph relations, and derived interaction features were recomputed. "
           "Drug/perturbation ablation also removed LINCS and dependency-derived node scores. Each variant "
           "was retrained with the same objective, hyperparameters and 13 fixed held-out folds as the full model, "
           "using five seeds per fold (65 fits per variant). Supervision remained unchanged. Small points are "
           "seed-averaged paired differences for each fold; large points and bars show the mean and 95% "
           "percentile bootstrap interval across folds (5,000 resamples). Positive differences indicate better "
           "performance of the full model. Stars denote two-sided paired Wilcoxon tests across 13 fold means, "
           "with Holm correction across four modules separately for each metric: *P<0.05, **P<0.01, ***P<0.001; "
           "ns, not significant. These intervals summarize variation across evaluation folds.")


def panel_f():
    source = MODEL_RESULTS / "tables"
    metrics = pd.read_csv(source / "external_per_cancer_metrics.csv")
    ci = pd.read_csv(source / "external_direct_matching_auroc_ci.csv")
    rows = metrics[metrics.variant.isin(MODELS)].copy()
    rows["top20_matched_count"] = (20*rows.top20_matching_fraction).round().astype(int)
    rows.to_csv(TABLES / "Figure4F_external_matching_by_cancer.csv", index=False)
    ci = ci[ci.variant.isin(MODELS)].copy()
    ci.to_csv(TABLES / "Figure4F_direct_compound_macro_AUROC.csv", index=False)
    pair = pd.read_csv(source / "external_pair_level_inputs.csv")
    assert len(pair) == 272 and pair.any_direct_compound_drug_response_hit.sum() == 15
    use = ["cancer", "compound_master_id", "compound_representative_name", "target_gene_symbol"] + MODELS + [
        "any_direct_compound_drug_response_hit", "any_target_or_regulator_class_hit", "hpa_support_hit"]
    pair[use].to_csv(TABLES / "Figure4F_pair_level_source_data.csv", index=False)
    endpoints = ["direct_compound_matching", "target_regulator_class_matching", "protein_support"]
    mat = np.empty((3, 6))
    for i, endpoint in enumerate(endpoints):
        for j, cancer in enumerate(CANCERS):
            for k, model in enumerate(MODELS):
                r = rows[(rows.endpoint == endpoint) & (rows.cancer == cancer) & (rows.variant == model)].iloc[0]
                mat[i, j*2+k] = r.top20_matching_fraction
    cmap = LinearSegmentedColormap.from_list("coverage", ["#F2F5F7", "#A8C6D6", "#285B7D"])
    fig = plt.figure(figsize=(180*MM, 78*MM))
    header(fig, "F", "External-resource matching")
    ax = fig.add_axes([.21, .32, .415, .44])
    im = ax.pcolormesh(np.arange(7)-.5, np.arange(4)-.5, mat,
                       cmap=cmap, vmin=0, vmax=1, shading="flat", rasterized=False)
    ax.set_xlim(-.5, 5.5)
    ax.set_ylim(2.5, -.5)
    for i in range(3):
        for j in range(6):
            ax.text(j, i, f"{int(round(mat[i,j]*20))}/20", ha="center", va="center", fontsize=7.4,
                    color="white" if mat[i,j] >= .65 else "#20242B")
    ax.set_yticks(range(3), ["Compound match", "Target-class match", "Protein support"])
    ax.set_xticks(range(6), ["Rank", "Score"]*3)
    ax.set_xticks(np.arange(-.5, 6), minor=True)
    ax.set_yticks(np.arange(-.5, 3), minor=True)
    ax.grid(which="minor", color="white", lw=1.2)
    ax.tick_params(which="both", length=0)
    ax.tick_params(axis="y", pad=7)
    for spine in ax.spines.values():
        spine.set_visible(False)
    for j, cancer in enumerate(CANCERS):
        ax.text((2*j+.5)/6+1/12, 1.08, cancer, transform=ax.transAxes, ha="center", va="bottom",
                fontsize=8, fontweight="bold")
    for xline in [1.5, 3.5]:
        ax.axvline(xline, color="white", lw=4)
    fig.text(.21, .852, "Top20 resource matches", fontsize=8.5, fontweight="bold")
    cax = fig.add_axes([.21, .155, .19, .032])
    cb = fig.colorbar(im, cax=cax, orientation="horizontal", ticks=[0, .5, 1])
    cb.solids.set_rasterized(False)
    cb.ax.set_xticklabels(["0", "50", "100"])
    cb.ax.tick_params(labelsize=7, length=2)
    cb.outline.set_linewidth(.4)
    fig.text(.21, .23, "Matched fraction (%)", fontsize=7.3)
    fig.text(.43, .165, "Rank: SC-HGNN-Rank\nScore: S_integrated", fontsize=7.2, linespacing=1.5)

    bx = fig.add_axes([.72, .32, .245, .44])
    colors = ["#B63F55", "#597486"]
    for y, model, color in zip([1, 0], MODELS, colors):
        r = ci[ci.variant.eq(model)].iloc[0]
        bx.errorbar(r.estimate, y, xerr=[[r.estimate-r.ci95_low], [r.ci95_high-r.estimate]],
                    fmt="o", ms=5, lw=1.6, capsize=3, color=color)
        bx.text(.02, y+.32, "SC-HGNN-Rank" if y else "S_integrated", transform=bx.get_yaxis_transform(),
                fontsize=7.7, va="bottom", color=color, fontweight="bold")
        bx.text(.98, y-.27, f"{r.estimate:.3f} [{r.ci95_low:.3f}, {r.ci95_high:.3f}]",
                transform=bx.get_yaxis_transform(), ha="right", va="top", fontsize=7)
    bx.axvline(.5, lw=.8, ls=(0, (3, 3)), color="#858B93")
    bx.set_ylim(-.62, 1.65)
    bx.set_xlim(0, 1)
    bx.set_xticks([0, .5, 1])
    bx.set_yticks([])
    bx.spines["left"].set_visible(False)
    bx.set_xlabel("Macro-average AUROC", labelpad=5)
    fig.text(.72, .852, "Compound matching", fontsize=8.5, fontweight="bold")
    fig.text(.72, .145, "15 matches / 272 unique units\n95% bootstrap CI", fontsize=7.2,
             color="#5A626D", linespacing=1.5, va="top")
    export(fig, "Figure4F_external_resource_matching")
    legend("F", "(F) External-resource matching of SC-HGNN-Rank and the integrated-evidence score (S_integrated). "
           "Left: resource-supported entries among the 20 highest-ranked unique cancer-compound-target units "
           "within each cancer. Cell values show matched counts out of 20, and color indicates the matching "
           "fraction. Compound matching denotes an original-compound drug-response record; target-class "
           "matching denotes target- or regulator-class support; protein support denotes an HPA support score "
           "of at least 0.5. Candidate scores were aggregated from held-out predictions by within-cancer "
           "percentiles, followed by maximum aggregation across axes sharing the same cancer-compound-target "
           "unit, as in the accompanying source tables. Right: cancer-macro-averaged AUROC for compound-resource "
           "matching, with 95% bootstrap intervals (5,000 cancer-stratified unit resamples). There were 272 unique "
           "units and 15 matched units. The endpoint is presence of resource support; an unmatched entry does "
           "not denote experimentally established drug inactivity.")


def panel_g():
    source = pd.read_csv(TABLES / "Figure4G_module_sensitivity_summary.csv")
    modes = ["input_features", "graph_evidence"]
    maps = [LinearSegmentedColormap.from_list("features", ["#F6F2EF", "#DDA899", "#9C3E51"]),
            LinearSegmentedColormap.from_list("edges", ["#F0F4F7", "#96B8CC", "#25587B"])]
    fig = plt.figure(figsize=(180*MM, 73*MM))
    header(fig, "G", "Evidence dependence of the trained model")
    for i, (mode, cmap, top) in enumerate(zip(modes, maps, [.30, .030])):
        left = .225 if i == 0 else .625
        ax = fig.add_axes([left, .25, .265, .51])
        mat = np.array([[source[(source["module"]==module) & (source["mode"]==mode) & (source["cancer"]==cancer)]
                         .mean_absolute_change.iloc[0] for cancer in CANCERS] for module in MODULES])
        assert mat.min() >= 0 and mat.max() <= top
        im = ax.pcolormesh(np.arange(4)-.5, np.arange(5)-.5, mat,
                           cmap=cmap, norm=Normalize(0, top), shading="flat", rasterized=False)
        ax.set_xlim(-.5, 2.5)
        ax.set_ylim(3.5, -.5)
        for y in range(4):
            for x in range(3):
                ax.text(x, y, f"{mat[y,x]:.3f}", ha="center", va="center", fontsize=8,
                        color="white" if mat[y,x]/top > .66 else "#20242B")
        ax.set_xticks(range(3), CANCERS)
        ax.set_yticks(range(4), [MODULES[m]["label"] for m in MODULES] if i == 0 else [])
        ax.set_xticks(np.arange(-.5, 3), minor=True)
        ax.set_yticks(np.arange(-.5, 4), minor=True)
        ax.grid(which="minor", color="white", lw=1.6)
        ax.tick_params(which="both", length=0)
        ax.tick_params(axis="y", pad=8)
        for spine in ax.spines.values():
            spine.set_visible(False)
        fig.text(left, .823, "Feature masking" if i == 0 else "Graph-evidence masking", fontsize=8.5, fontweight="bold")
        cax = fig.add_axes([left+.28, .25, .013, .51])
        ticks = np.linspace(0, top, 4)
        cb = fig.colorbar(im, cax=cax, ticks=ticks)
        cb.solids.set_rasterized(False)
        cb.ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
        cb.ax.tick_params(labelsize=7, length=2)
        cb.outline.set_linewidth(.4)
    fig.text(.225, .10, "Mean absolute change in model score; 200 candidate axes per cancer", fontsize=7.4, color="#5A626D")
    export(fig, "Figure4G_model_evidence_dependence")
    legend("G", "(G) Sensitivity of held-out axis scores to feature-group and graph-evidence masking in the "
           "trained SC-HGNN-Rank model. The same 65 full-model checkpoints used for the primary benchmark were "
           "evaluated without retraining. Left: one evidence group was zeroed in the base features and derived "
           "interactions while the graph was retained. Right: the corresponding graph relations and source-specific "
           "node scores were removed, with graph-degree features recomputed and axis/metapath features retained. "
           "The four evidence-group definitions match panel E. Each cell reports the mean absolute score change, "
           "first averaged over 15 held-out evaluations per axis (three split designs and five seeds) and then over "
           "all 200 axes within each cancer. The two heatmaps use different, explicitly shown linear color scales "
           "but the same score units. Values measure one-group-at-a-time model sensitivity, not an additive "
           "percentage contribution.")


def main(panels):
    TABLES.mkdir(parents=True, exist_ok=True)
    funcs = {"E": panel_e, "F": panel_f, "G": panel_g}
    for p in panels:
        funcs[p]()
        print(f"Exported Figure4{p}: PDF, SVG, 600-dpi PNG and English legend", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--panels", default="EFG")
    main(parser.parse_args().panels.upper())
