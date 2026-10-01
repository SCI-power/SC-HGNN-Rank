"""Summarize the fixed GraphOnly run and redraw only Supplementary Fig. S3L."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, FormatStrFormatter
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from run_graph_only import ROOT, VARIANT, COMPARATORS

TABLES = ROOT / "tables"
PLOT_ORDER = ["HGT-like-simple", "R-GCN-simple", "GCN-simple", "GAT-simple"]
LABELS = {"HGT-like-simple": "HGT-like", "R-GCN-simple": "R-GCN", "GCN-simple": "GCN", "GAT-simple": "GAT"}


def holm(p):
    p = np.asarray(p, float)
    order = np.argsort(p)
    values = np.minimum(1., np.maximum.accumulate(p[order]*np.arange(len(p), 0, -1)))
    result = np.empty_like(values)
    result[order] = values
    return result


def stars(p):
    return "***" if p < .001 else "**" if p < .01 else "*" if p < .05 else "ns"


def summarize():
    graph = pd.read_csv(TABLES / "graph_only_run_metrics.csv")
    refs = pd.read_csv(TABLES / "reused_standard_gnn_runs.csv")
    assert len(graph) == 65 and len(refs) == 260
    keys = ["validation_type", "fold_id", "seed"]
    metrics = ["spearman", "ndcg20", "mae", "ndcg50"]
    pairs = refs[keys+["variant"]+metrics].rename(columns={"variant": "comparator"}).merge(
        graph[keys+metrics], on=keys, how="left", validate="many_to_one", suffixes=("_reference", "_graphonly"))
    assert len(pairs) == 260
    for metric in metrics:
        pairs[f"delta_{metric}"] = pairs[f"{metric}_graphonly"] - pairs[f"{metric}_reference"]
    pairs.to_csv(TABLES / "S3L_seed_fold_paired_values.csv", index=False)
    columns = [f"{metric}_{role}" for metric in metrics for role in ["reference", "graphonly"]]
    columns += [f"delta_{metric}" for metric in metrics]
    folds = pairs.groupby(["comparator", "validation_type", "fold_id"], as_index=False)[columns].mean()
    assert folds.groupby("comparator").size().eq(13).all()
    folds.to_csv(TABLES / "S3L_seed_averaged_fold_differences.csv", index=False)
    rows = []
    rng = np.random.default_rng(20260923)
    resamples = rng.integers(0, 13, (5000, 13))
    for metric in metrics:
        for comparator in COMPARATORS:
            data = folds[folds.comparator.eq(comparator)].sort_values(["validation_type", "fold_id"])
            values = data[f"delta_{metric}"].to_numpy()
            bootstrap = values[resamples].mean(axis=1)
            lo, hi = np.quantile(bootstrap, [.025, .975])
            p = float(wilcoxon(values, alternative="two-sided", method="exact").pvalue) if np.any(values) else 1.
            rows.append({"comparator": comparator, "metric": metric, "n_folds": 13, "n_seeds_per_fold": 5,
                "n_seed_fold_pairs": 65, "graphonly_mean": data[f"{metric}_graphonly"].mean(),
                "reference_mean": data[f"{metric}_reference"].mean(), "mean_delta_graphonly_minus_reference": values.mean(),
                "ci95_low": lo, "ci95_high": hi, "wilcoxon_p_two_sided": p,
                "n_positive_fold_differences": int((values > 0).sum()), "n_bootstrap": 5000})
    result = pd.DataFrame(rows)
    result["p_holm_4_comparators"] = result.groupby("metric").wilcoxon_p_two_sided.transform(holm)
    result["significance"] = result.p_holm_4_comparators.map(stars)
    result.to_csv(TABLES / "S3L_paired_tests_all_metrics.csv", index=False)
    primary = result[result.metric.eq("spearman")].copy()
    primary.to_csv(TABLES / "S3L_forest_source_data.csv", index=False)
    summaries = []
    for (variant, design), g in pd.concat([graph, refs], ignore_index=True).groupby(["variant", "validation_type"]):
        row = {"variant": variant, "validation_type": design, "n_runs": len(g),
               "n_folds": g.fold_id.nunique(), "parameter_count": int(g.parameter_count.iloc[0])}
        for metric in metrics:
            row[f"mean_{metric}"] = g[metric].mean()
            row[f"sd_{metric}"] = g[metric].std(ddof=1)
        summaries.append(row)
    pd.DataFrame(summaries).to_csv(TABLES / "S3L_model_summary_by_design.csv", index=False)
    return primary


def draw(primary):
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial"], "font.size": 8,
        "axes.labelsize": 8, "xtick.labelsize": 7.3, "ytick.labelsize": 8,
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
        "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": .7,
        "xtick.major.size": 2.8, "ytick.major.size": 2.8,
        "text.color": "#253340", "axes.labelcolor": "#253340", "axes.edgecolor": "#506273",
        "savefig.facecolor": "white", "figure.facecolor": "white"})
    df = primary.set_index("comparator").loc[PLOT_ORDER]
    fig = plt.figure(figsize=(110/25.4, 78/25.4))
    fig.text(.015, .973, "L", fontsize=12, va="top")
    fig.text(.54, .963, "Graph-only model comparison", fontsize=9, ha="center", va="top")
    ax = fig.add_axes([.21, .26, .54, .55])
    for y, (comparator, row) in enumerate(df.iterrows()):
        mean = row.mean_delta_graphonly_minus_reference
        ax.errorbar(mean, y, xerr=[[mean-row.ci95_low], [row.ci95_high-mean]], fmt="o", ms=4.8,
                    color="#8A6FA8", ecolor="#8195A5", lw=1.45, capsize=2.8,
                    markeredgecolor="white", markeredgewidth=.4, zorder=3)
        p = row.p_holm_4_comparators
        text = "<0.001" if p < .001 else f"{p:.3f}"
        ax.text(1.10, y, f"{text} {row.significance}", transform=ax.get_yaxis_transform(),
                ha="left", va="center", fontsize=7.2)
    ax.text(1.10, 1.075, "P (Holm)", transform=ax.transAxes, fontsize=7.3)
    lo, hi = min(0., df.ci95_low.min()), max(0., df.ci95_high.max())
    span = max(hi-lo, .01)
    ax.set_xlim(lo-span*.13, hi+span*.13)
    ax.set_ylim(3.5, -.5)
    ax.axvline(0, color="#6D7A86", lw=.8, ls=(0, (3, 3)), zorder=1)
    ax.set_yticks(range(4), [LABELS[c] for c in PLOT_ORDER])
    ax.grid(axis="x", lw=.5, color="#DFE7EC")
    ax.set_axisbelow(True)
    ax.xaxis.set_major_locator(MaxNLocator(5, steps=[1, 2, 5, 10]))
    ax.xaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    ax.set_xlabel("GraphOnly - standard GNN Spearman", fontsize=7.5, labelpad=7)
    fig.text(.21, .075, "13 fold means; 5 seeds per fold\nMean difference with 95% bootstrap CI",
             fontsize=7, color="#5D6D78", linespacing=1.45)
    out = ROOT / "figures"
    out.mkdir(parents=True, exist_ok=True)
    for suffix in ["pdf", "svg", "png"]:
        fig.savefig(out / f"Supplementary_Figure_S3L_updated.{suffix}", dpi=600)
    (ROOT / "previews").mkdir(exist_ok=True)
    fig.savefig(ROOT / "previews/Supplementary_Figure_S3L_updated.png", dpi=200)
    plt.close(fig)


def write_notes(primary):
    out = ROOT / "legends"
    out.mkdir(parents=True, exist_ok=True)
    caption = (
        "(L) Matched graph-only model comparison. The original GraphOnly HGNN module, including its learned "
        "skip connection, was evaluated without direct candidate-axis or metapath feature vectors. It retained "
        "the same prior-weighted graph and node attributes as the standard GNN controls. All displayed models "
        "used hidden dimension 112, two graph layers, dropout 0.18, AdamW (learning rate 0.0016; weight decay "
        "0.00015), a maximum of 160 epochs, patience 28, and the same classification plus 0.25-weighted "
        "smooth-L1 regression objective. The fixed 13 partitions, five seeds, training-only sample weights and "
        "entity-edge masking protocol were identical. Points show GraphOnly-minus-comparator Spearman "
        "differences after averaging seeds within each fold; bars show unadjusted 95% percentile bootstrap intervals "
        "(5,000 fold resamples). Positive values favor GraphOnly. P values are from two-sided paired Wilcoxon "
        "tests across 13 fold means, with Holm correction across four comparators; *P<0.05, **P<0.01, "
        "***P<0.001, ns, not significant. This graph-only control is distinct from the complete SC-HGNN-Rank model."
    )
    (out / "Supplementary_Figure_S3L_legend_EN.txt").write_text(caption+"\n", encoding="utf-8")
    chinese = (
        "（L）GraphOnly与标准GNN的匹配比较。保留原GraphOnly模块及其可学习跳连，不输入候选轴或元路径特征向量；"
        "使用与标准GNN相同的含先验权重图及节点属性。所有图中模型统一隐藏维度112、图层数2、dropout 0.18、"
        "AdamW优化器（学习率0.0016，权重衰减0.00015）、训练上限160轮、早停patience 28，以及分类损失加0.25倍"
        "smooth-L1回归损失。采用相同的13个固定fold、5个种子、仅训练子集计算的样本权重及实体边遮蔽协议。"
        "每个fold先平均5个种子，大点表示GraphOnly减去相应对照的Spearman平均差值，横线为5,000次fold级"
        "bootstrap得到的95%置信区间。正值有利于GraphOnly。P值来自13个fold平均差值的双侧配对Wilcoxon检验，"
        "对四个对照进行Holm校正。GraphOnly是独立对照模型，不是完整SC-HGNN-Rank主模型。\n"
    )
    (out / "Supplementary_Figure_S3L_legend_CN.txt").write_text(chinese, encoding="utf-8")
    selected = primary.set_index("comparator").loc[PLOT_ORDER]
    lines = ["# S3L补做完成", "", "只替换原S3L的森林图；此前S3F-K及S5E的七张更新图保持不变。",
             "", "## 运行", "", "GraphOnly新增65次训练（13个固定fold × 5个种子），四个标准GNN的260次既有运行只读复用。",
             "训练和比较协议在运行前固定，没有根据结果调参。完整SC-HGNN-Rank主模型、候选轴、最终优先级和实验数据未改变。",
             "", "GraphOnly保留原类定义和可学习跳连，匹配设置覆盖原默认的层数、dropout等参数；不是将HGT-like改名。",
             "旧训练中的额外随机删边不用于本次比较，因为当前标准GNN对照没有该额外步骤。",
             "", "## 结果", "", "| 对照 | GraphOnly均值 | 对照均值 | 差值 | 95% CI | Holm P |",
             "|---|---:|---:|---:|---|---:|"]
    for comparator, row in selected.iterrows():
        lines.append(f"| {LABELS[comparator]} | {row.graphonly_mean:.4f} | {row.reference_mean:.4f} | "
                     f"{row.mean_delta_graphonly_minus_reference:+.4f} | {row.ci95_low:.4f}至{row.ci95_high:.4f} | "
                     f"{row.p_holm_4_comparators:.4f} |")
    lines += ["", "数值是GraphOnly相对标准GNN的比较，不能当作完整SC-HGNN-Rank相对这些对照的差值。",
              "置信区间反映固定交叉验证fold的变异，不是独立患者队列的生物学置信区间。",
              "", "## 文件", "", "- figures：可编辑PDF/SVG和600 dpi PNG。",
              "- tables：逐次运行、逐fold差值、全部指标和统计检验。",
              "- fits：65个控制模型检查点、预测与训练历史。",
              "- scripts：原GraphOnly类、匹配运行脚本及统计绘图代码。",
              "- legends：中英文图注。", "", "此前说明中的‘S3L尚未完成’状态由本次完成记录替代。",
              "原投稿TIFF、既有七张替换图和旧压缩包均未覆盖。"]
    (ROOT / "S3L补做结果与替换说明.md").write_text("\n".join(lines)+"\n", encoding="utf-8")


if __name__ == "__main__":
    primary = summarize()
    draw(primary)
    write_notes(primary)
    print(primary.to_string(index=False), flush=True)
