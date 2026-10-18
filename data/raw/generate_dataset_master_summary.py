from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager, rcParams
from matplotlib.colors import LinearSegmentedColormap


ROOT = Path(__file__).resolve().parent
sys.path.append(str(ROOT))
ARIAL_FONT_FILE = Path("/mnt/c/Windows/Fonts/arial.ttf")

import generate_dataset_advanced_figures as adv
import generate_dataset_comparison_figures as comp


OUT_SVG = ROOT / "dataset_master_summary_figure.svg"
OUT_PNG = ROOT / "dataset_master_summary_figure.png"
HEATMAP_CMAP = LinearSegmentedColormap.from_list(
    "custom_paper_heatmap",
    ["#92B4C8", "#ABD3E1", "#E3EDE0", "#FFE9BE", "#FAC795", "#EEA599"],
)
SPLIT_CMAP = LinearSegmentedColormap.from_list(
    "split_hygiene_heatmap",
    ["#F6FBFD", "#ABD3E1", "#EEA599"],
)


def postprocess_svg_font(svg_path: Path) -> None:
    text = svg_path.read_text(encoding="utf-8")
    text = text.replace("font-family: 'DejaVu Sans'", "font-family: 'Arial'")
    text = text.replace("font-family: 'sans-serif'", "font-family: 'Arial'")
    svg_path.write_text(text, encoding="utf-8")


def configure_font() -> None:
    if ARIAL_FONT_FILE.exists():
        font_manager.fontManager.addfont(str(ARIAL_FONT_FILE))
        rcParams["font.family"] = "Arial"
        rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans"]
    else:
        rcParams["font.family"] = "DejaVu Sans"
        rcParams["font.sans-serif"] = ["DejaVu Sans"]


def add_panel_label(ax, label: str) -> None:
    ax.text(
        -0.12,
        1.08,
        label,
        transform=ax.transAxes,
        fontsize=18,
        fontweight="normal",
        fontfamily="Arial",
        va="top",
        ha="left",
    )


def draw_scale_balance(ax, summary_df, names):
    subset = summary_df.set_index("dataset").loc[names]
    pos = subset["positive"].to_numpy()
    neg = subset["negative"].to_numpy()
    x = np.arange(len(names))
    ax.bar(x, neg, color="#ABD3E1", edgecolor="#333333", linewidth=0.7, label="Negative")
    ax.bar(x, pos, bottom=neg, color="#EEA599", edgecolor="#333333", linewidth=0.7, label="Positive")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=25, ha="right")
    ax.set_ylabel("Unique sequences")
    ax.set_title("Scale and class composition")
    ax.legend(frameon=False, fontsize=9)


def draw_overlap(ax, overlap_df, names):
    data = overlap_df.loc[names, names].to_numpy()
    im = ax.imshow(data, cmap=HEATMAP_CMAP, vmin=0, vmax=1)
    ax.set_xticks(np.arange(len(names)))
    ax.set_xticklabels(names, rotation=30, ha="right")
    ax.set_yticks(np.arange(len(names)))
    ax.set_yticklabels(names)
    ax.set_title("Exact-sequence Jaccard overlap")
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            value = data[i, j]
            color = "white" if value > 0.55 else "#222222"
            ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=8, color=color)
    return im


def draw_split_integrity(ax, split_df, names):
    train_overlap = (
        split_df[split_df["comparison"].str.startswith("train vs")]
        .groupby("dataset")["exact_overlap_ratio"]
        .max()
        .reindex(names)
    )
    secondary_reuse = (
        split_df[~split_df["comparison"].str.startswith("train vs")]
        .groupby("dataset")["exact_overlap_ratio"]
        .max()
        .reindex(names)
    )
    label_conflict = split_df.groupby("dataset")["label_conflict"].sum().reindex(names).fillna(0).astype(int)

    max_conflict = max(int(label_conflict.max()), 1)
    conflict_burden = label_conflict / max_conflict

    matrix = np.column_stack(
        [
            train_overlap.to_numpy(dtype=float),
            secondary_reuse.to_numpy(dtype=float),
            conflict_burden.to_numpy(dtype=float),
        ]
    )
    masked = np.ma.masked_invalid(matrix)
    cmap = SPLIT_CMAP.copy()
    cmap.set_bad("#E8E8E8")

    im = ax.imshow(masked, cmap=cmap, vmin=0.0, vmax=1.0, aspect="auto")
    ax.set_title("Split hygiene matrix")
    ax.set_xticks(np.arange(3))
    ax.set_xticklabels(
        ["Train/test\noverlap", "Extra-test\nreuse", "Label\nconflicts"],
        fontsize=9,
    )
    ax.set_yticks(np.arange(len(names)))
    ax.set_yticklabels(names, fontsize=9)

    ax.set_xticks(np.arange(-0.5, 3, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(names), 1), minor=True)
    ax.grid(which="minor", color="#FFFFFF", linestyle="-", linewidth=1.2)
    ax.tick_params(which="minor", bottom=False, left=False)

    def fmt_ratio(value: float | None) -> str:
        if value is None or np.isnan(value):
            return "NA"
        if value == 0:
            return "0"
        if value == 1:
            return "100%"
        if value < 0.01:
            return f"{value * 100:.2f}%"
        return f"{value:.2f}"

    for i, name in enumerate(names):
        display_values = [
            fmt_ratio(float(train_overlap.loc[name]) if pd.notna(train_overlap.loc[name]) else np.nan),
            fmt_ratio(float(secondary_reuse.loc[name]) if pd.notna(secondary_reuse.loc[name]) else np.nan),
            str(int(label_conflict.loc[name])),
        ]
        for j, text in enumerate(display_values):
            value = matrix[i, j]
            if np.isnan(value):
                color = "#666666"
            elif value > 0.58:
                color = "#FFFFFF"
            else:
                color = "#222222"
            ax.text(j, i, text, ha="center", va="center", fontsize=8.5, color=color)

    ax.text(
        0.0,
        -0.16,
        "Gray = not applicable (no secondary split).",
        transform=ax.transAxes,
        fontsize=8,
        ha="left",
        va="top",
        color="#555555",
    )
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
        spine.set_color("#666666")
    return im


def draw_hard_but_fair(ax, shortcut_df, novelty_df, names):
    shortcut_summary = shortcut_df.groupby("dataset").mean(numeric_only=True)
    novelty_summary = novelty_df.groupby("dataset")["score"].mean().rename("mean_similarity")
    merged = shortcut_summary.join(novelty_summary)
    merged["novelty"] = 1.0 - merged["mean_similarity"]
    for name in names:
        x = merged.loc[name, "mean_feature_ks"]
        y = merged.loc[name, "novelty"]
        ax.scatter(
            x,
            y,
            s=120,
            color=adv.DATASET_COLORS[name],
            edgecolor="#333333",
            linewidth=0.8,
        )
        ax.text(x + 0.001, y + 0.003, name, fontsize=9, color="#222222")
    ax.set_xlabel("Train-test shift on simple features (mean KS)")
    ax.set_ylabel("Sequence novelty (1 - nearest 3-mer similarity)")
    ax.set_title("Hard but fair regime")


def draw_length_balance(ax, records_df, names):
    bins = [(1, 10), (11, 20), (21, 30), (31, 40), (41, 50), (51, 100)]
    labels = [f"{lo}-{hi}" for lo, hi in bins]
    for name in names:
        subset = records_df.loc[records_df["dataset"] == name]
        ys = []
        for lo, hi in bins:
            bin_df = subset[(subset["length"] >= lo) & (subset["length"] <= hi)]
            ys.append(bin_df["label"].mean() if len(bin_df) else np.nan)
        ax.plot(labels, ys, marker="o", linewidth=2.0, label=name, color=adv.DATASET_COLORS[name])
    ax.axhline(0.5, color="#666666", linestyle="--", linewidth=1.0)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Positive ratio")
    ax.set_xlabel("Length bin (aa)")
    ax.set_title("Balance across length subgroups")
    ax.legend(frameon=False, fontsize=8, ncol=2)


def draw_motif_diversity(ax, diversity_df, names):
    subset = diversity_df.set_index("dataset").loc[names]
    tri = subset["tripeptide_count"].to_numpy(dtype=float)
    tetra = subset["tetrapeptide_count"].to_numpy(dtype=float)
    tri_scaled = tri / tri.max()
    tetra_scaled = tetra / tetra.max()
    x = np.arange(len(names))
    w = 0.36
    ax.bar(x - w / 2, tri_scaled, width=w, color="#ABD3E1", edgecolor="#333333", linewidth=0.7, label="3-mer coverage")
    ax.bar(x + w / 2, tetra_scaled, width=w, color="#EEA599", edgecolor="#333333", linewidth=0.7, label="4-mer coverage")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=25, ha="right")
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Normalized coverage")
    ax.set_title("Motif-space diversity")
    ax.legend(frameon=False, fontsize=8)


def draw_shortcut(ax, shortcut_df, names):
    subset = shortcut_df.groupby("dataset").mean(numeric_only=True).loc[names]
    simple_mcc = subset["simple_mcc"].to_numpy()
    cys_mcc = subset["cys_only_mcc"].to_numpy()
    x = np.arange(len(names))
    w = 0.36
    ax.bar(x - w / 2, simple_mcc, width=w, color="#ABD3E1", edgecolor="#333333", linewidth=0.7, label="5-feature baseline MCC")
    ax.bar(x + w / 2, cys_mcc, width=w, color="#EEA599", edgecolor="#333333", linewidth=0.7, label="Cys-only baseline MCC")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=25, ha="right")
    ax.set_ylim(0, max(0.72, float(max(simple_mcc.max(), cys_mcc.max()) + 0.05)))
    ax.set_ylabel("MCC")
    ax.set_title("Shortcut susceptibility")
    ax.legend(frameon=False, fontsize=8)


def draw_scorecard(ax, diversity_df, split_df, novelty_df, shortcut_df, names):
    integrity_summary = (
        split_df[split_df["comparison"].str.startswith("train vs")]
        .groupby("dataset")
        .agg(
            exact_overlap_ratio=("exact_overlap_ratio", "max"),
            label_conflict=("label_conflict", "sum"),
        )
    )
    novelty_summary = novelty_df.groupby("dataset")["score"].mean().rename("mean_similarity")
    shortcut_summary = shortcut_df.groupby("dataset").mean(numeric_only=True)["simple_mcc"].rename("simple_mcc")
    score_df = diversity_df.set_index("dataset").join(integrity_summary).join(novelty_summary).join(shortcut_summary)
    score_df["split_hygiene"] = 1.0 - score_df["exact_overlap_ratio"]
    score_df["novelty_score"] = 1.0 - score_df["mean_similarity"]
    score_df["conflict_free"] = 1.0 / (1.0 + score_df["label_conflict"])
    score_df["shortcut_resistance"] = 1.0 - score_df["simple_mcc"]

    display = score_df.loc[
        names,
        [
            "task_length_alignment",
            "balance_score",
            "task_length_charge_cell_occupancy",
            "tetrapeptide_count",
            "split_hygiene",
            "novelty_score",
            "shortcut_resistance",
            "conflict_free",
        ],
    ].copy()

    for col in ["task_length_charge_cell_occupancy", "tetrapeptide_count"]:
        values = display[col].to_numpy(dtype=float)
        min_v = values.min()
        max_v = values.max()
        display[col] = (values - min_v) / (max_v - min_v) if max_v > min_v else 1.0

    display.columns = [
        "Task-aligned",
        "Balanced",
        "Coverage",
        "4-mer diversity",
        "Clean split",
        "Novelty",
        "Shortcut-resistant",
        "Conflict-free",
    ]

    data = display.to_numpy(dtype=float)
    im = ax.imshow(data, cmap=HEATMAP_CMAP, vmin=0, vmax=1)
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_xticklabels(display.columns, rotation=30, ha="right")
    ax.set_yticks(np.arange(len(names)))
    ax.set_yticklabels(names)
    ax.set_title("Integrated benchmark scorecard")
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            value = data[i, j]
            color = "white" if value > 0.58 else "#222222"
            ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=8, color=color)
    return im


def main():
    configure_font()
    plt.rcParams["svg.fonttype"] = "none"

    bundles = comp.build_bundles()
    summary_df = comp.make_summary_df(bundles)
    overlap_df = comp.make_overlap_df(bundles)

    benchmarks = adv.build_benchmarks()
    split_df = adv.make_split_integrity_df(benchmarks)
    records_df = adv.make_combined_records_df(benchmarks)
    diversity_df = adv.make_diversity_df(records_df)
    novelty_df = adv.make_novelty_df(benchmarks)
    shortcut_df = adv.make_shortcut_df(benchmarks)

    names6 = ["ToxiPep 0.9", "ToxiPep 0.8", "CAPTP", "StrucToxNet", "ToxMSRC", "tAMPer"]

    fig = plt.figure(figsize=(18, 24))
    gs = fig.add_gridspec(4, 2, hspace=0.34, wspace=0.25)

    ax_a = fig.add_subplot(gs[0, 0])
    draw_scale_balance(ax_a, summary_df, names6)
    add_panel_label(ax_a, "a")

    ax_b = fig.add_subplot(gs[0, 1])
    im_b = draw_overlap(ax_b, overlap_df, names6)
    add_panel_label(ax_b, "b")
    fig.colorbar(im_b, ax=ax_b, fraction=0.046, pad=0.03, label="Jaccard overlap")

    ax_c = fig.add_subplot(gs[1, 0])
    draw_split_integrity(ax_c, split_df, names6)
    add_panel_label(ax_c, "c")

    ax_d = fig.add_subplot(gs[1, 1])
    draw_hard_but_fair(ax_d, shortcut_df, novelty_df, names6)
    add_panel_label(ax_d, "d")

    ax_e = fig.add_subplot(gs[2, 0])
    draw_length_balance(ax_e, records_df, names6)
    add_panel_label(ax_e, "e")

    ax_f = fig.add_subplot(gs[2, 1])
    draw_motif_diversity(ax_f, diversity_df, names6)
    add_panel_label(ax_f, "f")

    ax_g = fig.add_subplot(gs[3, 0])
    draw_shortcut(ax_g, shortcut_df, names6)
    add_panel_label(ax_g, "g")

    ax_h = fig.add_subplot(gs[3, 1])
    im_h = draw_scorecard(ax_h, diversity_df, split_df, novelty_df, shortcut_df, names6)
    add_panel_label(ax_h, "h")
    fig.colorbar(im_h, ax=ax_h, fraction=0.046, pad=0.03, label="Normalized score")

    fig.suptitle("Integrated comparison of peptide-toxicity benchmarks", fontsize=19, y=0.995)
    fig.savefig(OUT_SVG, bbox_inches="tight")
    fig.savefig(OUT_PNG, bbox_inches="tight", dpi=300)
    plt.close(fig)
    postprocess_svg_font(OUT_SVG)


if __name__ == "__main__":
    main()
