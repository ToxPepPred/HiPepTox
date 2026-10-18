from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path("/home/cns/peptide-aa/总数据集")
OUT_DIR = ROOT


AA_ORDER = list("ACDEFGHIKLMNPQRSTVWY")
HYDRO = {
    "A": 1.8,
    "C": 2.5,
    "D": -3.5,
    "E": -3.5,
    "F": 2.8,
    "G": -0.4,
    "H": -3.2,
    "I": 4.5,
    "K": -3.9,
    "L": 3.8,
    "M": 1.9,
    "N": -3.5,
    "P": -1.6,
    "Q": -3.5,
    "R": -4.5,
    "S": -0.8,
    "T": -0.7,
    "V": 4.2,
    "W": -0.9,
    "Y": -1.3,
}

DATASET_COLORS = {
    "ToxiPep 0.9": "#EEA599",
    "ToxiPep 0.8": "#FAC795",
    "ToxiPep Independent": "#FFE9BE",
    "CAPTP": "#E3EDE0",
    "StrucToxNet": "#ABD3E1",
    "ToxMSRC": "#92B4C8",
    "tAMPer": "#7EA6B8",
}


@dataclass
class DatasetBundle:
    name: str
    seq_to_label: Dict[str, int]

    @property
    def sequences(self) -> List[str]:
        return list(self.seq_to_label.keys())


def read_fasta(path: Path) -> List[Tuple[str, str]]:
    items: List[Tuple[str, str]] = []
    header: str | None = None
    seq_chunks: List[str] = []
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for raw in handle:
            line = raw.strip()
            if not line:
                continue
            if line.startswith(">"):
                if header is not None:
                    items.append((header, "".join(seq_chunks)))
                header = line[1:]
                seq_chunks = []
            else:
                seq_chunks.append(line)
        if header is not None:
            items.append((header, "".join(seq_chunks)))
    return items


def load_toxipep_split(split: str) -> DatasetBundle:
    seq_to_label: Dict[str, int] = {}
    for part in ("train", "test"):
        path = ROOT / "ToxiPep" / "Dataset" / split / f"{part}.csv"
        with path.open("r", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                seq_to_label[row["sequence"]] = int(row["toxicity"])
    return DatasetBundle(name=f"ToxiPep {split}", seq_to_label=seq_to_label)


def load_toxipep_independent() -> DatasetBundle:
    path = ROOT / "ToxiPep" / "Dataset" / "Independent set" / "independent test set.txt"
    seq_to_label: Dict[str, int] = {}
    for header, seq in read_fasta(path):
        seq_to_label[seq] = 1 if header.lower().startswith("pos") else 0
    return DatasetBundle(name="ToxiPep Independent", seq_to_label=seq_to_label)


def load_captp() -> DatasetBundle:
    seq_to_label: Dict[str, int] = {}
    for filename in ("train_data.txt", "test1.txt", "test2.txt"):
        path = ROOT / "CAPTP" / "data" / filename
        for header, seq in read_fasta(path):
            seq_to_label[seq] = 1 if header.endswith("|1") else 0
    return DatasetBundle(name="CAPTP", seq_to_label=seq_to_label)


def load_structoxnet() -> DatasetBundle:
    seq_to_label: Dict[str, int] = {}
    for filename in ("train_data.txt", "test_data.txt"):
        path = ROOT / "StrucToxNet" / "data" / filename
        for header, seq in read_fasta(path):
            seq_to_label[seq] = 1 if header.endswith("|1") else 0
    return DatasetBundle(name="StrucToxNet", seq_to_label=seq_to_label)


def load_toxmsrc() -> DatasetBundle:
    seq_to_label: Dict[str, int] = {}
    for filename in ("train_data.fasta", "test1.fasta", "test2.fasta"):
        path = ROOT / "ToxMSRC" / "Raw data" / filename
        for header, seq in read_fasta(path):
            seq_to_label[seq] = 1 if header.endswith("|1") else 0
    return DatasetBundle(name="ToxMSRC", seq_to_label=seq_to_label)


def load_tamper() -> DatasetBundle:
    seq_to_label: Dict[str, int] = {}
    config = [
        ("tr_pos.faa", 1),
        ("val_pos.faa", 1),
        ("tr_neg.faa", 0),
        ("val_neg.faa", 0),
    ]
    for filename, label in config:
        path = ROOT / "tAMPer" / "data" / "sequences" / filename
        for _header, seq in read_fasta(path):
            seq_to_label[seq] = label
    return DatasetBundle(name="tAMPer", seq_to_label=seq_to_label)


def build_bundles() -> List[DatasetBundle]:
    return [
        load_toxipep_split("0.9"),
        load_toxipep_split("0.8"),
        load_toxipep_independent(),
        load_captp(),
        load_structoxnet(),
        load_toxmsrc(),
        load_tamper(),
    ]


def net_charge(seq: str) -> float:
    pos = seq.count("K") + seq.count("R") + 0.1 * seq.count("H")
    neg = seq.count("D") + seq.count("E")
    return pos - neg


def mean_hydrophobicity(seq: str) -> float:
    return float(np.mean([HYDRO[aa] for aa in seq if aa in HYDRO]))


def cysteine_fraction(seq: str) -> float:
    return seq.count("C") / len(seq)


def aromatic_fraction(seq: str) -> float:
    aromatic = seq.count("F") + seq.count("W") + seq.count("Y")
    return aromatic / len(seq)


def aa_frequency(seqs: Iterable[str]) -> Dict[str, float]:
    counts = {aa: 0 for aa in AA_ORDER}
    total = 0
    for seq in seqs:
        for aa in seq:
            if aa in counts:
                counts[aa] += 1
                total += 1
    return {aa: counts[aa] / total if total else 0.0 for aa in AA_ORDER}


def make_summary_df(bundles: List[DatasetBundle]) -> pd.DataFrame:
    rows = []
    for bundle in bundles:
        seqs = bundle.sequences
        labels = list(bundle.seq_to_label.values())
        lengths = np.array([len(seq) for seq in seqs], dtype=float)
        rows.append(
            {
                "dataset": bundle.name,
                "unique_sequences": len(seqs),
                "positive": int(sum(labels)),
                "negative": int(len(labels) - sum(labels)),
                "positive_ratio": float(sum(labels) / len(labels)),
                "mean_length": float(lengths.mean()),
                "median_length": float(np.median(lengths)),
                "min_length": int(lengths.min()),
                "max_length": int(lengths.max()),
            }
        )
    df = pd.DataFrame(rows)
    return df


def make_property_df(bundles: List[DatasetBundle]) -> pd.DataFrame:
    rows = []
    for bundle in bundles:
        for seq, label in bundle.seq_to_label.items():
            rows.append(
                {
                    "dataset": bundle.name,
                    "label": label,
                    "sequence": seq,
                    "length": len(seq),
                    "net_charge": net_charge(seq),
                    "hydrophobicity": mean_hydrophobicity(seq),
                    "cys_fraction": cysteine_fraction(seq),
                    "aromatic_fraction": aromatic_fraction(seq),
                }
            )
    return pd.DataFrame(rows)


def make_overlap_df(bundles: List[DatasetBundle]) -> pd.DataFrame:
    names = [bundle.name for bundle in bundles]
    sets = {bundle.name: set(bundle.sequences) for bundle in bundles}
    matrix = []
    for a in names:
        row = []
        for b in names:
            union = len(sets[a] | sets[b])
            inter = len(sets[a] & sets[b])
            row.append(inter / union if union else 0.0)
        matrix.append(row)
    return pd.DataFrame(matrix, index=names, columns=names)


def make_exact_overlap_df(bundles: List[DatasetBundle]) -> pd.DataFrame:
    names = [bundle.name for bundle in bundles]
    sets = {bundle.name: set(bundle.sequences) for bundle in bundles}
    matrix = []
    for a in names:
        row = []
        for b in names:
            inter = len(sets[a] & sets[b])
            row.append(inter)
        matrix.append(row)
    return pd.DataFrame(matrix, index=names, columns=names)


def save_summary_table(df: pd.DataFrame) -> None:
    out = OUT_DIR / "dataset_summary_table.csv"
    df.to_csv(out, index=False)


def save_property_table(df: pd.DataFrame) -> None:
    dataset_level = (
        df.groupby("dataset")
        .agg(
            mean_net_charge=("net_charge", "mean"),
            median_net_charge=("net_charge", "median"),
            mean_hydrophobicity=("hydrophobicity", "mean"),
            mean_cys_fraction=("cys_fraction", "mean"),
            mean_aromatic_fraction=("aromatic_fraction", "mean"),
        )
        .reset_index()
    )
    dataset_level.to_csv(OUT_DIR / "dataset_physchem_summary.csv", index=False)


def plot_counts(summary_df: pd.DataFrame) -> None:
    names = summary_df["dataset"].tolist()
    pos = summary_df["positive"].to_numpy()
    neg = summary_df["negative"].to_numpy()
    total = summary_df["unique_sequences"].to_numpy()
    x = np.arange(len(names))

    fig, axes = plt.subplots(1, 2, figsize=(15, 5.2), constrained_layout=True)

    colors = [DATASET_COLORS[name] for name in names]
    axes[0].bar(x, total, color=colors, edgecolor="#333333", linewidth=0.8)
    axes[0].set_title("Unique sequence scale")
    axes[0].set_ylabel("Unique sequences")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(names, rotation=25, ha="right")

    axes[1].bar(x, neg, label="Negative", color="#ABD3E1", edgecolor="#333333", linewidth=0.6)
    axes[1].bar(x, pos, bottom=neg, label="Positive", color="#EEA599", edgecolor="#333333", linewidth=0.6)
    axes[1].set_title("Class balance")
    axes[1].set_ylabel("Sequences")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(names, rotation=25, ha="right")
    axes[1].legend(frameon=False)

    fig.suptitle("Benchmark scale and class composition across peptide-toxicity datasets", fontsize=15)
    fig.savefig(OUT_DIR / "dataset_summary_counts.svg", bbox_inches="tight")
    plt.close(fig)


def plot_overlap_heatmap(overlap_df: pd.DataFrame) -> None:
    names = overlap_df.index.tolist()
    data = overlap_df.to_numpy()
    fig, ax = plt.subplots(figsize=(8.8, 7.2), constrained_layout=True)
    im = ax.imshow(data, cmap="YlGnBu", vmin=0, vmax=1)
    ax.set_xticks(np.arange(len(names)))
    ax.set_yticks(np.arange(len(names)))
    ax.set_xticklabels(names, rotation=30, ha="right")
    ax.set_yticklabels(names)
    ax.set_title("Exact-sequence Jaccard overlap between datasets")

    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            value = data[i, j]
            text_color = "white" if value > 0.55 else "#222222"
            ax.text(j, i, f"{value:.2f}", ha="center", va="center", color=text_color, fontsize=9)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Jaccard overlap")
    fig.savefig(OUT_DIR / "dataset_overlap_heatmap.svg", bbox_inches="tight")
    plt.close(fig)


def plot_length_ecdf(property_df: pd.DataFrame, names: List[str]) -> None:
    fig, ax = plt.subplots(figsize=(10.5, 6.0), constrained_layout=True)
    for name in names:
        subset = np.sort(property_df.loc[property_df["dataset"] == name, "length"].to_numpy())
        y = np.arange(1, len(subset) + 1) / len(subset)
        ax.step(subset, y, where="post", label=name, linewidth=2.0, color=DATASET_COLORS[name])
    ax.set_xlim(left=0)
    ax.set_xlabel("Peptide length (aa)")
    ax.set_ylabel("ECDF")
    ax.set_title("Length coverage across datasets")
    ax.legend(frameon=False, ncol=2)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    fig.savefig(OUT_DIR / "dataset_length_ecdf.svg", bbox_inches="tight")
    plt.close(fig)


def _draw_boxplot(ax, property_df: pd.DataFrame, names: List[str], column: str, title: str, ylabel: str) -> None:
    data = [property_df.loc[property_df["dataset"] == name, column].to_numpy() for name in names]
    bp = ax.boxplot(
        data,
        patch_artist=True,
        tick_labels=names,
        widths=0.62,
        showfliers=False,
        medianprops={"color": "#333333", "linewidth": 1.4},
        whiskerprops={"color": "#555555", "linewidth": 1.0},
        capprops={"color": "#555555", "linewidth": 1.0},
        boxprops={"edgecolor": "#444444", "linewidth": 0.9},
    )
    for patch, name in zip(bp["boxes"], names):
        patch.set_facecolor(DATASET_COLORS[name])
        patch.set_alpha(0.9)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", rotation=25)
    ax.grid(axis="y", linestyle="--", alpha=0.3)


def plot_physchem(property_df: pd.DataFrame, names: List[str]) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.4), constrained_layout=True)
    _draw_boxplot(axes[0], property_df, names, "net_charge", "Net charge distribution", "Approximate net charge")
    _draw_boxplot(
        axes[1],
        property_df,
        names,
        "hydrophobicity",
        "Mean hydrophobicity distribution",
        "Mean Kyte-Doolittle score",
    )
    _draw_boxplot(
        axes[2],
        property_df,
        names,
        "cys_fraction",
        "Cysteine fraction distribution",
        "Cys fraction",
    )
    fig.suptitle("Physicochemical coverage across datasets", fontsize=15)
    fig.savefig(OUT_DIR / "dataset_physchem_boxplots.svg", bbox_inches="tight")
    plt.close(fig)


def plot_aa_heatmap(bundles: List[DatasetBundle], names: List[str]) -> None:
    freqs = [aa_frequency(next(bundle for bundle in bundles if bundle.name == name).sequences) for name in names]
    data = np.array([[freq[aa] for aa in AA_ORDER] for freq in freqs])

    fig, ax = plt.subplots(figsize=(11.5, 5.8), constrained_layout=True)
    im = ax.imshow(data, cmap="GnBu")
    ax.set_xticks(np.arange(len(AA_ORDER)))
    ax.set_yticks(np.arange(len(names)))
    ax.set_xticklabels(AA_ORDER)
    ax.set_yticklabels(names)
    ax.set_title("Amino-acid composition across datasets")

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Frequency")
    fig.savefig(OUT_DIR / "dataset_amino_acid_composition_heatmap.svg", bbox_inches="tight")
    plt.close(fig)


def plot_overview(summary_df: pd.DataFrame, overlap_df: pd.DataFrame, property_df: pd.DataFrame, names: List[str]) -> None:
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 2, hspace=0.26, wspace=0.24)

    ax1 = fig.add_subplot(gs[0, 0])
    x = np.arange(len(names))
    total = summary_df.set_index("dataset").loc[names, "unique_sequences"].to_numpy()
    ax1.bar(x, total, color=[DATASET_COLORS[name] for name in names], edgecolor="#333333", linewidth=0.8)
    ax1.set_title("Unique sequence scale")
    ax1.set_ylabel("Unique sequences")
    ax1.set_xticks(x)
    ax1.set_xticklabels(names, rotation=25, ha="right")

    ax2 = fig.add_subplot(gs[0, 1])
    data = overlap_df.loc[names, names].to_numpy()
    im = ax2.imshow(data, cmap="YlGnBu", vmin=0, vmax=1)
    ax2.set_title("Exact-sequence Jaccard overlap")
    ax2.set_xticks(np.arange(len(names)))
    ax2.set_yticks(np.arange(len(names)))
    ax2.set_xticklabels(names, rotation=30, ha="right")
    ax2.set_yticklabels(names)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            value = data[i, j]
            color = "white" if value > 0.55 else "#222222"
            ax2.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=8, color=color)
    fig.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)

    ax3 = fig.add_subplot(gs[1, 0])
    for name in names:
        subset = np.sort(property_df.loc[property_df["dataset"] == name, "length"].to_numpy())
        y = np.arange(1, len(subset) + 1) / len(subset)
        ax3.step(subset, y, where="post", label=name, linewidth=2.0, color=DATASET_COLORS[name])
    ax3.set_title("Length coverage")
    ax3.set_xlabel("Peptide length (aa)")
    ax3.set_ylabel("ECDF")
    ax3.legend(frameon=False, ncol=2, fontsize=9)
    ax3.grid(axis="y", linestyle="--", alpha=0.35)

    ax4 = fig.add_subplot(gs[1, 1])
    property_order = names
    box_data = [property_df.loc[property_df["dataset"] == name, "cys_fraction"].to_numpy() for name in property_order]
    bp = ax4.boxplot(
        box_data,
        patch_artist=True,
        tick_labels=property_order,
        widths=0.62,
        showfliers=False,
        medianprops={"color": "#333333", "linewidth": 1.4},
    )
    for patch, name in zip(bp["boxes"], property_order):
        patch.set_facecolor(DATASET_COLORS[name])
        patch.set_edgecolor("#444444")
        patch.set_alpha(0.9)
    ax4.set_title("Cysteine-fraction distribution")
    ax4.set_ylabel("Cys fraction")
    ax4.tick_params(axis="x", rotation=25)
    ax4.grid(axis="y", linestyle="--", alpha=0.3)

    fig.suptitle("Comparative landscape of peptide-toxicity datasets", fontsize=17, y=0.98)
    fig.savefig(OUT_DIR / "dataset_comparison_overview.svg", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    plt.rcParams["font.family"] = "DejaVu Sans"
    plt.rcParams["svg.fonttype"] = "none"

    bundles = build_bundles()
    names = [bundle.name for bundle in bundles]

    summary_df = make_summary_df(bundles)
    property_df = make_property_df(bundles)
    overlap_df = make_overlap_df(bundles)
    exact_overlap_df = make_exact_overlap_df(bundles)

    save_summary_table(summary_df)
    save_property_table(property_df)
    overlap_df.to_csv(OUT_DIR / "dataset_overlap_jaccard.csv")
    exact_overlap_df.to_csv(OUT_DIR / "dataset_overlap_exact_counts.csv")

    plot_counts(summary_df)
    plot_overlap_heatmap(overlap_df)
    plot_length_ecdf(property_df, names)
    plot_physchem(property_df, names)
    plot_aa_heatmap(bundles, names)
    plot_overview(summary_df, overlap_df, property_df, names)


if __name__ == "__main__":
    main()
