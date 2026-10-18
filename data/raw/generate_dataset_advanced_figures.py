from __future__ import annotations

import csv
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, matthews_corrcoef, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path("/home/cns/peptide-aa/总数据集")
OUT_DIR = ROOT

DATASET_COLORS = {
    "ToxiPep 0.9": "#EEA599",
    "ToxiPep 0.8": "#FAC795",
    "CAPTP": "#E3EDE0",
    "StrucToxNet": "#ABD3E1",
    "ToxMSRC": "#92B4C8",
    "tAMPer": "#7EA6B8",
}

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


@dataclass
class SplitBenchmark:
    name: str
    train: List[Tuple[str, int]]
    tests: Dict[str, List[Tuple[str, int]]]


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


def load_toxipep_split(split: str) -> SplitBenchmark:
    train_rows: List[Tuple[str, int]] = []
    test_rows: List[Tuple[str, int]] = []
    for part, target in (("train", train_rows), ("test", test_rows)):
        path = ROOT / "ToxiPep" / "Dataset" / split / f"{part}.csv"
        with path.open("r", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                target.append((row["sequence"], int(row["toxicity"])))
    return SplitBenchmark(name=f"ToxiPep {split}", train=train_rows, tests={"test": test_rows})


def load_captp() -> SplitBenchmark:
    train = [(seq, 1 if header.endswith("|1") else 0) for header, seq in read_fasta(ROOT / "CAPTP" / "data" / "train_data.txt")]
    test1 = [(seq, 1 if header.endswith("|1") else 0) for header, seq in read_fasta(ROOT / "CAPTP" / "data" / "test1.txt")]
    test2 = [(seq, 1 if header.endswith("|1") else 0) for header, seq in read_fasta(ROOT / "CAPTP" / "data" / "test2.txt")]
    return SplitBenchmark(name="CAPTP", train=train, tests={"test1": test1, "test2": test2})


def load_structoxnet() -> SplitBenchmark:
    train = [
        (seq, 1 if header.endswith("|1") else 0)
        for header, seq in read_fasta(ROOT / "StrucToxNet" / "data" / "train_data.txt")
    ]
    test = [
        (seq, 1 if header.endswith("|1") else 0)
        for header, seq in read_fasta(ROOT / "StrucToxNet" / "data" / "test_data.txt")
    ]
    return SplitBenchmark(name="StrucToxNet", train=train, tests={"test": test})


def load_toxmsrc() -> SplitBenchmark:
    train = [
        (seq, 1 if header.endswith("|1") else 0)
        for header, seq in read_fasta(ROOT / "ToxMSRC" / "Raw data" / "train_data.fasta")
    ]
    test1 = [
        (seq, 1 if header.endswith("|1") else 0)
        for header, seq in read_fasta(ROOT / "ToxMSRC" / "Raw data" / "test1.fasta")
    ]
    test2 = [
        (seq, 1 if header.endswith("|1") else 0)
        for header, seq in read_fasta(ROOT / "ToxMSRC" / "Raw data" / "test2.fasta")
    ]
    return SplitBenchmark(name="ToxMSRC", train=train, tests={"test1": test1, "test2": test2})


def load_tamper() -> SplitBenchmark:
    train_pos = [(seq, 1) for _header, seq in read_fasta(ROOT / "tAMPer" / "data" / "sequences" / "tr_pos.faa")]
    train_neg = [(seq, 0) for _header, seq in read_fasta(ROOT / "tAMPer" / "data" / "sequences" / "tr_neg.faa")]
    val_pos = [(seq, 1) for _header, seq in read_fasta(ROOT / "tAMPer" / "data" / "sequences" / "val_pos.faa")]
    val_neg = [(seq, 0) for _header, seq in read_fasta(ROOT / "tAMPer" / "data" / "sequences" / "val_neg.faa")]
    return SplitBenchmark(name="tAMPer", train=train_pos + train_neg, tests={"val": val_pos + val_neg})


def build_benchmarks() -> List[SplitBenchmark]:
    return [
        load_toxipep_split("0.9"),
        load_toxipep_split("0.8"),
        load_captp(),
        load_structoxnet(),
        load_toxmsrc(),
        load_tamper(),
    ]


def sequence_entropy(seq: str) -> float:
    counts = Counter(seq)
    n = len(seq)
    return -sum((count / n) * math.log2(count / n) for count in counts.values())


def net_charge(seq: str) -> float:
    return seq.count("K") + seq.count("R") + 0.1 * seq.count("H") - seq.count("D") - seq.count("E")


def feature_vector(seq: str) -> List[float]:
    length = len(seq)
    hydrophobicity = float(np.mean([HYDRO[aa] for aa in seq if aa in HYDRO]))
    cys_fraction = seq.count("C") / length
    aromatic_fraction = (seq.count("F") + seq.count("W") + seq.count("Y")) / length
    return [length, net_charge(seq), hydrophobicity, cys_fraction, aromatic_fraction]


def kmer_set(seq: str, k: int) -> set[str]:
    if len(seq) < k:
        return {seq}
    return {seq[i : i + k] for i in range(len(seq) - k + 1)}


def combine_unique(records: Sequence[Tuple[str, int]]) -> Dict[str, int]:
    seq_to_label: Dict[str, int] = {}
    for seq, label in records:
        if seq not in seq_to_label:
            seq_to_label[seq] = label
    return seq_to_label


def make_split_integrity_df(benchmarks: List[SplitBenchmark]) -> pd.DataFrame:
    rows = []
    for benchmark in benchmarks:
        train_records = benchmark.train
        train_unique = combine_unique(train_records)
        test_union_records: List[Tuple[str, int]] = []
        for split_name, records in benchmark.tests.items():
            test_union_records.extend(records)
            test_unique = combine_unique(records)
            inter = set(train_unique) & set(test_unique)
            label_conflict = sum(1 for seq in inter if train_unique[seq] != test_unique[seq])
            rows.append(
                {
                    "dataset": benchmark.name,
                    "comparison": f"train vs {split_name}",
                    "train_records": len(train_records),
                    "train_unique": len(train_unique),
                    "test_records": len(records),
                    "test_unique": len(test_unique),
                    "exact_overlap": len(inter),
                    "exact_overlap_ratio": len(inter) / len(test_unique) if test_unique else 0.0,
                    "label_conflict": label_conflict,
                }
            )

        test_names = list(benchmark.tests.keys())
        if len(test_names) > 1:
            for i, name_a in enumerate(test_names):
                for name_b in test_names[i + 1 :]:
                    a_unique = combine_unique(benchmark.tests[name_a])
                    b_unique = combine_unique(benchmark.tests[name_b])
                    inter = set(a_unique) & set(b_unique)
                    label_conflict = sum(1 for seq in inter if a_unique[seq] != b_unique[seq])
                    rows.append(
                        {
                            "dataset": benchmark.name,
                            "comparison": f"{name_a} vs {name_b}",
                            "train_records": len(benchmark.tests[name_a]),
                            "train_unique": len(a_unique),
                            "test_records": len(benchmark.tests[name_b]),
                            "test_unique": len(b_unique),
                            "exact_overlap": len(inter),
                            "exact_overlap_ratio": len(inter) / len(b_unique) if b_unique else 0.0,
                            "label_conflict": label_conflict,
                        }
                    )
    return pd.DataFrame(rows)


def make_combined_records_df(benchmarks: List[SplitBenchmark]) -> pd.DataFrame:
    rows = []
    for benchmark in benchmarks:
        all_records = list(benchmark.train)
        for records in benchmark.tests.values():
            all_records.extend(records)
        for seq, label in all_records:
            rows.append(
                {
                    "dataset": benchmark.name,
                    "sequence": seq,
                    "label": label,
                    "length": len(seq),
                    "entropy": sequence_entropy(seq),
                    "net_charge": net_charge(seq),
                }
            )
    return pd.DataFrame(rows)


def unique_sequence_map(records_df: pd.DataFrame, dataset: str) -> Dict[str, int]:
    subset = records_df.loc[records_df["dataset"] == dataset, ["sequence", "label"]].drop_duplicates()
    return {row.sequence: int(row.label) for row in subset.itertuples(index=False)}


def make_diversity_df(records_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for dataset in records_df["dataset"].unique():
        seq_to_label = unique_sequence_map(records_df, dataset)
        seqs = list(seq_to_label.keys())
        seqs_task = [seq for seq in seqs if len(seq) <= 50]
        tri = set()
        tetra = set()
        length_bins = set()
        charge_bins = set()
        joint_cells = set()
        for seq in seqs:
            tri |= kmer_set(seq, 3)
            tetra |= kmer_set(seq, 4)
        for seq in seqs_task:
            lbin = min((len(seq) - 1) // 5, 9)
            cbin = max(-2, min(9, math.floor((net_charge(seq) + 8) / 2)))
            length_bins.add(lbin)
            charge_bins.add(cbin)
            joint_cells.add((lbin, cbin))

        labels = list(seq_to_label.values())
        pos_ratio = sum(labels) / len(labels)
        rows.append(
            {
                "dataset": dataset,
                "unique_sequences": len(seqs),
                "task_length_alignment": len(seqs_task) / len(seqs) if seqs else 0.0,
                "balance_score": 1.0 - abs(pos_ratio - 0.5) / 0.5,
                "mean_entropy": float(np.mean([sequence_entropy(seq) for seq in seqs])),
                "tripeptide_count": len(tri),
                "tetrapeptide_count": len(tetra),
                "task_length_bin_occupancy": len(length_bins),
                "task_charge_bin_occupancy": len(charge_bins),
                "task_length_charge_cell_occupancy": len(joint_cells),
            }
        )
    return pd.DataFrame(rows)


def nearest_kmer_jaccard(test_seq: str, train_ksets: List[set[str]], inv_index: Dict[str, List[int]]) -> float:
    query = kmer_set(test_seq, 3)
    candidates: set[int] = set()
    for kmer in query:
        candidates.update(inv_index.get(kmer, []))
    if not candidates:
        return 0.0
    best = 0.0
    for idx in candidates:
        candidate = train_ksets[idx]
        inter = len(query & candidate)
        union = len(query | candidate)
        score = inter / union if union else 0.0
        if score > best:
            best = score
    return best


def make_novelty_df(benchmarks: List[SplitBenchmark]) -> pd.DataFrame:
    rows = []
    for benchmark in benchmarks:
        train_ksets = [kmer_set(seq, 3) for seq, _label in benchmark.train]
        inv_index: Dict[str, List[int]] = defaultdict(list)
        for idx, seq_kset in enumerate(train_ksets):
            for kmer in seq_kset:
                inv_index[kmer].append(idx)

        for split_name, records in benchmark.tests.items():
            scores = [nearest_kmer_jaccard(seq, train_ksets, inv_index) for seq, _label in records]
            rows.extend(
                {
                    "dataset": benchmark.name,
                    "test_split": split_name,
                    "score": score,
                }
                for score in scores
            )
    return pd.DataFrame(rows)


def make_shortcut_df(benchmarks: List[SplitBenchmark]) -> pd.DataFrame:
    rows = []
    for benchmark in benchmarks:
        X_train = np.array([feature_vector(seq) for seq, _label in benchmark.train], dtype=float)
        y_train = np.array([label for _seq, label in benchmark.train], dtype=int)

        for split_name, records in benchmark.tests.items():
            X_test = np.array([feature_vector(seq) for seq, _label in records], dtype=float)
            y_test = np.array([label for _seq, label in records], dtype=int)

            model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000))
            model.fit(X_train, y_train)
            probs = model.predict_proba(X_test)[:, 1]
            preds = (probs >= 0.5).astype(int)

            cys_model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000))
            cys_model.fit(X_train[:, [3]], y_train)
            cys_probs = cys_model.predict_proba(X_test[:, [3]])[:, 1]
            cys_preds = (cys_probs >= 0.5).astype(int)

            ks_stats = [ks_2samp(X_train[:, i], X_test[:, i]).statistic for i in range(X_train.shape[1])]

            rows.append(
                {
                    "dataset": benchmark.name,
                    "test_split": split_name,
                    "simple_auc": roc_auc_score(y_test, probs),
                    "simple_mcc": matthews_corrcoef(y_test, preds),
                    "simple_bacc": balanced_accuracy_score(y_test, preds),
                    "cys_only_auc": roc_auc_score(y_test, cys_probs),
                    "cys_only_mcc": matthews_corrcoef(y_test, cys_preds),
                    "mean_feature_ks": float(np.mean(ks_stats)),
                    "length_ks": ks_stats[0],
                    "charge_ks": ks_stats[1],
                    "hydrophobicity_ks": ks_stats[2],
                    "cys_fraction_ks": ks_stats[3],
                    "aromatic_fraction_ks": ks_stats[4],
                }
            )
    return pd.DataFrame(rows)


def save_tables(split_df: pd.DataFrame, diversity_df: pd.DataFrame, novelty_df: pd.DataFrame, shortcut_df: pd.DataFrame) -> None:
    split_df.to_csv(OUT_DIR / "dataset_split_integrity_table.csv", index=False)
    diversity_df.to_csv(OUT_DIR / "dataset_diversity_metrics_table.csv", index=False)
    novelty_summary = (
        novelty_df.groupby(["dataset", "test_split"])["score"]
        .agg(["mean", "median", lambda x: float(np.quantile(x, 0.9))])
        .reset_index()
        .rename(columns={"<lambda_0>": "p90"})
    )
    novelty_summary.to_csv(OUT_DIR / "dataset_train_test_novelty_summary.csv", index=False)
    shortcut_df.to_csv(OUT_DIR / "dataset_shortcut_baseline_table.csv", index=False)


def plot_length_bin_balance(records_df: pd.DataFrame, names: List[str]) -> None:
    bins = [(1, 10), (11, 20), (21, 30), (31, 40), (41, 50), (51, 100)]
    labels = [f"{lo}-{hi}" for lo, hi in bins]
    fig, ax = plt.subplots(figsize=(10.8, 6.2), constrained_layout=True)
    for name in names:
        ys = []
        subset = records_df.loc[records_df["dataset"] == name]
        for lo, hi in bins:
            bin_df = subset[(subset["length"] >= lo) & (subset["length"] <= hi)]
            if len(bin_df) == 0:
                ys.append(np.nan)
            else:
                ys.append(bin_df["label"].mean())
        ax.plot(labels, ys, marker="o", linewidth=2.2, color=DATASET_COLORS[name], label=name)
    ax.axhline(0.5, color="#666666", linestyle="--", linewidth=1.0)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Positive-class ratio")
    ax.set_xlabel("Length bin (aa)")
    ax.set_title("Class balance across peptide-length subgroups")
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.legend(frameon=False, ncol=2)
    fig.savefig(OUT_DIR / "dataset_balance_by_length_bin.svg", bbox_inches="tight")
    plt.close(fig)


def plot_complexity_boxplot(records_df: pd.DataFrame, names: List[str]) -> None:
    fig, ax = plt.subplots(figsize=(10.6, 6.0), constrained_layout=True)
    data = [
        records_df.loc[records_df["dataset"] == name, ["sequence", "entropy"]]
        .drop_duplicates(subset=["sequence"])["entropy"]
        .to_numpy()
        for name in names
    ]
    bp = ax.boxplot(
        data,
        patch_artist=True,
        tick_labels=names,
        widths=0.6,
        showfliers=False,
        medianprops={"color": "#333333", "linewidth": 1.4},
        boxprops={"edgecolor": "#444444", "linewidth": 0.9},
        whiskerprops={"color": "#555555", "linewidth": 1.0},
        capprops={"color": "#555555", "linewidth": 1.0},
    )
    for patch, name in zip(bp["boxes"], names):
        patch.set_facecolor(DATASET_COLORS[name])
        patch.set_alpha(0.9)
    ax.set_ylabel("Per-sequence Shannon entropy")
    ax.set_title("Sequence complexity distribution")
    ax.tick_params(axis="x", rotation=25)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    fig.savefig(OUT_DIR / "dataset_complexity_boxplots.svg", bbox_inches="tight")
    plt.close(fig)


def plot_motif_diversity(diversity_df: pd.DataFrame, names: List[str]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13.6, 5.3), constrained_layout=True)
    x = np.arange(len(names))
    tri = diversity_df.set_index("dataset").loc[names, "tripeptide_count"].to_numpy()
    tetra = diversity_df.set_index("dataset").loc[names, "tetrapeptide_count"].to_numpy()
    colors = [DATASET_COLORS[name] for name in names]

    axes[0].bar(x, tri, color=colors, edgecolor="#333333", linewidth=0.8)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(names, rotation=25, ha="right")
    axes[0].set_ylabel("Unique 3-mers")
    axes[0].set_title("Tripeptide motif coverage")

    axes[1].bar(x, tetra, color=colors, edgecolor="#333333", linewidth=0.8)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(names, rotation=25, ha="right")
    axes[1].set_ylabel("Unique 4-mers")
    axes[1].set_title("Tetrapeptide motif coverage")

    fig.suptitle("Motif-space diversity across datasets", fontsize=15)
    fig.savefig(OUT_DIR / "dataset_motif_diversity.svg", bbox_inches="tight")
    plt.close(fig)


def plot_novelty_boxplot(novelty_df: pd.DataFrame, names: List[str]) -> None:
    fig, ax = plt.subplots(figsize=(10.8, 6.0), constrained_layout=True)
    data = [novelty_df.loc[novelty_df["dataset"] == name, "score"].to_numpy() for name in names]
    bp = ax.boxplot(
        data,
        patch_artist=True,
        tick_labels=names,
        widths=0.62,
        showfliers=False,
        medianprops={"color": "#333333", "linewidth": 1.4},
        boxprops={"edgecolor": "#444444", "linewidth": 0.9},
        whiskerprops={"color": "#555555", "linewidth": 1.0},
        capprops={"color": "#555555", "linewidth": 1.0},
    )
    for patch, name in zip(bp["boxes"], names):
        patch.set_facecolor(DATASET_COLORS[name])
        patch.set_alpha(0.9)
    ax.set_ylabel("Nearest-train 3-mer Jaccard similarity")
    ax.set_title("Train-test neighborhood similarity")
    ax.tick_params(axis="x", rotation=25)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    fig.savefig(OUT_DIR / "dataset_nearest_3mer_similarity.svg", bbox_inches="tight")
    plt.close(fig)


def plot_split_integrity(split_df: pd.DataFrame) -> None:
    rows = split_df.copy()
    rows["row_name"] = rows["dataset"] + " | " + rows["comparison"]
    y = np.arange(len(rows))
    fig, axes = plt.subplots(1, 2, figsize=(14.2, 5.8), constrained_layout=True)

    axes[0].barh(y, rows["exact_overlap_ratio"], color="#92B4C8", edgecolor="#333333", linewidth=0.7)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(rows["row_name"])
    axes[0].invert_yaxis()
    axes[0].set_xlabel("Exact-overlap ratio")
    axes[0].set_title("Split leakage / subset issue")
    axes[0].grid(axis="x", linestyle="--", alpha=0.3)

    axes[1].barh(y, rows["label_conflict"], color="#EEA599", edgecolor="#333333", linewidth=0.7)
    axes[1].set_yticks(y)
    axes[1].set_yticklabels(rows["row_name"])
    axes[1].invert_yaxis()
    axes[1].set_xlabel("Label conflicts on overlapping sequences")
    axes[1].set_title("Split label inconsistency")
    axes[1].grid(axis="x", linestyle="--", alpha=0.3)

    fig.suptitle("Split-integrity diagnostics across benchmarks", fontsize=15)
    fig.savefig(OUT_DIR / "dataset_split_integrity_overview.svg", bbox_inches="tight")
    plt.close(fig)


def plot_shortcut_baseline(shortcut_df: pd.DataFrame, names: List[str]) -> None:
    shortcut_summary = shortcut_df.groupby("dataset").mean(numeric_only=True)
    auc_vals = shortcut_summary.loc[names, "simple_auc"].to_numpy()
    mcc_vals = shortcut_summary.loc[names, "simple_mcc"].to_numpy()
    cys_auc = shortcut_summary.loc[names, "cys_only_auc"].to_numpy()

    x = np.arange(len(names))
    width = 0.36
    fig, axes = plt.subplots(1, 2, figsize=(14.2, 5.6), constrained_layout=True)

    axes[0].bar(x - width / 2, auc_vals, width=width, color="#92B4C8", edgecolor="#333333", linewidth=0.7, label="Simple-feature AUC")
    axes[0].bar(x + width / 2, mcc_vals, width=width, color="#EEA599", edgecolor="#333333", linewidth=0.7, label="Simple-feature MCC")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(names, rotation=25, ha="right")
    axes[0].set_ylim(0, 1)
    axes[0].set_title("Performance of a simple physicochemical baseline")
    axes[0].grid(axis="y", linestyle="--", alpha=0.3)
    axes[0].legend(frameon=False)

    axes[1].bar(x, cys_auc, color=[DATASET_COLORS[name] for name in names], edgecolor="#333333", linewidth=0.7)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(names, rotation=25, ha="right")
    axes[1].set_ylim(0, 1)
    axes[1].set_title("Single-feature shortcut strength: Cys fraction only")
    axes[1].set_ylabel("AUC")
    axes[1].grid(axis="y", linestyle="--", alpha=0.3)

    fig.suptitle("Shortcut susceptibility across peptide-toxicity benchmarks", fontsize=15)
    fig.savefig(OUT_DIR / "dataset_shortcut_baseline.svg", bbox_inches="tight")
    plt.close(fig)


def plot_hard_but_fair(shortcut_df: pd.DataFrame, novelty_df: pd.DataFrame, names: List[str]) -> None:
    shortcut_summary = shortcut_df.groupby("dataset").mean(numeric_only=True)
    novelty_summary = novelty_df.groupby("dataset")["score"].mean().rename("mean_similarity")
    merged = shortcut_summary.join(novelty_summary)
    merged["novelty"] = 1.0 - merged["mean_similarity"]

    fig, ax = plt.subplots(figsize=(8.8, 6.4), constrained_layout=True)
    for name in names:
        x = merged.loc[name, "mean_feature_ks"]
        y = merged.loc[name, "novelty"]
        ax.scatter(x, y, s=130, color=DATASET_COLORS[name], edgecolor="#333333", linewidth=0.9)
        ax.text(x + 0.0008, y + 0.003, name, fontsize=10, color="#222222")

    ax.set_xlabel("Train-test covariate shift (mean KS on simple features)")
    ax.set_ylabel("Sequence novelty score (1 - nearest-train 3-mer similarity)")
    ax.set_title("Hard but fair benchmark regime")
    ax.grid(linestyle="--", alpha=0.3)
    fig.savefig(OUT_DIR / "dataset_hard_but_fair_scatter.svg", bbox_inches="tight")
    plt.close(fig)


def plot_scorecard(diversity_df: pd.DataFrame, split_df: pd.DataFrame, novelty_df: pd.DataFrame, names: List[str]) -> None:
    integrity_summary = (
        split_df[split_df["comparison"].str.startswith("train vs")]
        .groupby("dataset")
        .agg(
            exact_overlap_ratio=("exact_overlap_ratio", "max"),
            label_conflict=("label_conflict", "sum"),
        )
    )
    novelty_summary = novelty_df.groupby("dataset")["score"].mean().rename("mean_similarity")
    score_df = diversity_df.set_index("dataset").join(integrity_summary).join(novelty_summary)

    score_df["split_hygiene"] = 1.0 - score_df["exact_overlap_ratio"]
    score_df["novelty_score"] = 1.0 - score_df["mean_similarity"]
    score_df["conflict_free"] = 1.0 / (1.0 + score_df["label_conflict"])

    display = score_df.loc[
        names,
        [
            "task_length_alignment",
            "balance_score",
            "task_length_bin_occupancy",
            "task_charge_bin_occupancy",
            "task_length_charge_cell_occupancy",
            "mean_entropy",
            "tripeptide_count",
            "tetrapeptide_count",
            "split_hygiene",
            "novelty_score",
            "conflict_free",
        ],
    ].copy()

    cols_to_scale = [
        "task_length_bin_occupancy",
        "task_charge_bin_occupancy",
        "task_length_charge_cell_occupancy",
        "mean_entropy",
        "tripeptide_count",
        "tetrapeptide_count",
    ]
    for col in cols_to_scale:
        values = display[col].to_numpy(dtype=float)
        min_v = values.min()
        max_v = values.max()
        if max_v > min_v:
            display[col] = (values - min_v) / (max_v - min_v)
        else:
            display[col] = 1.0

    pretty_cols = [
        "Task-aligned",
        "Balance",
        "Length bins",
        "Charge bins",
        "Length-charge cells",
        "Complexity",
        "3-mer diversity",
        "4-mer diversity",
        "Split hygiene",
        "Novelty",
        "Conflict-free",
    ]
    display.columns = pretty_cols

    data = display.to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(12.5, 6.4), constrained_layout=True)
    im = ax.imshow(data, cmap="YlGnBu", vmin=0, vmax=1)
    ax.set_xticks(np.arange(len(pretty_cols)))
    ax.set_xticklabels(pretty_cols, rotation=30, ha="right")
    ax.set_yticks(np.arange(len(names)))
    ax.set_yticklabels(names)
    ax.set_title("Multidimensional benchmark scorecard (higher is better)")

    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            value = data[i, j]
            color = "white" if value > 0.58 else "#222222"
            ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=8, color=color)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Normalized score")
    fig.savefig(OUT_DIR / "dataset_multidimensional_scorecard.svg", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    plt.rcParams["font.family"] = "DejaVu Sans"
    plt.rcParams["svg.fonttype"] = "none"

    benchmarks = build_benchmarks()
    names = [benchmark.name for benchmark in benchmarks]

    split_df = make_split_integrity_df(benchmarks)
    records_df = make_combined_records_df(benchmarks)
    diversity_df = make_diversity_df(records_df)
    novelty_df = make_novelty_df(benchmarks)
    shortcut_df = make_shortcut_df(benchmarks)

    save_tables(split_df, diversity_df, novelty_df, shortcut_df)
    plot_length_bin_balance(records_df, names)
    plot_complexity_boxplot(records_df, names)
    plot_motif_diversity(diversity_df, names)
    plot_novelty_boxplot(novelty_df, names)
    plot_split_integrity(split_df)
    plot_shortcut_baseline(shortcut_df, names)
    plot_hard_but_fair(shortcut_df, novelty_df, names)
    plot_scorecard(diversity_df, split_df, novelty_df, names)


if __name__ == "__main__":
    main()
