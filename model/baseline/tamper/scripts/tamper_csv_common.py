#!/usr/bin/env python
from __future__ import annotations

import csv
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

os.environ.setdefault("TORCH_HOME", str(Path(__file__).resolve().parents[1] / "assets" / "torch_cache"))

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch_geometric.loader import DataLoader

TAMPER_DIR = (Path(__file__).resolve().parents[1] / "local_implementation").resolve()
TAMPER_SRC = TAMPER_DIR / "src"
sys.path.insert(0, str(TAMPER_SRC))

from dataset import ToxicityData  # noqa: E402
from tAMPer import tAMPer  # noqa: E402


env_bin = Path(sys.executable).resolve().parent
os.environ["PATH"] = str(env_bin) + os.pathsep + os.environ.get("PATH", "")

AA_ALLOWED = set("ACDEFGHIKLMNPQRSTVWY")


@dataclass
class Sample:
    sample_id: str
    original_id: str
    sequence: str
    label: int


def sanitize_sequence(sequence: str) -> str:
    seq = str(sequence).strip().upper()
    invalid = sorted(set(seq) - AA_ALLOWED)
    if invalid:
        raise ValueError(f"Unsupported amino acids {invalid} in sequence: {sequence}")
    return seq


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def resolve_device(device_str: str) -> torch.device:
    if device_str == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_split_samples(
    csv_path: Path,
    sequence_col: str,
    label_col: str,
    id_col: str,
    split_col: str,
    train_split_value: str,
    valid_split_value: str,
) -> Tuple[List[Sample], List[Sample], List[Sample]]:
    df = pd.read_csv(csv_path).reset_index(drop=True)
    required = [sequence_col, label_col, split_col]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' not found in {csv_path}.")

    all_samples: List[Sample] = []
    train_samples: List[Sample] = []
    valid_samples: List[Sample] = []

    for idx, row in df.iterrows():
        label = int(row[label_col])
        sample = Sample(
            sample_id=f"sample_{idx:05d}_{label}",
            original_id=str(row[id_col]) if id_col in df.columns else f"sample_{idx:05d}_{label}",
            sequence=sanitize_sequence(row[sequence_col]),
            label=label,
        )
        all_samples.append(sample)
        split_value = str(row[split_col])
        if split_value == str(train_split_value):
            train_samples.append(sample)
        elif split_value == str(valid_split_value):
            valid_samples.append(sample)

    if not train_samples or not valid_samples:
        raise ValueError(
            f"Split selection failed: train={len(train_samples)} valid={len(valid_samples)} "
            f"for '{train_split_value}'/'{valid_split_value}'."
        )
    return all_samples, train_samples, valid_samples


def load_all_samples(
    csv_path: Path,
    sequence_col: str,
    label_col: str,
    id_col: str,
) -> List[Sample]:
    df = pd.read_csv(csv_path).reset_index(drop=True)
    required = [sequence_col, label_col]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' not found in {csv_path}.")

    samples: List[Sample] = []
    for idx, row in df.iterrows():
        label = int(row[label_col])
        samples.append(
            Sample(
                sample_id=f"sample_{idx:05d}_{label}",
                original_id=str(row[id_col]) if id_col in df.columns else f"sample_{idx:05d}_{label}",
                sequence=sanitize_sequence(row[sequence_col]),
                label=label,
            )
        )
    return samples


def verify_pdbs(samples: Sequence[Sample], pdb_dir: Path) -> None:
    missing = [sample.sample_id for sample in samples if not (pdb_dir / f"{sample.sample_id}.pdb").exists()]
    if missing:
        preview = ", ".join(missing[:5])
        raise FileNotFoundError(f"Missing {len(missing)} PDB files under {pdb_dir}. First missing: {preview}")


def write_fasta(samples: Sequence[Sample], path: Path) -> None:
    with path.open("w", encoding="utf-8") as fout:
        for sample in samples:
            fout.write(f">{sample.sample_id}\n{sample.sequence}\n")


def write_label_fastas(samples: Sequence[Sample], output_dir: Path, prefix: str) -> Tuple[Path, Path]:
    pos_path = output_dir / f"{prefix}_pos.faa"
    neg_path = output_dir / f"{prefix}_neg.faa"
    write_fasta([sample for sample in samples if sample.label == 1], pos_path)
    write_fasta([sample for sample in samples if sample.label == 0], neg_path)
    return pos_path, neg_path


def seq_input_dim(embedding_model: str) -> int:
    if embedding_model == "t6":
        return 320
    if embedding_model == "t12":
        return 480
    if embedding_model == "t30":
        return 640
    return 1280


def build_model(
    embedding_model: str,
    hdim: int,
    gru_layers: int,
    gnn_layers: int,
    modality: str,
) -> tAMPer:
    return tAMPer(
        input_modality=modality,
        seq_input_dim=seq_input_dim(embedding_model),
        node_dims=(6, 3),
        edge_dims=(32, 1),
        node_hdim=(hdim, 16),
        edge_hdim=(32, 1),
        n_heads=8,
        gru_hdim=hdim,
        n_grus=gru_layers,
        n_gnns=gnn_layers,
    )


def build_dataset(
    samples: Sequence[Sample],
    pdb_dir: Path,
    embedding_model: str,
    device: torch.device,
    d_max: int,
    fasta_dir: Path,
    prefix: str,
):
    pos_faa, neg_faa = write_label_fastas(samples, fasta_dir, prefix)
    return ToxicityData(
        pos_seqs=str(pos_faa),
        neg_seqs=str(neg_faa),
        pdbs_path=str(pdb_dir),
        embedding_model=embedding_model,
        device=device,
        max_d=d_max,
    )


def compute_binary_metrics(labels: Sequence[int], scores: Sequence[float]) -> Dict[str, float]:
    labels_np = np.asarray(labels, dtype=np.int32)
    scores_np = np.asarray(scores, dtype=np.float64)
    preds_np = (scores_np > 0.5).astype(np.int32)
    tn, fp, fn, tp = confusion_matrix(labels_np, preds_np, labels=[0, 1]).ravel()

    sensitivity = float(recall_score(labels_np, preds_np, zero_division=0))
    specificity = float(tn) / float(tn + fp) if (tn + fp) else 0.0
    try:
        auc = float(roc_auc_score(labels_np, scores_np))
    except ValueError:
        auc = 0.0
    try:
        aupr = float(average_precision_score(labels_np, scores_np))
    except ValueError:
        aupr = 0.0

    return {
        "accuracy": float(accuracy_score(labels_np, preds_np)),
        "sensitivity": sensitivity,
        "specificity": specificity,
        "precision": float(precision_score(labels_np, preds_np, zero_division=0)),
        "f1": float(f1_score(labels_np, preds_np, zero_division=0)),
        "mcc": float(matthews_corrcoef(labels_np, preds_np)),
        "auc": auc,
        "aupr": aupr,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def evaluate_model(
    model: torch.nn.Module,
    dataset,
    device: torch.device,
    batch_size: int,
    num_workers: int,
) -> Tuple[Dict[str, float], List[Dict[str, object]]]:
    loader = DataLoader(dataset=dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)
    model.eval()
    sigmoid = torch.nn.Sigmoid()

    rows: List[Dict[str, object]] = []
    labels: List[int] = []
    scores: List[float] = []

    for graphs in loader:
        with torch.no_grad():
            graphs = graphs.to(device)
            preds = model(graphs.embeddings, graphs)
            batch_scores = sigmoid(preds["tx"].flatten()).cpu().numpy()
            batch_labels = graphs.y.cpu().numpy().astype(int)
            batch_ids = list(graphs.id)
            batch_seqs = list(graphs.seq)

        for sample_id, seq, label, score in zip(batch_ids, batch_seqs, batch_labels, batch_scores):
            score_f = float(score)
            pred = 1 if score_f > 0.5 else 0
            rows.append(
                {
                    "sample_id": sample_id,
                    "sequence": seq,
                    "label": int(label),
                    "score": score_f,
                    "prediction": pred,
                }
            )
            labels.append(int(label))
            scores.append(score_f)

    return compute_binary_metrics(labels, scores), rows


def write_predictions_csv(path: Path, rows: Iterable[Dict[str, object]]) -> None:
    rows = list(rows)
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as fout:
        writer = csv.DictWriter(fout, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
