#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    recall_score,
    roc_auc_score,
)
from torch_geometric.loader import DataLoader

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from model.StrucToxNet import EGNNModel  # noqa: E402
from scripts.evaluate_csv import (  # noqa: E402
    DEFAULT_MKDSSP,
    PREPROCESS_DIR,
    Sample,
    build_dataset,
    build_dssp_and_contacts,
    build_plm_cache,
    ensure_dir,
    generate_pdbs,
    resolve_device,
    sanitize_sequence,
    write_fasta,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train StrucToxNet directly from a labeled CSV with train/validation splits."
    )
    parser.add_argument("--input-csv", required=True, help="Input CSV containing train/validation rows.")
    parser.add_argument("--sequence-col", default="sequence")
    parser.add_argument("--label-col", default="toxicity")
    parser.add_argument("--id-col", default="id")
    parser.add_argument("--split-col", default="split")
    parser.add_argument("--train-split-value", default="train")
    parser.add_argument("--valid-split-value", default="validation")
    parser.add_argument("--run-dir", required=True, help="Cache directory for PDB/DSSP/pdb2rr/plm files.")
    parser.add_argument("--output-dir", required=True, help="Directory to save checkpoints and logs.")
    parser.add_argument("--prot-t5-model", default=str(PROJECT_DIR / "models" / "prot_t5_xl_uniref50"))
    parser.add_argument("--esmfold-model", default=str(PROJECT_DIR / "models" / "esmfold_v1"))
    parser.add_argument("--mkdssp", default=DEFAULT_MKDSSP)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--train-batch-size", type=int, default=256)
    parser.add_argument("--eval-batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--num-layer", type=int, default=3)
    parser.add_argument("--dim-embedding", type=int, default=256)
    parser.add_argument("--in-edge-nf", type=int, default=16)
    parser.add_argument("--select-metric", default="mcc", choices=["accuracy", "sensitivity", "specificity", "f1", "mcc", "auc", "aupr"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--esmfold-batch-size", type=int, default=4)
    parser.add_argument("--esmfold-chunk-size", type=int, default=64)
    parser.add_argument("--prot-t5-batch-size", type=int, default=8)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_split_samples(args: argparse.Namespace) -> Tuple[List[Sample], List[Sample], List[Sample]]:
    df = pd.read_csv(args.input_csv).reset_index(drop=True)
    required = [args.sequence_col, args.label_col, args.split_col]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' not found in {args.input_csv}. Available columns: {list(df.columns)}")

    all_samples: List[Sample] = []
    train_samples: List[Sample] = []
    valid_samples: List[Sample] = []
    for idx, row in df.iterrows():
        label = int(row[args.label_col])
        sample = Sample(
            sample_id=f"sample_{idx:05d}_{label}",
            original_id=str(row[args.id_col]) if args.id_col in df.columns else f"sample_{idx:05d}_{label}",
            sequence=sanitize_sequence(row[args.sequence_col]),
            label=label,
        )
        all_samples.append(sample)
        split_value = str(row[args.split_col])
        if split_value == str(args.train_split_value):
            train_samples.append(sample)
        elif split_value == str(args.valid_split_value):
            valid_samples.append(sample)

    if not train_samples or not valid_samples:
        raise ValueError(
            f"Split selection failed. train={len(train_samples)} valid={len(valid_samples)} "
            f"for split column '{args.split_col}' values '{args.train_split_value}'/'{args.valid_split_value}'."
        )
    return all_samples, train_samples, valid_samples


def model_config(args: argparse.Namespace, device: torch.device) -> SimpleNamespace:
    return SimpleNamespace(
        in_edge_nf=args.in_edge_nf,
        num_layer=args.num_layer,
        dim_embedding=args.dim_embedding,
        device=str(device),
        dropout=args.dropout,
    )


def metric_specificity(labels: np.ndarray, predictions: np.ndarray) -> Tuple[float, int, int, int, int]:
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    specificity = float(tn) / float(tn + fp) if (tn + fp) else 0.0
    return specificity, int(tn), int(fp), int(fn), int(tp)


def metric_auc(labels: np.ndarray, probabilities: np.ndarray) -> float:
    try:
        return float(roc_auc_score(labels, probabilities))
    except ValueError:
        return 0.0


def metric_aupr(labels: np.ndarray, probabilities: np.ndarray) -> float:
    try:
        return float(average_precision_score(labels, probabilities))
    except ValueError:
        return 0.0


def evaluate_split(model: EGNNModel, dataset, device: torch.device, batch_size: int) -> Dict[str, float]:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=0)
    model.eval()
    losses = []
    labels_all = []
    predictions_all = []
    probabilities_all = []

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            labels = batch.label.long().to(device)
            logits = model.get_logits(batch)
            loss = F.cross_entropy(logits, labels)
            probabilities = F.softmax(logits, dim=1)[:, 1]
            predictions = torch.argmax(logits, dim=1)

            losses.append(float(loss.item()) * int(labels.shape[0]))
            labels_all.append(labels.cpu().numpy())
            predictions_all.append(predictions.cpu().numpy())
            probabilities_all.append(probabilities.cpu().numpy())

    labels_np = np.concatenate(labels_all).astype(int)
    predictions_np = np.concatenate(predictions_all).astype(int)
    probabilities_np = np.concatenate(probabilities_all).astype(np.float64)
    specificity, tn, fp, fn, tp = metric_specificity(labels_np, predictions_np)
    denom = len(labels_np) if len(labels_np) else 1
    return {
        "loss": float(sum(losses) / denom),
        "accuracy": float(accuracy_score(labels_np, predictions_np)),
        "sensitivity": float(recall_score(labels_np, predictions_np, zero_division=0)),
        "specificity": specificity,
        "f1": float(f1_score(labels_np, predictions_np, zero_division=0)),
        "mcc": float(matthews_corrcoef(labels_np, predictions_np)),
        "auc": metric_auc(labels_np, probabilities_np),
        "aupr": metric_aupr(labels_np, probabilities_np),
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }


def train_one_epoch(model: EGNNModel, dataset, optimizer: torch.optim.Optimizer, device: torch.device, batch_size: int) -> float:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=0)
    model.train()
    loss_total = 0.0
    count = 0
    for batch in loader:
        batch = batch.to(device)
        labels = batch.label.long().to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model.get_logits(batch)
        loss = F.cross_entropy(logits, labels)
        loss.backward()
        optimizer.step()
        loss_total += float(loss.item()) * int(labels.shape[0])
        count += int(labels.shape[0])
    return float(loss_total / max(count, 1))


def write_history(path: Path, rows: List[Dict[str, float]]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as fout:
        writer = csv.DictWriter(fout, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)
    run_dir = ensure_dir(Path(args.run_dir).resolve())
    output_dir = ensure_dir(Path(args.output_dir).resolve())
    fasta_path = run_dir / "input.fasta"
    pdb_dir = ensure_dir(run_dir / "pdb")
    raw_dssp_dir = ensure_dir(run_dir / "dssp_raw")
    dssp_dir = ensure_dir(run_dir / "dssp")
    pdb2rr_dir = ensure_dir(run_dir / "pdb2rr")
    plm_dir = ensure_dir(run_dir / "plm")

    all_samples, train_samples, valid_samples = load_split_samples(args)
    write_fasta(all_samples, fasta_path)

    print(f"device={device}")
    print(f"train_rows={len(train_samples)}")
    print(f"valid_rows={len(valid_samples)}")

    generate_pdbs(
        samples=all_samples,
        pdb_dir=pdb_dir,
        model_path=Path(args.esmfold_model),
        device=device,
        batch_size=args.esmfold_batch_size,
        chunk_size=args.esmfold_chunk_size,
    )
    build_dssp_and_contacts(all_samples, args, pdb_dir, raw_dssp_dir, dssp_dir, pdb2rr_dir)
    build_plm_cache(
        samples=all_samples,
        plm_dir=plm_dir,
        model_path=Path(args.prot_t5_model),
        x_min_path=PREPROCESS_DIR / "inputs" / "x_min.npy",
        x_max_path=PREPROCESS_DIR / "inputs" / "x_max.npy",
        device=device,
        batch_size=args.prot_t5_batch_size,
    )

    train_dataset = build_dataset(train_samples, pdb_dir, dssp_dir, pdb2rr_dir, plm_dir)
    valid_dataset = build_dataset(valid_samples, pdb_dir, dssp_dir, pdb2rr_dir, plm_dir)

    model = EGNNModel(model_config(args, device)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    history: List[Dict[str, float]] = []
    best_value = float("-inf")
    best_epoch = -1
    best_metrics: Dict[str, float] | None = None
    best_checkpoint = output_dir / "best_valid_model.pth"

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_dataset, optimizer, device, args.train_batch_size)
        valid_metrics = evaluate_split(model, valid_dataset, device, args.eval_batch_size)

        row: Dict[str, float] = {"epoch": epoch, "train_loss": train_loss}
        for key, value in valid_metrics.items():
            row[f"valid_{key}"] = value
        history.append(row)

        epoch_path = output_dir / f"model_epoch_{epoch}.pth"
        torch.save(model.state_dict(), epoch_path)

        score = float(valid_metrics[args.select_metric])
        improved = score > best_value
        if improved:
            best_value = score
            best_epoch = epoch
            best_metrics = valid_metrics
            torch.save(model.state_dict(), best_checkpoint)

        print(
            f"epoch={epoch} train_loss={train_loss:.6f} "
            f"valid_accuracy={valid_metrics['accuracy']:.6f} "
            f"valid_sensitivity={valid_metrics['sensitivity']:.6f} "
            f"valid_specificity={valid_metrics['specificity']:.6f} "
            f"valid_f1={valid_metrics['f1']:.6f} "
            f"valid_mcc={valid_metrics['mcc']:.6f} "
            f"select_metric={args.select_metric}:{score:.6f} "
            f"best_epoch={best_epoch}"
        )

    history_path = output_dir / "training_history.csv"
    write_history(history_path, history)

    summary = {
        "input_csv": str(Path(args.input_csv).resolve()),
        "run_dir": str(run_dir),
        "output_dir": str(output_dir),
        "train_rows": len(train_samples),
        "valid_rows": len(valid_samples),
        "select_metric": args.select_metric,
        "best_epoch": best_epoch,
        "best_checkpoint": str(best_checkpoint),
        "best_valid_metrics": best_metrics,
    }
    summary_path = output_dir / "training_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"best_epoch={best_epoch}")
    print(f"best_checkpoint={best_checkpoint}")
    print(f"training_history={history_path}")
    print(f"training_summary={summary_path}")


if __name__ == "__main__":
    main()
