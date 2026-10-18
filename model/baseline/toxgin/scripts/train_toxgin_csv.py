#!/usr/bin/env python
"""Train ToxGIN on a labeled CSV with explicit train/validation splits."""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
)
from torch.utils.data import DataLoader, Subset

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from scripts.evaluate_toxgin import (
    Sample,
    ToxGINDataset,
    GNNModel,
    build_graphs_and_features,
    collate_fn,
    resolve_device,
    sanitize_sequence,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", required=True, help="CSV containing train/validation rows.")
    parser.add_argument("--pdb-dir", required=True, help="Directory with sample_id-named PDB files.")
    parser.add_argument("--run-dir", required=True, help="Directory for generated ToxGIN graph/feature cache.")
    parser.add_argument("--output-dir", required=True, help="Directory to save checkpoints and logs.")
    parser.add_argument(
        "--aaindex",
        default=str(Path(__file__).resolve().parents[1] / "aaindex1.csv"),
        help="Path to ToxGIN aaindex1.csv.",
    )
    parser.add_argument(
        "--esm-model",
        default="facebook/esm2_t36_3B_UR50D",
        help="Hugging Face model id or local path for the ESM2 encoder.",
    )
    parser.add_argument("--sequence-col", default="sequence")
    parser.add_argument("--label-col", default="toxicity")
    parser.add_argument("--id-col", default="id")
    parser.add_argument("--split-col", default="split")
    parser.add_argument("--train-split-value", default="train")
    parser.add_argument("--valid-split-value", default="validation")
    parser.add_argument("--plm-device", default="auto", help="Device for ESM2 feature extraction.")
    parser.add_argument("--gnn-device", default="auto", help="Device for ToxGIN training and validation.")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--hidden-dim", type=int, default=1024)
    parser.add_argument("--max-length", type=int, default=60)
    parser.add_argument("--threshold", type=float, default=8.0)
    parser.add_argument(
        "--select-metric",
        default="accuracy",
        choices=["accuracy", "sensitivity", "specificity", "f1", "mcc", "auc", "aupr"],
    )
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--save-every-epoch", action="store_true")
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


def load_split_samples(args: argparse.Namespace) -> tuple[List[Sample], List[Sample], List[Sample]]:
    df = pd.read_csv(args.input_csv).reset_index(drop=True)
    required = [args.sequence_col, args.label_col, args.split_col]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' not found in {args.input_csv}.")

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
            f"Split selection failed: train={len(train_samples)} valid={len(valid_samples)} "
            f"for column '{args.split_col}' values '{args.train_split_value}'/'{args.valid_split_value}'."
        )
    return all_samples, train_samples, valid_samples


def compute_binary_metrics(
    labels: np.ndarray,
    scores: np.ndarray,
    average_loss: float,
) -> Dict[str, float]:
    predictions = (scores >= 0.5).astype(np.int32)
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()

    sensitivity = float(tp) / float(tp + fn) if (tp + fn) else 0.0
    specificity = float(tn) / float(tn + fp) if (tn + fp) else 0.0

    try:
        auc = float(roc_auc_score(labels, scores))
    except ValueError:
        auc = 0.0

    try:
        aupr = float(average_precision_score(labels, scores))
    except ValueError:
        aupr = 0.0

    return {
        "loss": float(average_loss),
        "accuracy": float(accuracy_score(labels, predictions)),
        "sensitivity": sensitivity,
        "specificity": specificity,
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "mcc": float(matthews_corrcoef(labels, predictions)),
        "auc": auc,
        "aupr": aupr,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def evaluate(model: GNNModel, dataloader: DataLoader, device: torch.device) -> Dict[str, float]:
    criterion = torch.nn.BCELoss(reduction="sum")
    model.eval()
    labels_all = []
    scores_all = []
    loss_sum = 0.0
    count = 0

    with torch.no_grad():
        for graph, samples in dataloader:
            graph = graph.to(device)
            labels = torch.tensor([sample.label for sample in samples], dtype=torch.float32, device=device)
            scores = model(graph, graph.ndata["wildtype_seq"]).squeeze(1)
            loss = criterion(scores, labels)

            batch_size = int(labels.shape[0])
            loss_sum += float(loss.item())
            count += batch_size
            labels_all.append(labels.detach().cpu().numpy().astype(np.int32))
            scores_all.append(scores.detach().cpu().numpy().astype(np.float32))

    labels_np = np.concatenate(labels_all)
    scores_np = np.concatenate(scores_all)
    return compute_binary_metrics(labels_np, scores_np, average_loss=float(loss_sum / max(count, 1)))


def train_one_epoch(
    model: GNNModel,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    criterion = torch.nn.BCELoss(reduction="sum")
    model.train()
    loss_sum = 0.0
    count = 0

    for graph, samples in dataloader:
        graph = graph.to(device)
        labels = torch.tensor([sample.label for sample in samples], dtype=torch.float32, device=device)
        scores = model(graph, graph.ndata["wildtype_seq"]).squeeze(1)
        loss = criterion(scores, labels)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        batch_size = int(labels.shape[0])
        loss_sum += float(loss.item())
        count += batch_size

    return float(loss_sum / max(count, 1))


def write_history(path: Path, rows: Sequence[Dict[str, float]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as fout:
        writer = csv.DictWriter(fout, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    plm_device = resolve_device(args.plm_device)
    gnn_device = resolve_device(args.gnn_device)
    run_dir = Path(args.run_dir).resolve()
    data_dir = run_dir / "toxgin_data"
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    all_samples, train_samples, valid_samples = load_split_samples(args)

    print(f"plm_device={plm_device}")
    print(f"gnn_device={gnn_device}")
    print(f"all_rows={len(all_samples)}")
    print(f"train_rows={len(train_samples)}")
    print(f"valid_rows={len(valid_samples)}")

    build_graphs_and_features(
        samples=all_samples,
        pdb_dir=Path(args.pdb_dir),
        data_dir=data_dir,
        aaindex_path=Path(args.aaindex),
        esm_model=args.esm_model,
        plm_device=plm_device,
        max_length=args.max_length,
        threshold=args.threshold,
    )

    dataset = ToxGINDataset(all_samples, data_dir)
    sample_to_index = {sample.sample_id: idx for idx, sample in enumerate(all_samples)}
    train_dataset = Subset(dataset, [sample_to_index[sample.sample_id] for sample in train_samples])
    valid_dataset = Subset(dataset, [sample_to_index[sample.sample_id] for sample in valid_samples])

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn, num_workers=0)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn, num_workers=0)

    model = GNNModel(in_dim=3113, hidden_dim=args.hidden_dim, dropout_rate=args.dropout).to(gnn_device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)

    history: List[Dict[str, float]] = []
    best_value = float("-inf")
    best_epoch = -1
    best_metrics: Dict[str, float] | None = None
    best_checkpoint = output_dir / "best_valid_model.pth"
    bad_epochs = 0

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, gnn_device)
        valid_metrics = evaluate(model, valid_loader, gnn_device)
        scheduler.step()

        row: Dict[str, float] = {"epoch": epoch, "train_loss": train_loss}
        row.update({f"valid_{key}": value for key, value in valid_metrics.items()})
        history.append(row)

        if args.save_every_epoch:
            torch.save(model.state_dict(), output_dir / f"model_epoch_{epoch}.pth")

        score = float(valid_metrics[args.select_metric])
        if score > best_value:
            best_value = score
            best_epoch = epoch
            best_metrics = valid_metrics
            bad_epochs = 0
            torch.save(model.state_dict(), best_checkpoint)
        else:
            bad_epochs += 1

        print(
            f"epoch={epoch} "
            f"train_loss={train_loss:.6f} "
            f"valid_accuracy={valid_metrics['accuracy']:.6f} "
            f"valid_sensitivity={valid_metrics['sensitivity']:.6f} "
            f"valid_specificity={valid_metrics['specificity']:.6f} "
            f"valid_f1={valid_metrics['f1']:.6f} "
            f"valid_mcc={valid_metrics['mcc']:.6f} "
            f"select_metric={args.select_metric}:{score:.6f} "
            f"best_epoch={best_epoch}"
        )

        if bad_epochs >= args.patience:
            print(f"early_stop_epoch={epoch}")
            break

    history_path = output_dir / "training_history.csv"
    write_history(history_path, history)

    summary = {
        "input_csv": str(Path(args.input_csv).resolve()),
        "pdb_dir": str(Path(args.pdb_dir).resolve()),
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
