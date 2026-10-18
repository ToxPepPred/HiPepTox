#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    recall_score,
    roc_auc_score,
)
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_DIR))

from model_torch import ToxMSRCPyTorch  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train and evaluate a PyTorch reimplementation of ToxMSRC "
            "using precomputed Word2Vec features."
        )
    )
    parser.add_argument("--train-csv", required=True, help="Training CSV with split column.")
    parser.add_argument(
        "--train-features-npz",
        required=True,
        help="Feature npz for the training CSV. Expected key: 'x'.",
    )
    parser.add_argument("--external-csv", required=True, help="External test CSV.")
    parser.add_argument(
        "--external-features-npz",
        required=True,
        help="Feature npz for the external test CSV. Expected key: 'x'.",
    )
    parser.add_argument("--output-dir", required=True, help="Directory to save outputs.")
    parser.add_argument("--label-col", default="toxicity", help="Label column name.")
    parser.add_argument("--split-col", default="split", help="Split column name.")
    parser.add_argument(
        "--train-split-value",
        default="train",
        help="Value in split column denoting the training subset.",
    )
    parser.add_argument(
        "--valid-split-value",
        default="validation",
        help="Value in split column denoting the validation subset.",
    )
    parser.add_argument("--epochs", type=int, default=30, help="Maximum training epochs.")
    parser.add_argument("--batch-size", type=int, default=64, help="Mini-batch size.")
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-4,
        help="Learning rate for Adam.",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.0,
        help="Weight decay for Adam.",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=6,
        help="Early stopping patience measured on validation MCC.",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use, e.g. cpu or cuda.",
    )
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader workers.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--amp",
        action="store_true",
        help="Use torch.cuda.amp mixed precision when training on CUDA.",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def safe_divide(numerator: float, denominator: float) -> float:
    return float(numerator) / float(denominator) if denominator else 0.0


def load_features(path: str) -> np.ndarray:
    data = np.load(Path(path).resolve())
    if "x" not in data:
        raise KeyError(f"Feature file {path} does not contain key 'x'.")
    return data["x"].astype(np.float32)


def compute_metrics(labels: np.ndarray, probabilities: np.ndarray) -> dict[str, float | int]:
    predictions = probabilities.argmax(axis=1)
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    metrics: dict[str, float | int] = {
        "accuracy": float(accuracy_score(labels, predictions)),
        "sensitivity": float(recall_score(labels, predictions, zero_division=0)),
        "specificity": safe_divide(tn, tn + fp),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "mcc": float(matthews_corrcoef(labels, predictions)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }
    unique_labels = np.unique(labels)
    if len(unique_labels) == 2:
        metrics["auc"] = float(roc_auc_score(labels, probabilities[:, 1]))
    else:
        metrics["auc"] = float("nan")
    return metrics


def make_loader(
    features: np.ndarray,
    labels: np.ndarray,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    pin_memory: bool,
) -> DataLoader:
    dataset = TensorDataset(
        torch.from_numpy(features),
        torch.from_numpy(labels.astype(np.int64)),
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, dict[str, float | int], np.ndarray, np.ndarray]:
    model.eval()
    total_loss = 0.0
    total_count = 0
    all_labels: list[np.ndarray] = []
    all_probs: list[np.ndarray] = []
    with torch.no_grad():
        for features, labels in loader:
            features = features.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            logits = model(features)
            loss = criterion(logits, labels)
            probs = torch.softmax(logits, dim=1)

            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            total_count += batch_size

            all_labels.append(labels.cpu().numpy())
            all_probs.append(probs.cpu().numpy())

    labels_np = np.concatenate(all_labels, axis=0)
    probs_np = np.concatenate(all_probs, axis=0)
    metrics = compute_metrics(labels_np, probs_np)
    avg_loss = total_loss / total_count if total_count else 0.0
    predictions = probs_np.argmax(axis=1)
    return avg_loss, metrics, probs_np, predictions


def save_checkpoint(
    path: Path,
    model: nn.Module,
    epoch: int,
    args: argparse.Namespace,
    metric_name: str,
    metric_value: float,
) -> None:
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "metric_name": metric_name,
            "metric_value": metric_value,
            "model_class": "ToxMSRCPyTorch",
            "args": vars(args),
        },
        path,
    )


def load_checkpoint(path: Path, device: torch.device) -> ToxMSRCPyTorch:
    checkpoint = torch.load(path, map_location=device)
    model = ToxMSRCPyTorch().to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False.")

    train_csv = Path(args.train_csv).resolve()
    external_csv = Path(args.external_csv).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    train_df = pd.read_csv(train_csv)
    external_df = pd.read_csv(external_csv)
    if args.label_col not in train_df.columns or args.label_col not in external_df.columns:
        raise ValueError(f"Missing label column '{args.label_col}'.")
    if args.split_col not in train_df.columns:
        raise ValueError(f"Missing split column '{args.split_col}' in {train_csv}.")

    train_features_all = load_features(args.train_features_npz)
    external_features = load_features(args.external_features_npz)
    if len(train_df) != len(train_features_all):
        raise ValueError(
            f"Training row mismatch: csv={len(train_df)} features={len(train_features_all)}"
        )
    if len(external_df) != len(external_features):
        raise ValueError(
            f"External row mismatch: csv={len(external_df)} features={len(external_features)}"
        )

    split_series = train_df[args.split_col].astype(str).str.lower()
    train_mask = split_series == args.train_split_value.lower()
    valid_mask = split_series == args.valid_split_value.lower()
    if not train_mask.any():
        raise ValueError(f"No rows found for train split value '{args.train_split_value}'.")
    if not valid_mask.any():
        raise ValueError(f"No rows found for validation split value '{args.valid_split_value}'.")

    train_features = train_features_all[train_mask.to_numpy()]
    valid_features = train_features_all[valid_mask.to_numpy()]
    train_labels = train_df.loc[train_mask, args.label_col].astype(int).to_numpy()
    valid_labels = train_df.loc[valid_mask, args.label_col].astype(int).to_numpy()
    external_labels = external_df[args.label_col].astype(int).to_numpy()

    pin_memory = device.type == "cuda"
    train_loader = make_loader(
        train_features,
        train_labels,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )
    valid_loader = make_loader(
        valid_features,
        valid_labels,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )
    external_loader = make_loader(
        external_features,
        external_labels,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )

    model = ToxMSRCPyTorch().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    use_amp = args.amp and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    best_mcc_path = output_dir / "best_model_by_valid_mcc.pth"
    best_acc_path = output_dir / "best_model_by_valid_acc.pth"

    history: list[dict[str, float | int]] = []
    best_epoch_by_mcc = 0
    best_epoch_by_acc = 0
    best_mcc = -float("inf")
    best_acc = -float("inf")
    best_valid_mcc_metrics: dict[str, float | int] | None = None
    best_valid_acc_metrics: dict[str, float | int] | None = None
    epochs_without_improvement = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss_total = 0.0
        train_count = 0
        train_correct = 0

        for features, labels in train_loader:
            features = features.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=use_amp):
                logits = model(features)
                loss = criterion(logits, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            batch_size = labels.size(0)
            train_loss_total += loss.item() * batch_size
            train_count += batch_size
            train_correct += (logits.argmax(dim=1) == labels).sum().item()

        train_loss = train_loss_total / train_count if train_count else 0.0
        train_accuracy = train_correct / train_count if train_count else 0.0

        valid_loss, valid_metrics, _, _ = evaluate(model, valid_loader, criterion, device)
        epoch_record = {
            "epoch": epoch,
            "train_loss": float(train_loss),
            "train_accuracy": float(train_accuracy),
            "valid_loss": float(valid_loss),
            **valid_metrics,
        }
        history.append(epoch_record)

        improved_mcc = valid_metrics["mcc"] > best_mcc
        improved_acc = valid_metrics["accuracy"] > best_acc

        if improved_mcc:
            best_mcc = float(valid_metrics["mcc"])
            best_epoch_by_mcc = epoch
            best_valid_mcc_metrics = dict(valid_metrics)
            save_checkpoint(best_mcc_path, model, epoch, args, "valid_mcc", best_mcc)
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if improved_acc:
            best_acc = float(valid_metrics["accuracy"])
            best_epoch_by_acc = epoch
            best_valid_acc_metrics = dict(valid_metrics)
            save_checkpoint(best_acc_path, model, epoch, args, "valid_accuracy", best_acc)

        print(
            f"epoch={epoch} train_loss={train_loss:.6f} train_acc={train_accuracy:.6f} "
            f"valid_loss={valid_loss:.6f} valid_acc={valid_metrics['accuracy']:.6f} "
            f"valid_mcc={valid_metrics['mcc']:.6f} valid_sn={valid_metrics['sensitivity']:.6f} "
            f"valid_sp={valid_metrics['specificity']:.6f}"
        )

        if epochs_without_improvement >= args.patience:
            print(
                f"Early stopping triggered at epoch={epoch} "
                f"(patience={args.patience}, metric=valid_mcc)."
            )
            break

    if best_epoch_by_mcc == 0 or best_valid_mcc_metrics is None:
        raise RuntimeError("No checkpoint selected by validation MCC.")
    if best_epoch_by_acc == 0 or best_valid_acc_metrics is None:
        raise RuntimeError("No checkpoint selected by validation ACC.")

    model_by_mcc = load_checkpoint(best_mcc_path, device)
    _, external_metrics_by_mcc, external_prob_by_mcc, external_pred_by_mcc = evaluate(
        model_by_mcc,
        external_loader,
        criterion,
        device,
    )

    model_by_acc = load_checkpoint(best_acc_path, device)
    _, external_metrics_by_acc, external_prob_by_acc, external_pred_by_acc = evaluate(
        model_by_acc,
        external_loader,
        criterion,
        device,
    )

    predictions_df = external_df.copy()
    predictions_df["pred_best_valid_mcc"] = external_pred_by_mcc
    predictions_df["prob_non_toxic_best_valid_mcc"] = external_prob_by_mcc[:, 0]
    predictions_df["prob_toxic_best_valid_mcc"] = external_prob_by_mcc[:, 1]
    predictions_df["pred_best_valid_acc"] = external_pred_by_acc
    predictions_df["prob_non_toxic_best_valid_acc"] = external_prob_by_acc[:, 0]
    predictions_df["prob_toxic_best_valid_acc"] = external_prob_by_acc[:, 1]
    predictions_df.to_csv(output_dir / "external_predictions.csv", index=False)

    summary = {
        "train_csv": str(train_csv),
        "train_features_npz": str(Path(args.train_features_npz).resolve()),
        "external_csv": str(external_csv),
        "external_features_npz": str(Path(args.external_features_npz).resolve()),
        "output_dir": str(output_dir),
        "num_train_samples": int(len(train_features)),
        "num_validation_samples": int(len(valid_features)),
        "num_external_samples": int(len(external_features)),
        "feature_shape": list(train_features.shape[1:]),
        "training_config": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "patience": args.patience,
            "device": str(device),
            "seed": args.seed,
            "amp": bool(use_amp),
        },
        "validation_history": history,
        "best_epoch_by_valid_mcc": int(best_epoch_by_mcc),
        "best_valid_mcc": float(best_mcc),
        "best_valid_metrics_at_mcc_epoch": best_valid_mcc_metrics,
        "best_model_by_valid_mcc": str(best_mcc_path),
        "external_metrics_by_valid_mcc": external_metrics_by_mcc,
        "best_epoch_by_valid_acc": int(best_epoch_by_acc),
        "best_valid_acc": float(best_acc),
        "best_valid_metrics_at_acc_epoch": best_valid_acc_metrics,
        "best_model_by_valid_acc": str(best_acc_path),
        "external_metrics_by_valid_acc": external_metrics_by_acc,
    }

    summary_path = output_dir / "train_eval_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"summary_json={summary_path}")
    print(f"best_epoch_by_valid_mcc={best_epoch_by_mcc}")
    print(f"best_epoch_by_valid_acc={best_epoch_by_acc}")
    print(
        f"external_by_valid_mcc accuracy={external_metrics_by_mcc['accuracy']:.6f} "
        f"sensitivity={external_metrics_by_mcc['sensitivity']:.6f} "
        f"specificity={external_metrics_by_mcc['specificity']:.6f} "
        f"mcc={external_metrics_by_mcc['mcc']:.6f}"
    )
    print(
        f"external_by_valid_acc accuracy={external_metrics_by_acc['accuracy']:.6f} "
        f"sensitivity={external_metrics_by_acc['sensitivity']:.6f} "
        f"specificity={external_metrics_by_acc['specificity']:.6f} "
        f"mcc={external_metrics_by_acc['mcc']:.6f}"
    )


if __name__ == "__main__":
    main()
