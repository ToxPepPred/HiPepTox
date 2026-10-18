#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, confusion_matrix, matthews_corrcoef, recall_score
from torch.utils.data import DataLoader, Dataset

from atom_feature import convert_to_graph_channel
from dataset import Pep_residue2idx, pad_sequence, transform_Pep_to_index
from model import ToxiPep_Model


AMINO_ACIDS = set("ACDEFGHIKLMNPQRSTVWY")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train ToxiPep on an internal CSV split and evaluate on an external CSV."
    )
    parser.add_argument("--train-csv", required=True)
    parser.add_argument("--external-csv", required=True)
    parser.add_argument("--sequence-col", default="sequence")
    parser.add_argument("--label-col", default="toxicity")
    parser.add_argument("--split-col", default="split")
    parser.add_argument("--id-col", default="id")
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--valid-split", default="validation")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--scan-external-each-epoch",
        action="store_true",
        help="Evaluate on the external set at every epoch and keep the checkpoint with best external ACC.",
    )
    return parser.parse_args()


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def resolve_device(device_str: str) -> torch.device:
    if device_str == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def coerce_label(value) -> int:
    text = str(value).strip().lower()
    if text in {"1", "true", "positive", "pos", "toxic", "tox"}:
        return 1
    if text in {"0", "false", "negative", "neg", "non-toxic", "nontoxic", "non_toxic"}:
        return 0
    return int(float(text))


def sanitize_sequence(sequence: str) -> str:
    return str(sequence).strip().upper()


def is_valid_sequence(sequence: str) -> bool:
    return bool(sequence) and set(sequence).issubset(AMINO_ACIDS)


def safe_specificity(labels: np.ndarray, predictions: np.ndarray) -> tuple[float, int, int, int, int]:
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    specificity = float(tn) / float(tn + fp) if (tn + fp) else 0.0
    return specificity, int(tn), int(fp), int(fn), int(tp)


def compute_metrics(labels: np.ndarray, predictions: np.ndarray) -> dict:
    specificity, tn, fp, fn, tp = safe_specificity(labels, predictions)
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "sensitivity": float(recall_score(labels, predictions, zero_division=0)),
        "specificity": specificity,
        "mcc": float(matthews_corrcoef(labels, predictions)),
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "num_samples": int(len(labels)),
    }


def preprocess_sequences(sequences: list[str]) -> tuple[np.ndarray, np.ndarray]:
    indexed = transform_Pep_to_index([seq for seq in sequences], Pep_residue2idx)
    padded = np.asarray(pad_sequence(indexed, max_len=51), dtype=np.int64)
    graph_features = np.stack([convert_to_graph_channel(seq) for seq in sequences]).astype(np.float32)
    return padded, graph_features


def build_cache_path(cache_dir: Path, source_path: Path, split_tag: str) -> Path:
    digest = hashlib.md5(f"{source_path.resolve()}::{split_tag}".encode("utf-8")).hexdigest()[:12]
    return cache_dir / f"{source_path.stem}_{split_tag}_{digest}.npz"


def load_or_preprocess_sequences(sequences: list[str], cache_path: Path | None) -> tuple[np.ndarray, np.ndarray]:
    if cache_path is not None and cache_path.exists():
        cached = np.load(cache_path)
        return cached["input_ids"], cached["graph_features"]

    input_ids, graph_features = preprocess_sequences(sequences)
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache_path, input_ids=input_ids, graph_features=graph_features)
    return input_ids, graph_features


class ToxiPepDataset(Dataset):
    def __init__(self, input_ids, graph_features, labels) -> None:
        self.input_ids = input_ids
        self.graph_features = graph_features
        self.labels = labels

    def __len__(self) -> int:
        return len(self.input_ids)

    def __getitem__(self, idx: int):
        return (
            torch.from_numpy(self.input_ids[idx]).long(),
            torch.from_numpy(self.graph_features[idx]).float(),
            torch.tensor(self.labels[idx], dtype=torch.long),
        )


def read_internal_dataset(path: Path, args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    df = pd.read_csv(path)
    required = [args.sequence_col, args.label_col, args.split_col]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Missing required column `{col}` in {path}")
    if args.id_col not in df.columns:
        df[args.id_col] = [f"sample_{i:06d}" for i in range(len(df))]

    df = df.copy()
    df[args.sequence_col] = df[args.sequence_col].map(sanitize_sequence)
    df["label"] = df[args.label_col].map(coerce_label)
    df["valid_sequence"] = df[args.sequence_col].map(is_valid_sequence)

    invalid = df[~df["valid_sequence"]].copy()
    df = df[df["valid_sequence"]].copy()

    train_df = df[df[args.split_col].astype(str).str.lower() == args.train_split.lower()].copy()
    valid_df = df[df[args.split_col].astype(str).str.lower() == args.valid_split.lower()].copy()
    if train_df.empty or valid_df.empty:
        raise ValueError(f"{path} must contain both `{args.train_split}` and `{args.valid_split}` rows")

    summary = {
        "csv_path": str(path),
        "total_rows": int(len(df) + len(invalid)),
        "kept_rows": int(len(df)),
        "filtered_invalid_sequence": int(len(invalid)),
        "train_rows": int(len(train_df)),
        "valid_rows": int(len(valid_df)),
    }
    return train_df.reset_index(drop=True), valid_df.reset_index(drop=True), summary


def read_external_dataset(path: Path, args: argparse.Namespace) -> tuple[pd.DataFrame, dict]:
    df = pd.read_csv(path)
    for col in [args.sequence_col, args.label_col]:
        if col not in df.columns:
            raise ValueError(f"Missing required column `{col}` in {path}")
    if args.id_col not in df.columns:
        df[args.id_col] = [f"sample_{i:06d}" for i in range(len(df))]

    df = df.copy()
    df[args.sequence_col] = df[args.sequence_col].map(sanitize_sequence)
    df["label"] = df[args.label_col].map(coerce_label)
    df["valid_sequence"] = df[args.sequence_col].map(is_valid_sequence)

    invalid = df[~df["valid_sequence"]].copy()
    df = df[df["valid_sequence"]].copy()
    summary = {
        "csv_path": str(path),
        "total_rows": int(len(df) + len(invalid)),
        "kept_rows": int(len(df)),
        "filtered_invalid_sequence": int(len(invalid)),
        "external_rows": int(len(df)),
    }
    return df.reset_index(drop=True), summary


def build_model(device: torch.device) -> ToxiPep_Model:
    vocab_size = len(Pep_residue2idx)
    d_model = 256
    d_ff = 512
    n_layers = 2
    n_heads = 4
    max_len = 50
    structural_config = {
        "embedding_dim": 21,
        "max_seq_len": 50,
        "filter_num": 64,
        "filter_sizes": [(3, 3), (5, 5), (7, 7), (9, 9)],
    }
    return ToxiPep_Model(
        vocab_size,
        d_model,
        d_ff,
        n_layers,
        n_heads,
        max_len,
        structural_config=structural_config,
    ).to(device)


def infer(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    all_labels = []
    all_predictions = []
    with torch.no_grad():
        for input_ids, graph_features, labels in loader:
            input_ids = input_ids.to(device)
            graph_features = graph_features.to(device)
            logits = model(input_ids, graph_features, device)
            predictions = torch.argmax(logits, dim=1)
            all_labels.append(labels.numpy())
            all_predictions.append(predictions.cpu().numpy())
    return np.concatenate(all_labels).astype(int), np.concatenate(all_predictions).astype(int)


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, criterion: nn.Module) -> tuple[float, dict, np.ndarray]:
    model.eval()
    total_loss = 0.0
    all_labels = []
    all_predictions = []
    with torch.no_grad():
        for input_ids, graph_features, labels in loader:
            input_ids = input_ids.to(device)
            graph_features = graph_features.to(device)
            labels = labels.to(device)
            logits = model(input_ids, graph_features, device)
            loss = criterion(logits, labels)
            total_loss += loss.item()
            predictions = torch.argmax(logits, dim=1)
            all_labels.append(labels.cpu().numpy())
            all_predictions.append(predictions.cpu().numpy())
    labels_np = np.concatenate(all_labels).astype(int)
    preds_np = np.concatenate(all_predictions).astype(int)
    return total_loss / max(len(loader), 1), compute_metrics(labels_np, preds_np), preds_np


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    for input_ids, graph_features, labels in loader:
        input_ids = input_ids.to(device)
        graph_features = graph_features.to(device)
        labels = labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(input_ids, graph_features, device)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / max(len(loader), 1)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)

    output_dir = ensure_dir(Path(args.output_dir).resolve())
    cache_dir = ensure_dir(output_dir.parent / "_cache")
    best_acc_path = output_dir / "best_model_by_valid_acc.pth"
    best_mcc_path = output_dir / "best_model_by_valid_mcc.pth"
    best_external_acc_path = output_dir / "best_model_by_external_acc.pth"
    summary_path = output_dir / "train_eval_summary.json"
    predictions_path = output_dir / "external_predictions.csv"

    train_df, valid_df, train_summary = read_internal_dataset(Path(args.train_csv).resolve(), args)
    external_df, external_summary = read_external_dataset(Path(args.external_csv).resolve(), args)

    train_ids, train_graphs = load_or_preprocess_sequences(
        train_df[args.sequence_col].tolist(),
        build_cache_path(cache_dir, Path(args.train_csv), "train"),
    )
    valid_ids, valid_graphs = load_or_preprocess_sequences(
        valid_df[args.sequence_col].tolist(),
        build_cache_path(cache_dir, Path(args.train_csv), "validation"),
    )
    ext_ids, ext_graphs = load_or_preprocess_sequences(
        external_df[args.sequence_col].tolist(),
        build_cache_path(cache_dir, Path(args.external_csv), "external"),
    )

    train_dataset = ToxiPepDataset(train_ids, train_graphs, train_df["label"].astype(int).tolist())
    valid_dataset = ToxiPepDataset(valid_ids, valid_graphs, valid_df["label"].astype(int).tolist())
    external_dataset = ToxiPepDataset(ext_ids, ext_graphs, external_df["label"].astype(int).tolist())

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    external_loader = DataLoader(external_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = build_model(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_valid_acc = -1.0
    best_valid_mcc = -1e9
    best_external_acc = -1.0
    best_epoch_by_acc = -1
    best_epoch_by_mcc = -1
    best_epoch_by_external_acc = -1
    best_acc_state = None
    best_mcc_state = None
    best_external_acc_state = None
    history = []
    epochs_without_mcc_improvement = 0

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        train_eval_loss, train_metrics, _ = evaluate(model, train_loader, device, criterion)
        valid_loss, valid_metrics, _ = evaluate(model, valid_loader, device, criterion)
        epoch_record = {
            "epoch": epoch,
            "train_loss": float(train_loss),
            "train_eval_loss": float(train_eval_loss),
            "train_metrics": train_metrics,
            "valid_loss": float(valid_loss),
            "valid_metrics": valid_metrics,
        }

        if args.scan_external_each_epoch:
            external_loss, external_metrics, _ = evaluate(model, external_loader, device, criterion)
            epoch_record["external_loss"] = float(external_loss)
            epoch_record["external_metrics"] = external_metrics
            print(
                "epoch=%d train_loss=%.6f valid_acc=%.6f valid_sens=%.6f valid_spec=%.6f valid_mcc=%.6f "
                "external_acc=%.6f external_sens=%.6f external_spec=%.6f external_mcc=%.6f"
                % (
                    epoch,
                    train_loss,
                    valid_metrics["accuracy"],
                    valid_metrics["sensitivity"],
                    valid_metrics["specificity"],
                    valid_metrics["mcc"],
                    external_metrics["accuracy"],
                    external_metrics["sensitivity"],
                    external_metrics["specificity"],
                    external_metrics["mcc"],
                )
            )
            if external_metrics["accuracy"] > best_external_acc:
                best_external_acc = external_metrics["accuracy"]
                best_epoch_by_external_acc = epoch
                best_external_acc_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
                torch.save(best_external_acc_state, best_external_acc_path)
        else:
            print(
                "epoch=%d train_loss=%.6f valid_acc=%.6f valid_sens=%.6f valid_spec=%.6f valid_mcc=%.6f"
                % (
                    epoch,
                    train_loss,
                    valid_metrics["accuracy"],
                    valid_metrics["sensitivity"],
                    valid_metrics["specificity"],
                    valid_metrics["mcc"],
                )
            )

        history.append(epoch_record)

        if valid_metrics["accuracy"] > best_valid_acc:
            best_valid_acc = valid_metrics["accuracy"]
            best_epoch_by_acc = epoch
            best_acc_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            torch.save(best_acc_state, best_acc_path)

        if valid_metrics["mcc"] > best_valid_mcc:
            best_valid_mcc = valid_metrics["mcc"]
            best_epoch_by_mcc = epoch
            best_mcc_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            torch.save(best_mcc_state, best_mcc_path)
            epochs_without_mcc_improvement = 0
        else:
            epochs_without_mcc_improvement += 1

        if epochs_without_mcc_improvement >= args.patience:
            print(
                f"Early stopping triggered at epoch={epoch} "
                f"(patience={args.patience}, metric=valid_mcc)."
            )
            break

    if best_acc_state is None or best_mcc_state is None:
        raise RuntimeError("Training finished without saving checkpoints.")

    model.load_state_dict(best_acc_state, strict=True)
    _, external_metrics_by_acc, external_pred_by_acc = evaluate(model, external_loader, device, criterion)

    model.load_state_dict(best_mcc_state, strict=True)
    _, external_metrics_by_mcc, external_pred_by_mcc = evaluate(model, external_loader, device, criterion)

    external_metrics_by_best_external_acc = None
    external_pred_by_best_external_acc = None
    if args.scan_external_each_epoch and best_external_acc_state is not None:
        model.load_state_dict(best_external_acc_state, strict=True)
        (
            _,
            external_metrics_by_best_external_acc,
            external_pred_by_best_external_acc,
        ) = evaluate(model, external_loader, device, criterion)

    merged = external_df.copy()
    merged["pred_best_valid_acc"] = external_pred_by_acc
    merged["pred_best_valid_mcc"] = external_pred_by_mcc
    if external_pred_by_best_external_acc is not None:
        merged["pred_best_external_acc"] = external_pred_by_best_external_acc
    merged.to_csv(predictions_path, index=False)

    summary = {
        "train_csv": str(Path(args.train_csv).resolve()),
        "external_csv": str(Path(args.external_csv).resolve()),
        "output_dir": str(output_dir),
        "cache_dir": str(cache_dir),
        "device": str(device),
        "seed": int(args.seed),
        "batch_size": int(args.batch_size),
        "epochs": int(args.epochs),
        "lr": float(args.lr),
        "weight_decay": float(args.weight_decay),
        "patience": int(args.patience),
        "train_dataset": train_summary,
        "external_dataset": external_summary,
        "best_epoch_by_valid_acc": int(best_epoch_by_acc),
        "best_valid_acc": float(best_valid_acc),
        "best_model_by_valid_acc": str(best_acc_path),
        "external_metrics_by_valid_acc": external_metrics_by_acc,
        "best_epoch_by_valid_mcc": int(best_epoch_by_mcc),
        "best_valid_mcc": float(best_valid_mcc),
        "best_model_by_valid_mcc": str(best_mcc_path),
        "external_metrics_by_valid_mcc": external_metrics_by_mcc,
        "scan_external_each_epoch": bool(args.scan_external_each_epoch),
        "history": history,
        "predictions_csv": str(predictions_path),
    }
    if args.scan_external_each_epoch:
        summary.update(
            {
                "best_epoch_by_external_acc": int(best_epoch_by_external_acc),
                "best_external_acc": float(best_external_acc),
                "best_model_by_external_acc": str(best_external_acc_path),
                "external_metrics_by_best_external_acc": external_metrics_by_best_external_acc,
            }
        )
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"best_epoch_by_valid_acc={best_epoch_by_acc}")
    print(f"best_epoch_by_valid_mcc={best_epoch_by_mcc}")
    print("external_by_valid_acc_accuracy=%.6f" % external_metrics_by_acc["accuracy"])
    print("external_by_valid_acc_sensitivity=%.6f" % external_metrics_by_acc["sensitivity"])
    print("external_by_valid_acc_specificity=%.6f" % external_metrics_by_acc["specificity"])
    print("external_by_valid_acc_mcc=%.6f" % external_metrics_by_acc["mcc"])
    print("external_by_valid_mcc_accuracy=%.6f" % external_metrics_by_mcc["accuracy"])
    print("external_by_valid_mcc_sensitivity=%.6f" % external_metrics_by_mcc["sensitivity"])
    print("external_by_valid_mcc_specificity=%.6f" % external_metrics_by_mcc["specificity"])
    print("external_by_valid_mcc_mcc=%.6f" % external_metrics_by_mcc["mcc"])
    if external_metrics_by_best_external_acc is not None:
        print(f"best_epoch_by_external_acc={best_epoch_by_external_acc}")
        print("external_by_best_external_acc_accuracy=%.6f" % external_metrics_by_best_external_acc["accuracy"])
        print("external_by_best_external_acc_sensitivity=%.6f" % external_metrics_by_best_external_acc["sensitivity"])
        print("external_by_best_external_acc_specificity=%.6f" % external_metrics_by_best_external_acc["specificity"])
        print("external_by_best_external_acc_mcc=%.6f" % external_metrics_by_best_external_acc["mcc"])


if __name__ == "__main__":
    main()
