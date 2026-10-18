#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import pickle
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, confusion_matrix, matthews_corrcoef, recall_score, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, Dataset


PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "data"
MODEL_PATH = DATA_DIR / "model_saved.pkl"
AAINDEX_PATH = DATA_DIR / "AAindex.pkl"
MAX_TOKENS = 51

sys.path.insert(0, str(PROJECT_DIR))
from model import CAPTP_model  # noqa: E402


class SequenceDataset(Dataset):
    def __init__(self, encoded: np.ndarray, labels: np.ndarray) -> None:
        self.encoded = encoded
        self.labels = labels

    def __len__(self) -> int:
        return len(self.encoded)

    def __getitem__(self, index: int):
        return (
            torch.from_numpy(self.encoded[index]).long(),
            torch.tensor(self.labels[index], dtype=torch.long),
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train CAPTP on an internal CSV split and evaluate on an external CSV."
    )
    parser.add_argument("--train-csv", required=True)
    parser.add_argument("--external-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--sequence-col", default="sequence")
    parser.add_argument("--label-col", default="toxicity")
    parser.add_argument("--split-col", default="split")
    parser.add_argument("--id-col", default="id")
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--valid-split", default="validation")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--weights", default=str(MODEL_PATH), help="Optional initialization weights.")
    parser.add_argument("--aaindex", default=str(AAINDEX_PATH))
    parser.add_argument("--train-from-scratch", action="store_true")
    return parser.parse_args()


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def resolve_device(device_str: str) -> torch.device:
    if device_str == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def coerce_label(value) -> int:
    text = str(value).strip().lower()
    if text in {"1", "true", "positive", "pos", "toxic", "tox"}:
        return 1
    if text in {"0", "false", "negative", "neg", "non-toxic", "nontoxic", "non_toxic"}:
        return 0
    return int(float(text))


def sanitize_sequence(sequence: str, vocabulary: dict[str, int]) -> str:
    sequence = str(sequence).strip().upper()
    allowed = {token for token in vocabulary.keys() if len(token) == 1 and token.isalpha()}
    return "".join(residue for residue in sequence if residue in allowed)


def encode_sequences(sequences: list[str], vocabulary: dict[str, int]) -> np.ndarray:
    cls_id = vocabulary["[CLS]"]
    pad_id = vocabulary["[PAD]"]
    encoded = []
    for seq in sequences:
        token_ids = [vocabulary[residue] for residue in seq[: MAX_TOKENS - 1]]
        token_ids = [cls_id] + token_ids
        if len(token_ids) < MAX_TOKENS:
            token_ids.extend([pad_id] * (MAX_TOKENS - len(token_ids)))
        encoded.append(token_ids)
    return np.asarray(encoded, dtype=np.int64)


def read_internal_dataset(path: Path, args: argparse.Namespace, vocabulary: dict[str, int]):
    df = pd.read_csv(path)
    required = [args.sequence_col, args.label_col, args.split_col]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Missing required column `{col}` in {path}")
    if args.id_col not in df.columns:
        df[args.id_col] = [f"sample_{i:06d}" for i in range(len(df))]

    df = df.copy()
    df["sequence_clean"] = df[args.sequence_col].map(lambda s: sanitize_sequence(s, vocabulary))
    df["label"] = df[args.label_col].map(coerce_label).astype(int)
    invalid = df["sequence_clean"].str.len() == 0
    df = df.loc[~invalid].reset_index(drop=True)

    train_df = df[df[args.split_col].astype(str).str.lower() == args.train_split.lower()].copy()
    valid_df = df[df[args.split_col].astype(str).str.lower() == args.valid_split.lower()].copy()
    if train_df.empty or valid_df.empty:
        raise ValueError(f"{path} must contain `{args.train_split}` and `{args.valid_split}` rows")

    summary = {
        "csv_path": str(path),
        "total_rows": int(len(df) + int(invalid.sum())),
        "kept_rows": int(len(df)),
        "filtered_invalid_sequences": int(invalid.sum()),
        "train_rows": int(len(train_df)),
        "valid_rows": int(len(valid_df)),
    }
    return train_df.reset_index(drop=True), valid_df.reset_index(drop=True), summary


def read_external_dataset(path: Path, args: argparse.Namespace, vocabulary: dict[str, int]):
    df = pd.read_csv(path)
    for col in [args.sequence_col, args.label_col]:
        if col not in df.columns:
            raise ValueError(f"Missing required column `{col}` in {path}")
    if args.id_col not in df.columns:
        df[args.id_col] = [f"sample_{i:06d}" for i in range(len(df))]

    df = df.copy()
    df["sequence_clean"] = df[args.sequence_col].map(lambda s: sanitize_sequence(s, vocabulary))
    df["label"] = df[args.label_col].map(coerce_label).astype(int)
    invalid = df["sequence_clean"].str.len() == 0
    df = df.loc[~invalid].reset_index(drop=True)

    summary = {
        "csv_path": str(path),
        "total_rows": int(len(df) + int(invalid.sum())),
        "kept_rows": int(len(df)),
        "filtered_invalid_sequences": int(invalid.sum()),
        "external_rows": int(len(df)),
    }
    return df.reset_index(drop=True), summary


def build_loader(df: pd.DataFrame, vocabulary: dict[str, int], batch_size: int, shuffle: bool) -> DataLoader:
    encoded = encode_sequences(df["sequence_clean"].tolist(), vocabulary)
    labels = df["label"].to_numpy(dtype=np.int64)
    dataset = SequenceDataset(encoded, labels)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=False)


def model_logits(model: CAPTP_model.CAPTP, input_ids: torch.Tensor) -> torch.Tensor:
    representations = model(input_ids)
    return model.block2(representations)


def safe_specificity(labels: np.ndarray, predictions: np.ndarray) -> tuple[float, int, int, int, int]:
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    specificity = float(tn) / float(tn + fp) if (tn + fp) else 0.0
    return specificity, int(tn), int(fp), int(fn), int(tp)


def compute_metrics(labels: np.ndarray, predictions: np.ndarray, probabilities: np.ndarray) -> dict:
    specificity, tn, fp, fn, tp = safe_specificity(labels, predictions)
    auc = float(roc_auc_score(labels, probabilities)) if len(np.unique(labels)) == 2 else float("nan")
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "sensitivity": float(recall_score(labels, predictions, zero_division=0)),
        "specificity": specificity,
        "mcc": float(matthews_corrcoef(labels, predictions)),
        "auc": auc,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "num_samples": int(len(labels)),
    }


def evaluate(model, loader: DataLoader, device: torch.device, criterion: nn.Module):
    model.eval()
    total_loss = 0.0
    total_count = 0
    all_labels = []
    all_preds = []
    all_probs = []

    with torch.no_grad():
        for input_ids, labels in loader:
            input_ids = input_ids.to(device)
            labels = labels.to(device)
            logits = model_logits(model, input_ids)
            loss = criterion(logits, labels)
            probs = F.softmax(logits, dim=1)[:, 1]
            preds = torch.argmax(logits, dim=1)

            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            total_count += batch_size
            all_labels.append(labels.cpu().numpy())
            all_preds.append(preds.cpu().numpy())
            all_probs.append(probs.cpu().numpy())

    labels_np = np.concatenate(all_labels)
    preds_np = np.concatenate(all_preds)
    probs_np = np.concatenate(all_probs)
    metrics = compute_metrics(labels_np, preds_np, probs_np)
    return total_loss / max(total_count, 1), metrics, preds_np, probs_np


def save_checkpoint(path: Path, model: CAPTP_model.CAPTP, epoch: int, metric_name: str, metric_value: float, args):
    torch.save(
        {
            "epoch": epoch,
            "metric_name": metric_name,
            "metric_value": metric_value,
            "model_state_dict": model.state_dict(),
            "args": vars(args),
        },
        path,
    )


def load_checkpoint(path: Path, device: torch.device) -> CAPTP_model.CAPTP:
    checkpoint = torch.load(path, map_location=device)
    model = CAPTP_model.CAPTP().to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    return model


def export_predictions(df: pd.DataFrame, predictions: np.ndarray, probabilities: np.ndarray, output_path: Path) -> None:
    merged = df.copy()
    merged["predicted_label"] = predictions.astype(int)
    merged["prob_toxic"] = probabilities.astype(float)
    merged["prediction_text"] = np.where(predictions == 1, "Toxic peptide", "Non toxic peptide")
    merged.to_csv(output_path, index=False)


def main() -> None:
    args = parse_args()
    output_dir = ensure_dir(Path(args.output_dir).resolve())
    device = resolve_device(args.device)
    set_seed(args.seed)

    vocabulary = pickle.load(open(args.aaindex, "rb"))
    train_df, valid_df, train_summary = read_internal_dataset(Path(args.train_csv).resolve(), args, vocabulary)
    external_df, external_summary = read_external_dataset(Path(args.external_csv).resolve(), args, vocabulary)

    train_loader = build_loader(train_df, vocabulary, args.batch_size, shuffle=True)
    valid_loader = build_loader(valid_df, vocabulary, args.batch_size, shuffle=False)
    external_loader = build_loader(external_df, vocabulary, args.batch_size, shuffle=False)

    model = CAPTP_model.CAPTP().to(device)
    if not args.train_from_scratch:
        state_dict = torch.load(args.weights, map_location=device)
        model.load_state_dict(state_dict, strict=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()

    best_acc_path = output_dir / "best_model_by_valid_acc.pth"
    best_mcc_path = output_dir / "best_model_by_valid_mcc.pth"
    history = []

    best_valid_acc = -1.0
    best_valid_mcc = -1e9
    best_epoch_acc = -1
    best_epoch_mcc = -1
    epochs_without_mcc_improvement = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total_count = 0
        for input_ids, labels in train_loader:
            input_ids = input_ids.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            logits = model_logits(model, input_ids)
            loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            total_count += batch_size

        train_loss = total_loss / max(total_count, 1)
        valid_loss, valid_metrics, _, _ = evaluate(model, valid_loader, device, criterion)
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "valid_loss": valid_loss,
                "valid_metrics": valid_metrics,
            }
        )

        if valid_metrics["accuracy"] > best_valid_acc:
            best_valid_acc = valid_metrics["accuracy"]
            best_epoch_acc = epoch
            save_checkpoint(best_acc_path, model, epoch, "valid_acc", best_valid_acc, args)

        if valid_metrics["mcc"] > best_valid_mcc:
            best_valid_mcc = valid_metrics["mcc"]
            best_epoch_mcc = epoch
            epochs_without_mcc_improvement = 0
            save_checkpoint(best_mcc_path, model, epoch, "valid_mcc", best_valid_mcc, args)
        else:
            epochs_without_mcc_improvement += 1

        print(
            f"epoch={epoch} train_loss={train_loss:.6f} valid_acc={valid_metrics['accuracy']:.6f} "
            f"valid_sens={valid_metrics['sensitivity']:.6f} valid_spec={valid_metrics['specificity']:.6f} "
            f"valid_mcc={valid_metrics['mcc']:.6f} valid_auc={valid_metrics['auc']:.6f}"
        )

        if epochs_without_mcc_improvement >= args.patience:
            print(f"Early stopping at epoch {epoch} (patience={args.patience}, metric=valid_mcc)")
            break

    best_acc_model = load_checkpoint(best_acc_path, device)
    _, external_metrics_by_acc, external_pred_by_acc, external_prob_by_acc = evaluate(
        best_acc_model, external_loader, device, criterion
    )
    export_predictions(external_df, external_pred_by_acc, external_prob_by_acc, output_dir / "external_predictions_by_valid_acc.csv")

    best_mcc_model = load_checkpoint(best_mcc_path, device)
    _, external_metrics_by_mcc, external_pred_by_mcc, external_prob_by_mcc = evaluate(
        best_mcc_model, external_loader, device, criterion
    )
    export_predictions(external_df, external_pred_by_mcc, external_prob_by_mcc, output_dir / "external_predictions_by_valid_mcc.csv")

    summary = {
        "train_csv": str(Path(args.train_csv).resolve()),
        "external_csv": str(Path(args.external_csv).resolve()),
        "output_dir": str(output_dir),
        "device": str(device),
        "seed": args.seed,
        "initialized_from_pretrained": not args.train_from_scratch,
        "pretrained_weights": None if args.train_from_scratch else str(Path(args.weights).resolve()),
        "train_dataset": train_summary,
        "external_dataset": external_summary,
        "training_config": {
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "patience": args.patience,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
        },
        "best_epoch_by_valid_acc": best_epoch_acc,
        "best_epoch_by_valid_mcc": best_epoch_mcc,
        "best_valid_acc": best_valid_acc,
        "best_valid_mcc": best_valid_mcc,
        "best_model_by_valid_acc": str(best_acc_path),
        "best_model_by_valid_mcc": str(best_mcc_path),
        "external_metrics_by_valid_acc": external_metrics_by_acc,
        "external_metrics_by_valid_mcc": external_metrics_by_mcc,
        "history": history,
        "external_predictions_by_valid_acc": str(output_dir / "external_predictions_by_valid_acc.csv"),
        "external_predictions_by_valid_mcc": str(output_dir / "external_predictions_by_valid_mcc.csv"),
    }
    summary_path = output_dir / "train_eval_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
