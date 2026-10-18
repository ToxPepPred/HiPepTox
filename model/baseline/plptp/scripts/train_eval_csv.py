#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
import random
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, confusion_matrix, matthews_corrcoef, recall_score
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, EsmConfig, EsmModel


PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_ESM_DIR = PROJECT_DIR / "local_models" / "esm2_t30_150M_UR50D"
DEFAULT_INIT_WEIGHTS = PROJECT_DIR / "model" / "model.pkl"
AA_ALLOWED = set("LAGVSERTIDPKQNFYMHWCXB UZO.-")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train PLPTP on an internal CSV split and evaluate on an external CSV."
    )
    parser.add_argument("--train-csv", required=True, help="Internal CSV with train/validation split column.")
    parser.add_argument("--external-csv", required=True, help="External test CSV.")
    parser.add_argument("--sequence-col", default="sequence")
    parser.add_argument("--label-col", default="toxicity")
    parser.add_argument("--split-col", default="split")
    parser.add_argument("--id-col", default="id")
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--valid-split", default="validation")
    parser.add_argument("--output-dir", required=True, help="Directory to save checkpoints and reports.")
    parser.add_argument("--esm-model-dir", default=str(DEFAULT_ESM_DIR))
    parser.add_argument("--init-weights", default=str(DEFAULT_INIT_WEIGHTS))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=50)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--seed", type=int, default=777)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--freeze-bert", action="store_true", help="Freeze ESM encoder during finetuning.")
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


def sanitize_sequence(sequence: str) -> str:
    sequence = str(sequence).strip().upper()
    return "".join(residue if residue in AA_ALLOWED else "X" for residue in sequence)


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


class FocalLoss(nn.Module):
    def __init__(self, alpha: float = 0.6, gamma: float = 2.0, reduction: str = "mean") -> None:
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(logits, targets, reduction="none")
        pt = torch.exp(-ce_loss)
        loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        if self.reduction == "sum":
            return loss.sum()
        if self.reduction == "none":
            return loss
        return loss.mean()


class PLPTPDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        tokenizer,
        sequence_col: str,
        label_col: str,
        max_length: int,
    ) -> None:
        self.df = df.reset_index(drop=True).copy()
        self.tokenizer = tokenizer
        self.sequence_col = sequence_col
        self.label_col = label_col
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, index: int):
        row = self.df.iloc[index]
        seq = sanitize_sequence(row[self.sequence_col])
        encoding = self.tokenizer.encode_plus(
            " ".join(list(seq)),
            add_special_tokens=True,
            max_length=self.max_length,
            return_token_type_ids=False,
            return_attention_mask=True,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
        )
        sample = {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
        }
        label = int(row[self.label_col])
        return sample, label


class PLPTPModel(nn.Module):
    def __init__(self, esm_dir: Path) -> None:
        super().__init__()
        config = EsmConfig.from_pretrained(str(esm_dir))
        self.bert = EsmModel(config)
        self.LSTM = nn.LSTM(
            input_size=640,
            hidden_size=320,
            num_layers=2,
            bidirectional=True,
            dropout=0.3,
            batch_first=True,
        )
        self.global_avg_pooling_layer = nn.AdaptiveAvgPool1d(1)
        self.conv1 = nn.Conv1d(in_channels=1024, out_channels=256, kernel_size=2, stride=1)
        self.conv2 = nn.Conv1d(in_channels=1024, out_channels=256, kernel_size=3, stride=1)
        self.conv3 = nn.Conv1d(in_channels=1024, out_channels=256, kernel_size=4, stride=1)
        self.conv4 = nn.Conv1d(in_channels=1024, out_channels=256, kernel_size=5, stride=1)
        self.fc1 = nn.Linear(640, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 2)
        self.relu = nn.ReLU()
        self.lrelu = nn.LeakyReLU()
        self.max_pool1d = nn.MaxPool1d(kernel_size=5)
        self.dropout = nn.Dropout(0.3)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        sequence_output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        ).last_hidden_state
        _, (hidden, _) = self.LSTM(sequence_output)
        features = torch.cat((hidden[-2], hidden[-1]), dim=1)
        out = self.fc1(features)
        out = self.fc2(out)
        out = self.relu(out)
        return self.fc3(out)


def read_dataset(path: Path, args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(path)
    required = [args.sequence_col, args.label_col, args.split_col, args.id_col]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Missing required column `{col}` in {path}")
    train_df = df[df[args.split_col].astype(str).str.lower() == args.train_split.lower()].copy()
    valid_df = df[df[args.split_col].astype(str).str.lower() == args.valid_split.lower()].copy()
    if train_df.empty or valid_df.empty:
        raise ValueError(f"{path} must contain both `{args.train_split}` and `{args.valid_split}` rows")
    return train_df, valid_df


def read_external_dataset(path: Path, args: argparse.Namespace) -> pd.DataFrame:
    df = pd.read_csv(path)
    for col in [args.sequence_col, args.label_col, args.id_col]:
        if col not in df.columns:
            raise ValueError(f"Missing required column `{col}` in {path}")
    return df.copy()


def load_initial_weights(model: nn.Module, init_weights: Path) -> dict:
    state_dict = torch.load(init_weights, map_location="cpu")
    if not isinstance(state_dict, (dict, OrderedDict)):
        raise ValueError(f"Unexpected state dict type in {init_weights}: {type(state_dict)}")
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    return {
        "init_weights": str(init_weights),
        "missing_keys": list(missing),
        "unexpected_keys": list(unexpected),
    }


def maybe_freeze_bert(model: PLPTPModel, freeze_bert: bool) -> None:
    if not freeze_bert:
        return
    for param in model.bert.parameters():
        param.requires_grad = False


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    grad_accum_steps: int,
) -> float:
    model.train()
    total_loss = 0.0
    optimizer.zero_grad(set_to_none=True)

    for step, (samples, labels) in enumerate(loader, start=1):
        input_ids = samples["input_ids"].to(device)
        attention_mask = samples["attention_mask"].to(device)
        labels = labels.to(device).long()

        logits = model(input_ids, attention_mask)
        loss = criterion(logits, labels) / grad_accum_steps
        loss.backward()

        if step % grad_accum_steps == 0 or step == len(loader):
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        total_loss += loss.item() * grad_accum_steps

    return total_loss / max(len(loader), 1)


def infer(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    all_labels = []
    all_predictions = []
    with torch.no_grad():
        for samples, labels in loader:
            input_ids = samples["input_ids"].to(device)
            attention_mask = samples["attention_mask"].to(device)
            logits = model(input_ids, attention_mask)
            predictions = torch.argmax(logits, dim=1)
            all_labels.append(labels.numpy())
            all_predictions.append(predictions.cpu().numpy())
    return np.concatenate(all_labels).astype(int), np.concatenate(all_predictions).astype(int)


def evaluate_split(model: nn.Module, loader: DataLoader, device: torch.device) -> dict:
    labels, predictions = infer(model, loader, device)
    return compute_metrics(labels, predictions)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)

    output_dir = ensure_dir(Path(args.output_dir).resolve())
    checkpoint_path = output_dir / "best_model.pth"
    summary_path = output_dir / "train_eval_summary.json"
    predictions_path = output_dir / "external_predictions.csv"

    train_df, valid_df = read_dataset(Path(args.train_csv), args)
    external_df = read_external_dataset(Path(args.external_csv), args)

    tokenizer = AutoTokenizer.from_pretrained(str(Path(args.esm_model_dir).resolve()))
    train_dataset = PLPTPDataset(train_df, tokenizer, args.sequence_col, args.label_col, args.max_length)
    valid_dataset = PLPTPDataset(valid_df, tokenizer, args.sequence_col, args.label_col, args.max_length)
    external_dataset = PLPTPDataset(external_df, tokenizer, args.sequence_col, args.label_col, args.max_length)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    external_loader = DataLoader(
        external_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    model = PLPTPModel(Path(args.esm_model_dir).resolve())
    init_info = load_initial_weights(model, Path(args.init_weights).resolve())
    maybe_freeze_bert(model, args.freeze_bert)
    model = model.to(device)

    optimizer = torch.optim.Adam(
        [param for param in model.parameters() if param.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    criterion = FocalLoss()

    best_state_dict = None
    best_epoch = -1
    best_valid_mcc = -math.inf
    history = []

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            grad_accum_steps=max(1, args.grad_accum_steps),
        )
        train_metrics = evaluate_split(model, train_loader, device)
        valid_metrics = evaluate_split(model, valid_loader, device)

        epoch_record = {
            "epoch": epoch,
            "train_loss": float(train_loss),
            "train_metrics": train_metrics,
            "valid_metrics": valid_metrics,
        }
        history.append(epoch_record)
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

        if valid_metrics["mcc"] > best_valid_mcc:
            best_valid_mcc = valid_metrics["mcc"]
            best_epoch = epoch
            best_state_dict = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            torch.save(best_state_dict, checkpoint_path)

    if best_state_dict is None:
        raise RuntimeError("Training finished without producing a best checkpoint")

    model.load_state_dict(best_state_dict, strict=True)
    external_labels, external_predictions = infer(model, external_loader, device)
    external_metrics = compute_metrics(external_labels, external_predictions)

    merged = external_df.copy()
    merged["prediction"] = external_predictions
    merged.to_csv(predictions_path, index=False)

    summary = {
        "train_csv": str(Path(args.train_csv).resolve()),
        "external_csv": str(Path(args.external_csv).resolve()),
        "output_dir": str(output_dir),
        "checkpoint": str(checkpoint_path),
        "esm_model_dir": str(Path(args.esm_model_dir).resolve()),
        "device": str(device),
        "seed": int(args.seed),
        "batch_size": int(args.batch_size),
        "epochs": int(args.epochs),
        "lr": float(args.lr),
        "weight_decay": float(args.weight_decay),
        "grad_accum_steps": int(args.grad_accum_steps),
        "freeze_bert": bool(args.freeze_bert),
        "train_rows": int(len(train_df)),
        "valid_rows": int(len(valid_df)),
        "external_rows": int(len(external_df)),
        "initialization": init_info,
        "best_epoch": int(best_epoch),
        "best_valid_mcc": float(best_valid_mcc),
        "external_metrics": external_metrics,
        "history": history,
        "predictions_csv": str(predictions_path),
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print("best_epoch=%d" % best_epoch)
    print("external_accuracy=%.6f" % external_metrics["accuracy"])
    print("external_sensitivity=%.6f" % external_metrics["sensitivity"])
    print("external_specificity=%.6f" % external_metrics["specificity"])
    print("external_mcc=%.6f" % external_metrics["mcc"])
    print("checkpoint=%s" % checkpoint_path)


if __name__ == "__main__":
    main()
