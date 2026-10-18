#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, EsmConfig, EsmModel


PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_ESM_DIR = PROJECT_DIR / "local_models" / "esm2_t30_150M_UR50D"
DEFAULT_WEIGHTS = PROJECT_DIR / "model" / "model.pkl"

AA_ALLOWED = set("LAGVSERTIDPKQNFYMHWCXB UZO.-")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run PLPTP on a labeled CSV and compute evaluation metrics."
    )
    parser.add_argument("--input-csv", required=True, help="Input CSV file.")
    parser.add_argument("--sequence-col", default="sequence", help="Sequence column name.")
    parser.add_argument("--label-col", default="toxicity", help="Binary label column name.")
    parser.add_argument("--id-col", default="id", help="Identifier column name.")
    parser.add_argument(
        "--dataset-name",
        default=None,
        help="Optional dataset name used in output files.",
    )
    parser.add_argument("--run-dir", required=True, help="Directory for outputs.")
    parser.add_argument(
        "--weights",
        default=str(DEFAULT_WEIGHTS),
        help="Path to the PLPTP trained state_dict.",
    )
    parser.add_argument(
        "--esm-model-dir",
        default=str(DEFAULT_ESM_DIR),
        help="Local directory containing ESM2 config/tokenizer files.",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help='Device for inference. Use "cuda" or "cpu".',
    )
    parser.add_argument("--batch-size", type=int, default=64, help="Inference batch size.")
    parser.add_argument("--max-length", type=int, default=50, help="Tokenizer max_length.")
    parser.add_argument("--output-json", required=True, help="Path to save metrics JSON.")
    parser.add_argument(
        "--predictions-csv",
        required=True,
        help="Path to save row-level predictions.",
    )
    return parser.parse_args()


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def resolve_device(device_str: str) -> torch.device:
    if device_str == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def sanitize_sequence(sequence: str) -> str:
    sequence = str(sequence).strip().upper()
    return "".join(residue if residue in AA_ALLOWED else "X" for residue in sequence)


class PLPTPDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        tokenizer,
        sequence_col: str,
        label_col: str,
        max_length: int,
    ) -> None:
        self.df = df.reset_index(drop=True)
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
        out2 = self.fc1(features)
        out3 = self.fc2(out2)
        out3 = self.relu(out3)
        return self.fc3(out3)


def safe_specificity(labels: np.ndarray, predictions: np.ndarray) -> tuple[float, int, int, int, int]:
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    specificity = float(tn) / float(tn + fp) if (tn + fp) else 0.0
    return specificity, int(tn), int(fp), int(fn), int(tp)


def safe_auc(labels: np.ndarray, probabilities: np.ndarray) -> float:
    if len(np.unique(labels)) < 2:
        return float("nan")
    return float(roc_auc_score(labels, probabilities))


def safe_aupr(labels: np.ndarray, probabilities: np.ndarray) -> float:
    if len(np.unique(labels)) < 2:
        return float("nan")
    return float(average_precision_score(labels, probabilities))


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device):
    model.eval()
    labels_all = []
    predictions_all = []
    probabilities_all = []

    with torch.no_grad():
        for samples, labels in loader:
            input_ids = samples["input_ids"].to(device)
            attention_mask = samples["attention_mask"].to(device)
            logits = model(input_ids, attention_mask)
            probabilities = torch.softmax(logits, dim=1)[:, 1]
            predictions = torch.argmax(logits, dim=1)
            labels_all.append(labels.numpy())
            predictions_all.append(predictions.cpu().numpy())
            probabilities_all.append(probabilities.cpu().numpy())

    return (
        np.concatenate(labels_all).astype(int),
        np.concatenate(predictions_all).astype(int),
        np.concatenate(probabilities_all).astype(np.float64),
    )


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    run_dir = ensure_dir(Path(args.run_dir).resolve())
    output_json = Path(args.output_json).resolve()
    predictions_csv = Path(args.predictions_csv).resolve()
    ensure_dir(output_json.parent)
    ensure_dir(predictions_csv.parent)

    df = pd.read_csv(args.input_csv)
    for col in [args.sequence_col, args.label_col, args.id_col]:
        if col not in df.columns:
            raise ValueError(f"Missing required column '{col}' in {args.input_csv}")

    dataset_name = args.dataset_name or Path(args.input_csv).stem
    tokenizer = AutoTokenizer.from_pretrained(str(Path(args.esm_model_dir).resolve()))
    dataset = PLPTPDataset(df, tokenizer, args.sequence_col, args.label_col, args.max_length)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = PLPTPModel(Path(args.esm_model_dir).resolve())
    state_dict = torch.load(args.weights, map_location="cpu")
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    allowed_missing = set()
    allowed_unexpected = {"bert.embeddings.position_ids"}
    if set(missing) != allowed_missing or not set(unexpected).issubset(allowed_unexpected):
        raise RuntimeError(
            f"State dict mismatch. missing={missing} unexpected={unexpected}"
        )
    model = model.to(device)

    labels, predictions, probabilities = evaluate(model, loader, device)
    specificity, tn, fp, fn, tp = safe_specificity(labels, predictions)
    metrics = {
        "dataset": dataset_name,
        "num_samples": int(len(df)),
        "accuracy": float(accuracy_score(labels, predictions)),
        "sensitivity": float(recall_score(labels, predictions, zero_division=0)),
        "specificity": specificity,
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "mcc": float(matthews_corrcoef(labels, predictions)),
        "auc": safe_auc(labels, probabilities),
        "aupr": safe_aupr(labels, probabilities),
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }

    merged = df.copy()
    merged["prediction"] = predictions
    merged["prob_toxic"] = probabilities
    merged.to_csv(predictions_csv, index=False)
    output_json.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(f"dataset={metrics['dataset']}")
    print(f"num_samples={metrics['num_samples']}")
    print(f"accuracy={metrics['accuracy']:.6f}")
    print(f"sensitivity={metrics['sensitivity']:.6f}")
    print(f"specificity={metrics['specificity']:.6f}")
    print(f"f1={metrics['f1']:.6f}")
    print(f"mcc={metrics['mcc']:.6f}")
    print(f"auc={metrics['auc']:.6f}")
    print(f"aupr={metrics['aupr']:.6f}")
    print(f"confusion_matrix=TN:{metrics['tn']} FP:{metrics['fp']} FN:{metrics['fn']} TP:{metrics['tp']}")


if __name__ == "__main__":
    main()
