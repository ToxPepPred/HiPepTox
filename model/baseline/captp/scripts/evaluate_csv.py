#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, confusion_matrix, matthews_corrcoef, recall_score, roc_auc_score
from torch.utils.data import DataLoader, Dataset


PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "data"
MODEL_PATH = DATA_DIR / "model_saved.pkl"
AAINDEX_PATH = DATA_DIR / "AAindex.pkl"
MAX_TOKENS = 51

sys.path.insert(0, str(PROJECT_DIR))

from model import CAPTP_model


class SequenceDataset(Dataset):
    def __init__(self, encoded: np.ndarray) -> None:
        self.encoded = encoded

    def __len__(self) -> int:
        return len(self.encoded)

    def __getitem__(self, index: int) -> torch.Tensor:
        return torch.from_numpy(self.encoded[index]).long()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate CAPTP on a labeled peptide CSV and export row-level probabilities."
    )
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--sequence-col", default="sequence")
    parser.add_argument("--label-col", default="toxicity")
    parser.add_argument("--id-col", default="id")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--weights", default=str(MODEL_PATH))
    parser.add_argument("--aaindex", default=str(AAINDEX_PATH))
    return parser.parse_args()


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


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


def run_inference(model, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    pred_labels = []
    pred_probs = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            logits = model.get_logits(batch)
            probabilities = F.softmax(logits, dim=1)[:, 1]
            labels = torch.argmax(logits, dim=1)
            pred_probs.append(probabilities.detach().cpu().numpy())
            pred_labels.append(labels.detach().cpu().numpy())
    return np.concatenate(pred_labels), np.concatenate(pred_probs)


def main() -> None:
    args = parse_args()
    output_dir = ensure_dir(Path(args.output_dir).resolve())
    input_csv = Path(args.input_csv).resolve()
    device = resolve_device(args.device)

    vocabulary = pickle.load(open(args.aaindex, "rb"))
    df = pd.read_csv(input_csv)
    for col in [args.sequence_col, args.label_col]:
        if col not in df.columns:
            raise ValueError(f"Missing required column `{col}` in {input_csv}")
    if args.id_col not in df.columns:
        df[args.id_col] = [f"sample_{i:06d}" for i in range(len(df))]

    df = df.copy()
    df["sequence_clean"] = df[args.sequence_col].map(lambda s: sanitize_sequence(s, vocabulary))
    df["label"] = df[args.label_col].map(coerce_label).astype(int)
    invalid_mask = df["sequence_clean"].str.len() == 0
    filtered_df = df.loc[~invalid_mask].reset_index(drop=True)

    encoded = encode_sequences(filtered_df["sequence_clean"].tolist(), vocabulary)
    loader = DataLoader(SequenceDataset(encoded), batch_size=args.batch_size, shuffle=False, drop_last=False)

    model = CAPTP_model.CAPTP().to(device)
    state_dict = torch.load(args.weights, map_location=device)
    model.load_state_dict(state_dict, strict=True)

    pred_labels, pred_probs = run_inference(model, loader, device)
    labels = filtered_df["label"].to_numpy()
    metrics = compute_metrics(labels, pred_labels, pred_probs)

    predictions = filtered_df.copy()
    predictions["predicted_label"] = pred_labels.astype(int)
    predictions["prob_toxic"] = pred_probs.astype(float)
    predictions["prediction_text"] = np.where(pred_labels == 1, "Toxic peptide", "Non toxic peptide")
    predictions_path = output_dir / "predictions.csv"
    predictions.to_csv(predictions_path, index=False)

    summary = {
        "input_csv": str(input_csv),
        "weights": str(Path(args.weights).resolve()),
        "device": str(device),
        "filtered_invalid_sequences": int(invalid_mask.sum()),
        "num_rows_evaluated": int(len(filtered_df)),
        "metrics": metrics,
        "predictions_csv": str(predictions_path),
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
