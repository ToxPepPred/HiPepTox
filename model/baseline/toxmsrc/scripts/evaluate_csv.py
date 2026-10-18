#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    recall_score,
    roc_auc_score,
)

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_DIR))

from model import ourmodel  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate ToxMSRC on a labeled CSV using precomputed Word2Vec features."
    )
    parser.add_argument("--input-csv", required=True, help="Input CSV file.")
    parser.add_argument(
        "--features-npz",
        required=True,
        help="Feature file produced by build_word2vec_features.py.",
    )
    parser.add_argument(
        "--label-col",
        default="toxicity",
        help="Column name containing binary labels.",
    )
    parser.add_argument(
        "--weights",
        default=str(PROJECT_DIR / "model" / "ToxMSRC.h5"),
        help="Path to trained ToxMSRC weights.",
    )
    parser.add_argument(
        "--output-json",
        default=None,
        help="Optional path to save metrics as JSON.",
    )
    return parser.parse_args()


def safe_divide(numerator: float, denominator: float) -> float:
    return float(numerator) / float(denominator) if denominator else 0.0


def main() -> None:
    args = parse_args()
    input_csv = Path(args.input_csv).resolve()
    features_npz = Path(args.features_npz).resolve()
    weights = Path(args.weights).resolve()

    df = pd.read_csv(input_csv)
    if args.label_col not in df.columns:
        raise ValueError(
            f"Column '{args.label_col}' not found in {input_csv}. "
            f"Available columns: {list(df.columns)}"
        )

    labels = df[args.label_col].astype(int).to_numpy().reshape(-1)
    features = np.load(features_npz)["x"]
    if len(labels) != len(features):
        raise ValueError(
            f"Row mismatch: labels={len(labels)} features={len(features)}"
        )

    model = ourmodel()
    model.load_weights(str(weights))

    probabilities = model.predict([features], verbose=0)
    predictions = probabilities.argmax(axis=1)

    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()

    metrics = {
        "num_samples": int(len(labels)),
        "accuracy": float(accuracy_score(labels, predictions)),
        "sensitivity": float(recall_score(labels, predictions, zero_division=0)),
        "specificity": safe_divide(tn, tn + fp),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "mcc": float(matthews_corrcoef(labels, predictions)),
        "auc": float(roc_auc_score(labels, probabilities[:, 1])),
        "aupr": float(average_precision_score(labels, probabilities[:, 1])),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }

    if args.output_json:
        output_json = Path(args.output_json).resolve()
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(f"dataset={input_csv}")
    print(f"num_samples={metrics['num_samples']}")
    print(f"accuracy={metrics['accuracy']:.6f}")
    print(f"sensitivity={metrics['sensitivity']:.6f}")
    print(f"specificity={metrics['specificity']:.6f}")
    print(f"f1={metrics['f1']:.6f}")
    print(f"mcc={metrics['mcc']:.6f}")
    print(f"auc={metrics['auc']:.6f}")
    print(f"aupr={metrics['aupr']:.6f}")
    print(
        f"confusion_matrix=TN:{metrics['tn']} FP:{metrics['fp']} "
        f"FN:{metrics['fn']} TP:{metrics['tp']}"
    )


if __name__ == "__main__":
    main()
