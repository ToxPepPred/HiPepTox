#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    recall_score,
    roc_auc_score,
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_DIR))

from model import ourmodel  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train and evaluate ToxMSRC on labeled CSV datasets using precomputed "
            "Word2Vec features."
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
    parser.add_argument(
        "--label-col",
        default="toxicity",
        help="Label column name in both CSV files.",
    )
    parser.add_argument(
        "--split-col",
        default="split",
        help="Split column name in the training CSV.",
    )
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
    parser.add_argument(
        "--init-weights",
        default=str(PROJECT_DIR / "model" / "ToxMSRC.h5"),
        help="Optional initial weights file. Leave empty to train from scratch.",
    )
    parser.add_argument("--epochs", type=int, default=20, help="Maximum training epochs.")
    parser.add_argument("--batch-size", type=int, default=32, help="Mini-batch size.")
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-4,
        help="Learning rate for Adam.",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=5,
        help="Early stopping patience measured on validation MCC.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--verbose",
        type=int,
        default=0,
        choices=[0, 1, 2],
        help="Verbosity passed to keras fit/predict.",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def safe_divide(numerator: float, denominator: float) -> float:
    return float(numerator) / float(denominator) if denominator else 0.0


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


def load_features(path: str) -> np.ndarray:
    features = np.load(Path(path).resolve())
    if "x" not in features:
        raise KeyError(f"Feature file {path} does not contain key 'x'.")
    return features["x"]


def build_model(learning_rate: float) -> tf.keras.Model:
    model = ourmodel()
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def evaluate_model(
    model: tf.keras.Model,
    features: np.ndarray,
    labels: np.ndarray,
    verbose: int,
) -> tuple[dict[str, float | int], np.ndarray, np.ndarray]:
    probabilities = model.predict(features, verbose=verbose)
    metrics = compute_metrics(labels, probabilities)
    predictions = probabilities.argmax(axis=1)
    return metrics, probabilities, predictions


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

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

    train_labels_oh = to_categorical(train_labels, num_classes=2, dtype="float32")

    model = build_model(args.learning_rate)
    init_weights = str(args.init_weights).strip()
    init_weights_path = None
    if init_weights:
        init_weights_path = Path(init_weights).resolve()
        model.load_weights(str(init_weights_path))

    best_mcc_path = output_dir / "best_model_by_valid_mcc.h5"
    best_acc_path = output_dir / "best_model_by_valid_acc.h5"

    history: list[dict[str, float | int]] = []
    best_mcc = -float("inf")
    best_acc = -float("inf")
    best_epoch_by_mcc = 0
    best_epoch_by_acc = 0
    best_valid_mcc_metrics: dict[str, float | int] | None = None
    best_valid_acc_metrics: dict[str, float | int] | None = None
    epochs_without_improvement = 0

    for epoch in range(1, args.epochs + 1):
        fit_history = model.fit(
            train_features,
            train_labels_oh,
            batch_size=args.batch_size,
            epochs=1,
            shuffle=True,
            verbose=args.verbose,
        )

        valid_metrics, _, _ = evaluate_model(
            model, valid_features, valid_labels, verbose=args.verbose
        )
        epoch_record = {
            "epoch": epoch,
            "train_loss": float(fit_history.history["loss"][0]),
            "train_accuracy": float(fit_history.history["accuracy"][0]),
            **valid_metrics,
        }
        history.append(epoch_record)

        improved_mcc = valid_metrics["mcc"] > best_mcc
        improved_acc = valid_metrics["accuracy"] > best_acc

        if improved_mcc:
            best_mcc = float(valid_metrics["mcc"])
            best_epoch_by_mcc = epoch
            best_valid_mcc_metrics = dict(valid_metrics)
            model.save_weights(str(best_mcc_path))
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if improved_acc:
            best_acc = float(valid_metrics["accuracy"])
            best_epoch_by_acc = epoch
            best_valid_acc_metrics = dict(valid_metrics)
            model.save_weights(str(best_acc_path))

        print(
            f"epoch={epoch} train_loss={epoch_record['train_loss']:.6f} "
            f"train_acc={epoch_record['train_accuracy']:.6f} "
            f"valid_acc={valid_metrics['accuracy']:.6f} "
            f"valid_mcc={valid_metrics['mcc']:.6f} "
            f"valid_sn={valid_metrics['sensitivity']:.6f} "
            f"valid_sp={valid_metrics['specificity']:.6f}"
        )

        if epochs_without_improvement >= args.patience:
            print(
                f"Early stopping triggered at epoch={epoch} "
                f"(patience={args.patience}, metric=valid_mcc)."
            )
            break

    if best_epoch_by_mcc == 0 or best_valid_mcc_metrics is None:
        raise RuntimeError("No best checkpoint selected by validation MCC.")
    if best_epoch_by_acc == 0 or best_valid_acc_metrics is None:
        raise RuntimeError("No best checkpoint selected by validation ACC.")

    best_mcc_model = build_model(args.learning_rate)
    best_mcc_model.load_weights(str(best_mcc_path))
    external_metrics_by_mcc, external_prob_by_mcc, external_pred_by_mcc = evaluate_model(
        best_mcc_model,
        external_features,
        external_labels,
        verbose=args.verbose,
    )

    best_acc_model = build_model(args.learning_rate)
    best_acc_model.load_weights(str(best_acc_path))
    external_metrics_by_acc, external_prob_by_acc, external_pred_by_acc = evaluate_model(
        best_acc_model,
        external_features,
        external_labels,
        verbose=args.verbose,
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
        "init_weights": str(init_weights_path) if init_weights_path else None,
        "num_train_samples": int(len(train_features)),
        "num_validation_samples": int(len(valid_features)),
        "num_external_samples": int(len(external_features)),
        "feature_shape": list(train_features.shape[1:]),
        "training_config": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "patience": args.patience,
            "seed": args.seed,
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
