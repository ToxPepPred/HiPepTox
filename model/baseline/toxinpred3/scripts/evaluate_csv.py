#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    matthews_corrcoef,
    recall_score,
)


PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = PROJECT_DIR / "model" / "toxinpred3.0_model.pkl"
DEFAULT_MAIN = PROJECT_DIR / "toxinpred3.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run ToxinPred3 on a labeled CSV and compute evaluation metrics."
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
    parser.add_argument(
        "--run-dir",
        required=True,
        help="Directory for FASTA, raw ToxinPred3 output, and metrics.",
    )
    parser.add_argument(
        "--toxinpred3-script",
        default=str(DEFAULT_MAIN),
        help="Path to toxinpred3.py.",
    )
    parser.add_argument(
        "--model",
        type=int,
        default=2,
        choices=[1, 2],
        help="ToxinPred3 model: 1=ML, 2=Hybrid.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.38,
        help="Classification threshold passed to ToxinPred3.",
    )
    parser.add_argument(
        "--python-bin",
        default=sys.executable,
        help="Python executable used to run toxinpred3.py.",
    )
    parser.add_argument(
        "--output-json",
        required=True,
        help="Path to save metrics JSON.",
    )
    parser.add_argument(
        "--predictions-csv",
        required=True,
        help="Path to save merged row-level predictions.",
    )
    return parser.parse_args()


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def sanitize_sequence(sequence: str) -> str:
    allowed = set("ARNDCQEGHILKMFPSTWYV-")
    sequence = str(sequence).strip().upper()
    return "".join(residue for residue in sequence if residue in allowed)


def write_fasta(df: pd.DataFrame, id_col: str, sequence_col: str, fasta_path: Path) -> None:
    lines: list[str] = []
    for _, row in df.iterrows():
        seq_id = str(row[id_col]).strip()
        sequence = sanitize_sequence(row[sequence_col])
        lines.append(f">{seq_id}")
        lines.append(sequence)
    fasta_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_toxinpred3(args: argparse.Namespace, fasta_path: Path, raw_output_path: Path) -> None:
    work_dir = ensure_dir(fasta_path.parent / "workspace")
    model_link = work_dir / "model"
    if not model_link.exists():
        os.symlink(DEFAULT_MODEL.parent, model_link, target_is_directory=True)

    cmd = [
        args.python_bin,
        str(Path(args.toxinpred3_script).resolve()),
        "-i",
        str(fasta_path),
        "-o",
        str(raw_output_path),
        "-m",
        str(args.model),
        "-t",
        str(args.threshold),
        "-d",
        "2",
    ]
    subprocess.run(cmd, cwd=work_dir, check=True)


def safe_specificity(labels: pd.Series, predictions: pd.Series) -> tuple[float, int, int, int, int]:
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    specificity = float(tn) / float(tn + fp) if (tn + fp) else 0.0
    return specificity, int(tn), int(fp), int(fn), int(tp)


def main() -> None:
    args = parse_args()
    run_dir = ensure_dir(Path(args.run_dir).resolve())
    output_json = Path(args.output_json).resolve()
    predictions_csv = Path(args.predictions_csv).resolve()
    ensure_dir(output_json.parent)
    ensure_dir(predictions_csv.parent)

    model_path = DEFAULT_MODEL
    if not model_path.exists():
        raise FileNotFoundError(
            f"Expected model file not found: {model_path}. "
            "Unzip model/toxinpred3.0_model.pkl.zip first."
        )

    df = pd.read_csv(args.input_csv)
    required_cols = {args.sequence_col, args.label_col, args.id_col}
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {args.input_csv}: {missing}")

    dataset_name = args.dataset_name or Path(args.input_csv).stem
    fasta_path = run_dir / f"{dataset_name}.fa"
    raw_output_path = run_dir / f"{dataset_name}_toxinpred3_raw.csv"

    write_fasta(df, args.id_col, args.sequence_col, fasta_path)
    run_toxinpred3(args, fasta_path, raw_output_path)

    raw_pred = pd.read_csv(raw_output_path)
    score_col = "Hybrid Score" if args.model == 2 else "ML Score"
    merged = df.merge(raw_pred, left_on=args.id_col, right_on="Subject", how="left", validate="one_to_one")

    if merged[score_col].isna().any():
        missing_ids = merged.loc[merged[score_col].isna(), args.id_col].astype(str).tolist()[:10]
        raise ValueError(f"Missing predictions for some ids, e.g. {missing_ids}")

    merged["label"] = merged[args.label_col].astype(int)
    merged["prediction_binary"] = (merged["Prediction"] == "Toxin").astype(int)
    merged["score"] = merged[score_col].astype(float)

    labels = merged["label"]
    predictions = merged["prediction_binary"]
    scores = merged["score"]
    specificity, tn, fp, fn, tp = safe_specificity(labels, predictions)

    metrics = {
        "dataset": dataset_name,
        "num_samples": int(len(merged)),
        "model": int(args.model),
        "threshold": float(args.threshold),
        "accuracy": float(accuracy_score(labels, predictions)),
        "sensitivity": float(recall_score(labels, predictions, zero_division=0)),
        "specificity": specificity,
        "mcc": float(matthews_corrcoef(labels, predictions)),
        "aupr": float(average_precision_score(labels, scores)),
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }

    merged.to_csv(predictions_csv, index=False)
    output_json.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(f"dataset={metrics['dataset']}")
    print(f"num_samples={metrics['num_samples']}")
    print(f"accuracy={metrics['accuracy']:.6f}")
    print(f"sensitivity={metrics['sensitivity']:.6f}")
    print(f"specificity={metrics['specificity']:.6f}")
    print(f"mcc={metrics['mcc']:.6f}")
    print(f"aupr={metrics['aupr']:.6f}")
    print(f"confusion_matrix=TN:{metrics['tn']} FP:{metrics['fp']} FN:{metrics['fn']} TP:{metrics['tp']}")


if __name__ == "__main__":
    main()
