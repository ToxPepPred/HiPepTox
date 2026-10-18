#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from scripts.tamper_csv_common import (  # noqa: E402
    build_dataset,
    build_model,
    ensure_dir,
    evaluate_model,
    load_all_samples,
    resolve_device,
    verify_pdbs,
    write_predictions_csv,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate tAMPer on a labeled CSV and existing PDB cache.")
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--pdb-dir", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--predictions-csv", required=True)
    parser.add_argument("--dataset-name", default=None)
    parser.add_argument("--sequence-col", default="sequence")
    parser.add_argument("--label-col", default="toxicity")
    parser.add_argument("--id-col", default="id")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--embedding-model", default="t12")
    parser.add_argument("--hdim", type=int, default=64)
    parser.add_argument("--gru-layers", type=int, default=1)
    parser.add_argument("--gnn-layers", type=int, default=1)
    parser.add_argument("--modality", default="all")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--d-max", type=int, default=12)
    parser.add_argument("--num-workers", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    pdb_dir = Path(args.pdb_dir).resolve()
    output_json = Path(args.output_json).resolve()
    predictions_csv = Path(args.predictions_csv).resolve()
    output_dir = ensure_dir(output_json.parent)
    fasta_dir = ensure_dir(output_dir / "fastas")

    samples = load_all_samples(
        csv_path=Path(args.input_csv).resolve(),
        sequence_col=args.sequence_col,
        label_col=args.label_col,
        id_col=args.id_col,
    )
    verify_pdbs(samples, pdb_dir)

    dataset = build_dataset(
        samples=samples,
        pdb_dir=pdb_dir,
        embedding_model=args.embedding_model,
        device=device,
        d_max=args.d_max,
        fasta_dir=fasta_dir,
        prefix="eval",
    )

    model = build_model(
        embedding_model=args.embedding_model,
        hdim=args.hdim,
        gru_layers=args.gru_layers,
        gnn_layers=args.gnn_layers,
        modality=args.modality,
    ).to(device=device)
    model.load_state_dict(torch.load(Path(args.weights).resolve(), map_location=device)["model"])

    metrics, rows = evaluate_model(
        model=model,
        dataset=dataset,
        device=device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    metrics["dataset"] = args.dataset_name or Path(args.input_csv).resolve().parent.name
    metrics["num_samples"] = len(samples)

    ensure_dir(predictions_csv.parent)
    write_predictions_csv(predictions_csv, rows)
    output_json.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"accuracy={metrics['accuracy']:.6f}")
    print(f"sensitivity={metrics['sensitivity']:.6f}")
    print(f"specificity={metrics['specificity']:.6f}")
    print(f"f1={metrics['f1']:.6f}")
    print(f"mcc={metrics['mcc']:.6f}")
    print(f"auc={metrics['auc']:.6f}")
    print(f"aupr={metrics['aupr']:.6f}")
    print(
        "confusion_matrix="
        f"TN:{metrics['tn']} FP:{metrics['fp']} FN:{metrics['fn']} TP:{metrics['tp']}"
    )


if __name__ == "__main__":
    main()
