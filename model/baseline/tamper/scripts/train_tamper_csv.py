#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss
from torch.optim import Adam
from torch_geometric.loader import DataLoader

try:
    from loguru import logger
except ImportError:
    import logging

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    logger = logging.getLogger(__name__)

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from scripts.tamper_csv_common import (  # noqa: E402
    TAMPER_DIR,
    build_dataset,
    build_model,
    ensure_dir,
    evaluate_model,
    load_split_samples,
    resolve_device,
    set_seed,
    verify_pdbs,
)

sys.path.insert(0, str(TAMPER_DIR / "src"))
from train_tAMPer import train as tamper_train  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train tAMPer from a split CSV and existing PDB cache.")
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--pdb-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--sequence-col", default="sequence")
    parser.add_argument("--label-col", default="toxicity")
    parser.add_argument("--id-col", default="id")
    parser.add_argument("--split-col", default="split")
    parser.add_argument("--train-split-value", default="train")
    parser.add_argument("--valid-split-value", default="validation")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--embedding-model", default="t12")
    parser.add_argument("--hdim", type=int, default=64)
    parser.add_argument("--gru-layers", type=int, default=1)
    parser.add_argument("--gnn-layers", type=int, default=1)
    parser.add_argument("--modality", default="all")
    parser.add_argument("--lr", type=float, default=4e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-7)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--grad-acc", type=int, default=1)
    parser.add_argument("--d-max", type=int, default=12)
    parser.add_argument("--lammy", type=float, default=0.2)
    parser.add_argument("--monitor", default="accuracy")
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--pre-gnn-checkpoint",
        default=str(TAMPER_DIR / "checkpoints" / "trained" / "pre_GNNs.pt"),
    )
    return parser.parse_args()


def best_epoch_from_history(history: List[Dict[str, float]], monitor: str) -> int:
    if monitor == "loss":
        best_idx = min(range(len(history)), key=lambda i: history[i][monitor])
    else:
        best_idx = max(range(len(history)), key=lambda i: history[i][monitor])
    return best_idx + 1


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)
    output_dir = ensure_dir(Path(args.output_dir).resolve())
    fasta_dir = ensure_dir(output_dir / "fastas")
    pdb_dir = Path(args.pdb_dir).resolve()

    _, train_samples, valid_samples = load_split_samples(
        csv_path=Path(args.input_csv).resolve(),
        sequence_col=args.sequence_col,
        label_col=args.label_col,
        id_col=args.id_col,
        split_col=args.split_col,
        train_split_value=args.train_split_value,
        valid_split_value=args.valid_split_value,
    )
    verify_pdbs(train_samples + valid_samples, pdb_dir)

    logger.info(f"device={device}")
    logger.info(f"train_rows={len(train_samples)}")
    logger.info(f"valid_rows={len(valid_samples)}")

    train_data = build_dataset(
        samples=train_samples,
        pdb_dir=pdb_dir,
        embedding_model=args.embedding_model,
        device=device,
        d_max=args.d_max,
        fasta_dir=fasta_dir,
        prefix="train",
    )
    valid_data = build_dataset(
        samples=valid_samples,
        pdb_dir=pdb_dir,
        embedding_model=args.embedding_model,
        device=device,
        d_max=args.d_max,
        fasta_dir=fasta_dir,
        prefix="valid",
    )

    train_loader = DataLoader(dataset=train_data, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True)
    valid_loader = DataLoader(dataset=valid_data, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False)

    model = build_model(
        embedding_model=args.embedding_model,
        hdim=args.hdim,
        gru_layers=args.gru_layers,
        gnn_layers=args.gnn_layers,
        modality=args.modality,
    ).to(device=device)

    pre_gnn_checkpoint = Path(args.pre_gnn_checkpoint).resolve()
    if pre_gnn_checkpoint.exists():
        logger.info(f"Loading pre-trained GNN weights from {pre_gnn_checkpoint}")
        state_dict = torch.load(pre_gnn_checkpoint, map_location=device)["model"]
        model.load_state_dict(state_dict, strict=False)

    loss_function = {"tx": BCEWithLogitsLoss(), "ss": CrossEntropyLoss()}
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, gamma=0.5, step_size=50)

    best_checkpoint = output_dir / "best_valid_model.pt"
    log_path = output_dir / "training_log.npy"
    log_dict = tamper_train(
        model=model,
        max_num_epochs=args.epochs,
        check_val_every=1,
        training_loader=train_loader,
        val_loader=valid_loader,
        loss_fun=loss_function,
        lammy=args.lammy,
        optimizer=optimizer,
        accum_iter=args.grad_acc,
        device=device,
        monitor=args.monitor,
        checkpoint_address=str(best_checkpoint),
        patience=args.patience,
        scheduler=scheduler,
    )
    np.save(log_path, log_dict)

    history: List[Dict[str, float]] = []
    for idx, (train_metrics, valid_metrics) in enumerate(zip(log_dict["train_metrics"], log_dict["val_metrics"]), start=1):
        row: Dict[str, float] = {"epoch": idx}
        for key, value in train_metrics.items():
            row[f"train_{key}"] = value
        for key, value in valid_metrics.items():
            mapped_key = "auc" if key == "auROC" else "aupr" if key == "auPRC" else key
            row[f"valid_{mapped_key}"] = value
        history.append(row)

    history_path = output_dir / "training_history.csv"
    if history:
        import csv

        with history_path.open("w", encoding="utf-8", newline="") as fout:
            writer = csv.DictWriter(fout, fieldnames=list(history[0].keys()))
            writer.writeheader()
            writer.writerows(history)

    best_epoch = best_epoch_from_history(log_dict["val_metrics"], args.monitor)

    best_model = build_model(
        embedding_model=args.embedding_model,
        hdim=args.hdim,
        gru_layers=args.gru_layers,
        gnn_layers=args.gnn_layers,
        modality=args.modality,
    ).to(device=device)
    best_model.load_state_dict(torch.load(best_checkpoint, map_location=device)["model"])
    best_valid_metrics, _ = evaluate_model(
        model=best_model,
        dataset=valid_data,
        device=device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    summary = {
        "input_csv": str(Path(args.input_csv).resolve()),
        "pdb_dir": str(pdb_dir),
        "output_dir": str(output_dir),
        "train_rows": len(train_samples),
        "valid_rows": len(valid_samples),
        "select_metric": args.monitor,
        "best_epoch": best_epoch,
        "best_checkpoint": str(best_checkpoint),
        "embedding_model": args.embedding_model,
        "best_valid_metrics": best_valid_metrics,
    }
    summary_path = output_dir / "training_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"best_epoch={best_epoch}")
    print(f"best_checkpoint={best_checkpoint}")
    print(f"training_history={history_path}")
    print(f"training_summary={summary_path}")


if __name__ == "__main__":
    main()
