#!/usr/bin/env python
import argparse
import csv
import json
import math
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch_geometric
from torch_geometric.loader import DataLoader
from torch_geometric.nn import radius_graph

from PeptiTox import PeptiTox


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_pdb_xyz(pdb_path):
    current_pos = -1000
    x_rows = []
    current_aa = {}
    with open(pdb_path, "r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            record = line[0:4].strip()
            if (record == "ATOM" and int(line[22:26].strip()) != current_pos) or record == "TER":
                if current_aa:
                    required = ("N", "CA", "C", "O")
                    if not all(atom in current_aa for atom in required):
                        missing = [atom for atom in required if atom not in current_aa]
                        raise ValueError("%s missing backbone atoms %s" % (pdb_path, missing))
                    r_group = [coord for atom, coord in current_aa.items() if atom not in required]
                    if not r_group:
                        r_group = [current_aa["CA"]]
                    r_group = np.asarray(r_group, dtype=np.float32).mean(axis=0)
                    x_rows.append([current_aa["N"], current_aa["CA"], current_aa["C"], current_aa["O"], r_group])
                    current_aa = {}
                if record != "TER":
                    current_pos = int(line[22:26].strip())

            if record == "ATOM":
                atom = line[13:16].strip()
                if atom != "H":
                    xyz = np.asarray(
                        [line[30:38].strip(), line[38:46].strip(), line[46:54].strip()],
                        dtype=np.float32,
                    )
                    current_aa[atom] = xyz
    return np.asarray(x_rows, dtype=np.float32)


def load_rows(csv_path, split=None):
    rows = []
    with open(csv_path, "r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if split is None or row.get("split") == split:
                rows.append(row)
    if not rows:
        raise ValueError("No rows loaded from %s with split=%s" % (csv_path, split))
    return rows


def load_cache_dir(structure_feature_file):
    payload = torch.load(structure_feature_file, map_location="cpu")
    cache_dir = payload.get("cache_dir")
    if not cache_dir:
        raise ValueError("cache_dir not found in %s" % structure_feature_file)
    return Path(cache_dir)


def resolve_pdb_path(row, cache_dir):
    explicit_path = row.get("structoxnet_pdb_path")
    if explicit_path and Path(explicit_path).exists():
        return Path(explicit_path)
    sample_id = row.get("structoxnet_sample_id")
    if not sample_id:
        raise ValueError("Row %s has no structoxnet_sample_id" % row.get("id"))
    pdb_path = cache_dir / "pdb" / ("%s.pdb" % sample_id)
    if not pdb_path.exists():
        raise FileNotFoundError("Missing PDB for %s: %s" % (row.get("id"), pdb_path))
    return pdb_path


class CachedPeptiToxDataset(torch.utils.data.Dataset):
    def __init__(self, rows, plm_by_sequence, cache_dir, radius=15.0, max_neighbors=1000):
        self.graphs = []
        for idx, row in enumerate(rows, start=1):
            sample_id = row["id"]
            sequence = row["sequence"].strip().upper()
            label = float(row["toxicity"])
            if sequence not in plm_by_sequence:
                raise KeyError("Missing PLM embedding for sequence %s (%s)" % (sequence, sample_id))
            h_v = plm_by_sequence[sequence].float().contiguous()
            pdb_path = resolve_pdb_path(row, cache_dir)
            x = torch.from_numpy(get_pdb_xyz(pdb_path)).float()
            if x.shape[0] != len(sequence) or h_v.shape[0] != len(sequence):
                raise ValueError(
                    "Length mismatch for %s: seq=%d pdb=%d plm=%d"
                    % (sample_id, len(sequence), x.shape[0], h_v.shape[0])
                )
            edge_index = radius_graph(
                x[:, 1],
                r=radius,
                loop=True,
                max_num_neighbors=max_neighbors,
                num_workers=1,
            ).contiguous()
            self.graphs.append(
                torch_geometric.data.Data(
                    name=sample_id,
                    sequence=sequence,
                    X=x,
                    node_feat=h_v,
                    edge_index=edge_index,
                    y=torch.tensor(label, dtype=torch.float32),
                )
            )
            if idx % 1000 == 0:
                print("prepared %d / %d graphs" % (idx, len(rows)), flush=True)

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        return self.graphs[idx]


def binary_metrics(y_true, y_pred):
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    accuracy = (tp + tn) / max(tp + tn + fp + fn, 1)
    sensitivity = tp / (tp + fn) if (tp + fn) else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    denom = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = ((tp * tn - fp * fn) / denom) if denom else 0.0
    return {
        "Accuracy": accuracy,
        "Sensitivity": sensitivity,
        "Specificity": specificity,
        "MCC": mcc,
        "TP": tp,
        "TN": tn,
        "FP": fp,
        "FN": fn,
    }


def run_epoch(model, loader, criterion, device, optimizer=None):
    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0
    all_probs = []
    all_labels = []
    all_names = []
    all_sequences = []
    with torch.set_grad_enabled(is_train):
        for batch in loader:
            batch = batch.to(device)
            if is_train:
                optimizer.zero_grad(set_to_none=True)
            logits = model(X=batch.X, h_V=batch.node_feat, edge_index=batch.edge_index, batch_id=batch.batch)
            labels = batch.y.float()
            loss = criterion(logits, labels)
            if is_train:
                loss.backward()
                optimizer.step()
            total_loss += float(loss.item()) * batch.num_graphs
            probs = torch.sigmoid(logits.detach()).cpu().numpy()
            all_probs.extend(probs.tolist())
            all_labels.extend(labels.detach().cpu().numpy().astype(int).tolist())
            all_names.extend(batch.name)
            all_sequences.extend(batch.sequence)
    preds = (np.asarray(all_probs) >= 0.5).astype(int).tolist()
    metrics = binary_metrics(all_labels, preds)
    return total_loss / len(loader.dataset), metrics, all_names, all_sequences, all_labels, all_probs, preds


def format_metrics(metrics):
    return " ".join("%s=%.4f" % (key, metrics[key]) for key in ["Accuracy", "Sensitivity", "Specificity", "MCC"])


def main():
    parser = argparse.ArgumentParser(description="Train PeptiTox on cached PDB + residue ESM features and test externally.")
    parser.add_argument("--train-data-dir", required=True)
    parser.add_argument("--external-data-dir", required=True)
    parser.add_argument("--plm-embedding-file", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--augment-eps", type=float, default=0.1)
    parser.add_argument("--radius", type=float, default=15.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--limit-train", type=int, default=0, help="Debug only: keep the first N train rows.")
    parser.add_argument("--limit-valid", type=int, default=0, help="Debug only: keep the first N validation rows.")
    parser.add_argument("--limit-external", type=int, default=0, help="Debug only: keep the first N external rows.")
    args = parser.parse_args()

    seed_everything(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_dir = Path(args.train_data_dir)
    external_dir = Path(args.external_data_dir)
    train_csv = train_dir / "peptide_toxicity_structoxnet_mapping.csv"
    external_csv = external_dir / "peptide_toxicity_structoxnet_mapping.csv"
    train_cache_dir = load_cache_dir(train_dir / "structoxnet_residue_features.pt")
    external_cache_dir = load_cache_dir(external_dir / "structoxnet_residue_features.pt")
    plm_payload = torch.load(args.plm_embedding_file, map_location="cpu")
    plm_by_sequence = plm_payload["by_sequence"]
    plm_dim = int(plm_payload.get("embedding_dim") or next(iter(plm_by_sequence.values())).shape[1])

    train_rows = load_rows(train_csv, split="train")
    valid_rows = load_rows(train_csv, split="validation")
    external_rows = load_rows(external_csv, split=None)
    if args.limit_train:
        train_rows = train_rows[: args.limit_train]
    if args.limit_valid:
        valid_rows = valid_rows[: args.limit_valid]
    if args.limit_external:
        external_rows = external_rows[: args.limit_external]

    print("train=%d validation=%d external=%d plm_dim=%d" % (len(train_rows), len(valid_rows), len(external_rows), plm_dim), flush=True)
    print("building cached graph datasets", flush=True)
    train_dataset = CachedPeptiToxDataset(train_rows, plm_by_sequence, train_cache_dir, radius=args.radius)
    valid_dataset = CachedPeptiToxDataset(valid_rows, plm_by_sequence, train_cache_dir, radius=args.radius)
    external_dataset = CachedPeptiToxDataset(external_rows, plm_by_sequence, external_cache_dir, radius=args.radius)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    external_loader = DataLoader(external_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    model = PeptiTox(
        node_input_dim=plm_dim + 184,
        edge_input_dim=450,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        augment_eps=args.augment_eps,
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.BCEWithLogitsLoss()

    best_score = -1e9
    best_epoch = 0
    best_payload = None
    stale = 0
    log_rows = []
    for epoch in range(1, args.epochs + 1):
        train_loss, train_metrics, *_ = run_epoch(model, train_loader, criterion, device, optimizer=optimizer)
        valid_loss, valid_metrics, *_ = run_epoch(model, valid_loader, criterion, device)
        score = valid_metrics["MCC"]
        log_row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "valid_loss": valid_loss,
            **{"train_" + k: v for k, v in train_metrics.items()},
            **{"valid_" + k: v for k, v in valid_metrics.items()},
        }
        log_rows.append(log_row)
        print(
            "epoch=%03d train_loss=%.4f valid_loss=%.4f valid_%s"
            % (epoch, train_loss, valid_loss, format_metrics(valid_metrics)),
            flush=True,
        )
        if score > best_score:
            best_score = score
            best_epoch = epoch
            stale = 0
            best_payload = {
                "model_state_dict": model.state_dict(),
                "args": vars(args),
                "best_epoch": best_epoch,
                "valid_metrics": valid_metrics,
                "plm_dim": plm_dim,
            }
            torch.save(best_payload, output_dir / "best_model.pth")
        else:
            stale += 1
            if stale >= args.patience:
                print("early_stop epoch=%d best_epoch=%d" % (epoch, best_epoch), flush=True)
                break

    pd.DataFrame(log_rows).to_csv(output_dir / "training_log.csv", index=False)
    if best_payload is None:
        raise RuntimeError("No model checkpoint was saved")
    model.load_state_dict(torch.load(output_dir / "best_model.pth", map_location=device)["model_state_dict"])
    external_loss, external_metrics, names, sequences, labels, probs, preds = run_epoch(model, external_loader, criterion, device)
    external_metrics = dict(external_metrics)
    external_metrics["loss"] = external_loss
    external_metrics["best_epoch"] = best_epoch

    predictions = pd.DataFrame(
        {
            "id": names,
            "sequence": sequences,
            "label": labels,
            "probability": probs,
            "prediction": preds,
        }
    )
    predictions.to_csv(output_dir / "external_predictions.csv", index=False)
    with open(output_dir / "external_metrics.json", "w", encoding="utf-8") as handle:
        json.dump(external_metrics, handle, ensure_ascii=False, indent=2)
    pd.DataFrame([external_metrics]).to_csv(output_dir / "external_metrics.csv", index=False)
    print("external_loss=%.4f external_%s best_epoch=%d" % (external_loss, format_metrics(external_metrics), best_epoch), flush=True)
    print("saved_dir=%s" % output_dir, flush=True)


if __name__ == "__main__":
    main()
