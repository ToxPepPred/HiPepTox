#!/usr/bin/env python
"""Evaluate the official ToxGIN model on a CSV dataset."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import dgl
import numpy as np
import pandas as pd
import torch
from biopandas.pdb import PandasPdb
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
)
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoModel, AutoTokenizer


AA1_ALLOWED = set("ACDEFGHIKLMNPQRSTVWYX")
AA3_TO_AA1 = {
    "VAL": "V",
    "PRO": "P",
    "ASN": "N",
    "GLU": "E",
    "ASP": "D",
    "ALA": "A",
    "THR": "T",
    "SER": "S",
    "LEU": "L",
    "LYS": "K",
    "GLY": "G",
    "GLN": "Q",
    "ILE": "I",
    "PHE": "F",
    "CYS": "C",
    "TRP": "W",
    "ARG": "R",
    "TYR": "Y",
    "HIS": "H",
    "MET": "M",
}


@dataclass(frozen=True)
class Sample:
    sample_id: str
    original_id: str
    sequence: str
    label: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", required=True, help="CSV file with sequences and labels.")
    parser.add_argument("--pdb-dir", required=True, help="Directory with sample_id-named PDB files.")
    parser.add_argument("--run-dir", required=True, help="Directory for ToxGIN intermediate files.")
    parser.add_argument("--weights", required=True, help="Path to ToxGIN best_model.pth.")
    parser.add_argument(
        "--aaindex",
        default=str(Path(__file__).resolve().parents[1] / "aaindex1.csv"),
        help="Path to ToxGIN aaindex1.csv.",
    )
    parser.add_argument(
        "--esm-model",
        default="facebook/esm2_t36_3B_UR50D",
        help="Hugging Face model id or local path for the ESM2 encoder.",
    )
    parser.add_argument("--sequence-col", default="sequence")
    parser.add_argument("--label-col", default="toxicity")
    parser.add_argument("--id-col", default="id")
    parser.add_argument("--split-col", default="")
    parser.add_argument("--split-value", default="")
    parser.add_argument("--dataset-name", default="toxgin_eval")
    parser.add_argument(
        "--plm-device",
        default="auto",
        help="Device for ESM2 feature extraction. Use cuda, cpu, or auto.",
    )
    parser.add_argument(
        "--gnn-device",
        default="cpu",
        help="Device for the ToxGIN GNN inference. CPU is the safest default.",
    )
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--max-length", type=int, default=60)
    parser.add_argument("--threshold", type=float, default=8.0)
    parser.add_argument("--output-json", default="")
    parser.add_argument("--predictions-csv", default="")
    return parser.parse_args()


def resolve_device(spec: str) -> torch.device:
    if spec == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(spec)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available in the current session.")
    return device


def sanitize_sequence(sequence: str) -> str:
    sequence = str(sequence).strip().upper()
    return "".join(residue if residue in AA1_ALLOWED else "X" for residue in sequence)


def load_samples(args: argparse.Namespace) -> List[Sample]:
    df = pd.read_csv(args.input_csv)
    if args.split_col and args.split_value:
        if args.split_col not in df.columns:
            raise ValueError(f"Split column '{args.split_col}' not found in {args.input_csv}")
        df = df[df[args.split_col].astype(str) == str(args.split_value)].copy()

    for col in [args.sequence_col, args.label_col]:
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' not found in {args.input_csv}")

    samples: List[Sample] = []
    for idx, row in df.reset_index(drop=True).iterrows():
        label = int(row[args.label_col])
        sample_id = f"sample_{idx:05d}_{label}"
        original_id = str(row[args.id_col]) if args.id_col and args.id_col in df.columns else sample_id
        samples.append(
            Sample(
                sample_id=sample_id,
                original_id=original_id,
                sequence=sanitize_sequence(row[args.sequence_col]),
                label=label,
            )
        )

    if not samples:
        raise ValueError("No samples left after filtering.")
    return samples


def load_aaindex1(file_path: Path) -> dict:
    aaindex_df = pd.read_csv(file_path, index_col="Description")
    return {aa: aaindex_df[aa].values for aa in aaindex_df.columns}


def extract_features(sequence: str, aaindex_dict: dict) -> np.ndarray:
    feature_dim = len(next(iter(aaindex_dict.values())))
    features = []
    for aa in sequence:
        if aa in aaindex_dict:
            features.append(aaindex_dict[aa])
        else:
            features.append(np.full((feature_dim,), np.nan, dtype=np.float32))
    return np.asarray(features, dtype=np.float32)


def generate_graph(coords: np.ndarray, threshold: float) -> dgl.DGLGraph:
    diffs = np.expand_dims(coords, axis=1) - np.expand_dims(coords, axis=0)
    distance = np.sqrt(np.sum(np.power(diffs, 2), axis=-1))
    adj = distance < threshold
    u, v = np.nonzero(adj)
    return dgl.graph((torch.from_numpy(u), torch.from_numpy(v)), num_nodes=coords.shape[0])


def aggre(series: pd.Series):
    first = series.values[0]
    return first if isinstance(first, str) else np.mean(series.values)


def load_residue_df(pdb_path: Path) -> pd.DataFrame:
    atom_df = PandasPdb().read_pdb(str(pdb_path)).df["ATOM"]
    residue_df = atom_df.groupby("residue_number", as_index=False).agg(aggre).sort_values("residue_number")
    residue_df["letter"] = residue_df["residue_name"].map(AA3_TO_AA1)
    if residue_df["letter"].isna().any():
        missing = sorted(set(residue_df.loc[residue_df["letter"].isna(), "residue_name"].astype(str)))
        raise ValueError(f"Unsupported residue names in {pdb_path}: {missing}")
    return residue_df


def load_plm(model_name_or_path: str, device: torch.device):
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    config = AutoConfig.from_pretrained(model_name_or_path, output_hidden_states=True)
    config.hidden_dropout = 0.0
    config.hidden_dropout_prob = 0.0
    config.attention_dropout = 0.0
    config.attention_probs_dropout_prob = 0.0
    encoder = AutoModel.from_pretrained(model_name_or_path, config=config).to(device).eval()
    return tokenizer, encoder


def seq_encode(
    sequence: str,
    tokenizer,
    encoder,
    device: torch.device,
    max_length: int,
) -> np.ndarray:
    spaced_seq = " ".join(list(sequence))
    inputs = tokenizer.encode_plus(
        spaced_seq,
        return_tensors=None,
        add_special_tokens=True,
        max_length=max_length,
        padding=True,
        truncation=True,
    )
    tensor_inputs = {
        key: torch.tensor(value, dtype=torch.long, device=device).unsqueeze(0)
        for key, value in inputs.items()
    }
    with torch.no_grad():
        outputs = encoder(
            input_ids=tensor_inputs["input_ids"],
            attention_mask=tensor_inputs["attention_mask"],
        )
    last_hidden = outputs[0][0]
    mask = tensor_inputs["attention_mask"][0].bool()
    encoded = last_hidden[mask][1:-1]
    return encoded.detach().cpu().numpy().astype(np.float32)


def build_graphs_and_features(
    samples: Sequence[Sample],
    pdb_dir: Path,
    data_dir: Path,
    aaindex_path: Path,
    esm_model: str,
    plm_device: torch.device,
    max_length: int,
    threshold: float,
) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    graph_path = data_dir / "dgl_graph.bin"
    aaindex_dict = load_aaindex1(aaindex_path)
    scaler = MinMaxScaler()

    missing_features = [sample for sample in samples if not (data_dir / f"{sample.sample_id}.npz").exists()]
    tokenizer = encoder = None
    if missing_features:
        tokenizer, encoder = load_plm(esm_model, plm_device)

    graphs = []
    for sample in samples:
        pdb_path = pdb_dir / f"{sample.sample_id}.pdb"
        if not pdb_path.exists():
            raise FileNotFoundError(f"Missing PDB for {sample.sample_id}: {pdb_path}")

        residue_df = load_residue_df(pdb_path)
        pdb_sequence = "".join(residue_df["letter"].astype(str).tolist())
        graph = generate_graph(
            residue_df.loc[:, ["x_coord", "y_coord", "z_coord"]].to_numpy(dtype=np.float32),
            threshold=threshold,
        )
        graphs.append(graph)

        feature_path = data_dir / f"{sample.sample_id}.npz"
        if feature_path.exists():
            continue

        encoded_seq = seq_encode(pdb_sequence, tokenizer, encoder, plm_device, max_length=max_length)
        physchem = extract_features(pdb_sequence, aaindex_dict)
        normalized = scaler.fit_transform(physchem.astype(np.float32)).astype(np.float32)
        combined = np.concatenate([encoded_seq, normalized], axis=-1).astype(np.float32)
        np.savez_compressed(feature_path, wildtype_seq=combined, pdb_sequence=pdb_sequence)

    dgl.save_graphs(str(graph_path), graphs)


class ToxGINDataset(torch.utils.data.Dataset):
    def __init__(self, samples: Sequence[Sample], data_dir: Path):
        self.samples = list(samples)
        self.data_dir = data_dir
        self.graphs, _ = dgl.load_graphs(str(data_dir / "dgl_graph.bin"))
        if len(self.graphs) != len(self.samples):
            raise ValueError("Graph count does not match sample count.")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        sample = self.samples[index]
        graph = self.graphs[index].clone()
        with np.load(self.data_dir / f"{sample.sample_id}.npz") as feature_file:
            wildtype_seq = feature_file["wildtype_seq"]
        graph.ndata["wildtype_seq"] = torch.from_numpy(wildtype_seq)
        return graph, sample


class GIN(torch.nn.Module):
    def __init__(self, in_dim: int, layer_num: int = 4):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        self.activations = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()
        for _ in range(layer_num):
            self.convs.append(
                dgl.nn.pytorch.GINConv(
                    torch.nn.Sequential(
                        torch.nn.Linear(in_dim, in_dim, bias=False),
                        torch.nn.BatchNorm1d(in_dim),
                        torch.nn.LeakyReLU(),
                        torch.nn.Linear(in_dim, in_dim, bias=False),
                    ),
                    learn_eps=False,
                )
            )
            self.activations.append(torch.nn.LeakyReLU())
            self.batch_norms.append(torch.nn.BatchNorm1d(in_dim))
        self.out_dim = in_dim * (layer_num + 1)

    def forward(self, graph, features):
        hidden_states = [features]
        h = features
        for conv, batch_norm, activation in zip(self.convs, self.batch_norms, self.activations):
            h = conv(graph, h)
            h = batch_norm(h)
            h = activation(h)
            hidden_states.append(h)
        return torch.cat(hidden_states, dim=-1)


class GNNModel(torch.nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, dropout_rate: float = 0.5):
        super().__init__()
        self.comp = torch.nn.Sequential(
            torch.nn.Linear(in_dim, hidden_dim),
            torch.nn.LeakyReLU(),
        )
        self.gcn = GIN(hidden_dim)
        self.classifier = torch.nn.Sequential(
            torch.nn.Dropout(dropout_rate),
            torch.nn.Linear(self.gcn.out_dim, self.gcn.out_dim),
            torch.nn.LeakyReLU(),
            torch.nn.Dropout(dropout_rate),
            torch.nn.Linear(self.gcn.out_dim, 3000),
            torch.nn.LeakyReLU(),
            torch.nn.Dropout(dropout_rate),
            torch.nn.Linear(3000, 2000),
            torch.nn.LeakyReLU(),
            torch.nn.Dropout(dropout_rate),
            torch.nn.Linear(2000, 1000),
            torch.nn.LeakyReLU(),
            torch.nn.Dropout(dropout_rate),
            torch.nn.Linear(1000, 1),
            torch.nn.Sigmoid(),
        )

    def forward(self, graph, wildtype_seq):
        wildtype_h = self.comp(wildtype_seq)
        gcn_h = self.gcn(graph, wildtype_h)
        with graph.local_scope():
            graph.ndata["h"] = gcn_h
            hg = dgl.readout_nodes(graph, "h", op="sum")
        return self.classifier(hg)


def collate_fn(batch: Sequence[Tuple[dgl.DGLGraph, Sample]]):
    graphs, samples = zip(*batch)
    return dgl.batch(graphs), list(samples)


def infer(model: GNNModel, dataloader: DataLoader, device: torch.device):
    rows = []
    model.eval()
    with torch.no_grad():
        for graph, batch_samples in dataloader:
            graph = graph.to(device)
            wildtype_seq = graph.ndata["wildtype_seq"]
            scores = model(graph, wildtype_seq).squeeze(1).detach().cpu().numpy()
            for sample, score in zip(batch_samples, scores):
                rows.append(
                    {
                        "sample_id": sample.sample_id,
                        "original_id": sample.original_id,
                        "sequence": sample.sequence,
                        "label": sample.label,
                        "predicted_probability": float(score),
                        "predicted_label": int(score >= 0.5),
                    }
                )
    return pd.DataFrame(rows)


def compute_metrics(predictions: pd.DataFrame) -> dict:
    y_true = predictions["label"].to_numpy(dtype=np.int32)
    y_score = predictions["predicted_probability"].to_numpy(dtype=np.float32)
    y_pred = predictions["predicted_label"].to_numpy(dtype=np.int32)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    sensitivity = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0

    return {
        "dataset": predictions.attrs["dataset_name"],
        "num_samples": int(len(predictions)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "f1": float(f1_score(y_true, y_pred)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "auc": float(roc_auc_score(y_true, y_score)),
        "aupr": float(average_precision_score(y_true, y_score)),
        "confusion_matrix": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        },
    }


def main() -> None:
    args = parse_args()
    samples = load_samples(args)

    run_dir = Path(args.run_dir)
    data_dir = run_dir / "toxgin_data"
    pdb_dir = Path(args.pdb_dir)
    aaindex_path = Path(args.aaindex)
    output_json = Path(args.output_json) if args.output_json else run_dir / "metrics.json"
    predictions_csv = Path(args.predictions_csv) if args.predictions_csv else run_dir / "predictions.csv"

    run_dir.mkdir(parents=True, exist_ok=True)
    plm_device = resolve_device(args.plm_device)
    gnn_device = resolve_device(args.gnn_device)

    print(f"dataset={args.dataset_name}")
    print(f"num_samples={len(samples)}")
    print(f"plm_device={plm_device}")
    print(f"gnn_device={gnn_device}")

    build_graphs_and_features(
        samples=samples,
        pdb_dir=pdb_dir,
        data_dir=data_dir,
        aaindex_path=aaindex_path,
        esm_model=args.esm_model,
        plm_device=plm_device,
        max_length=args.max_length,
        threshold=args.threshold,
    )

    dataset = ToxGINDataset(samples, data_dir)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    model = GNNModel(in_dim=3113, hidden_dim=1024).to(gnn_device)
    state_dict = torch.load(args.weights, map_location=gnn_device)
    model.load_state_dict(state_dict)

    predictions = infer(model, dataloader, gnn_device)
    predictions.attrs["dataset_name"] = args.dataset_name
    metrics = compute_metrics(predictions)

    predictions_csv.parent.mkdir(parents=True, exist_ok=True)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(predictions_csv, index=False)
    output_json.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(f"accuracy={metrics['accuracy']:.6f}")
    print(f"sensitivity={metrics['sensitivity']:.6f}")
    print(f"specificity={metrics['specificity']:.6f}")
    print(f"f1={metrics['f1']:.6f}")
    print(f"mcc={metrics['mcc']:.6f}")
    print(f"auc={metrics['auc']:.6f}")
    print(f"aupr={metrics['aupr']:.6f}")
    cm = metrics["confusion_matrix"]
    print(f"confusion_matrix=TN:{cm['tn']} FP:{cm['fp']} FN:{cm['fn']} TP:{cm['tp']}")


if __name__ == "__main__":
    main()
