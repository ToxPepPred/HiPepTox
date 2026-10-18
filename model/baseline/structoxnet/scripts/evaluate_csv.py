#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch_cluster
import torch_scatter
from Bio.PDB import PDBParser
from scipy.spatial import ConvexHull
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    recall_score,
    roc_auc_score,
)
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from transformers import EsmForProteinFolding, T5EncoderModel, T5Tokenizer
from transformers.models.esm import modeling_esmfold

PROJECT_DIR = Path(__file__).resolve().parents[1]
PREPROCESS_DIR = PROJECT_DIR / "Preprocessing"
DEFAULT_MKDSSP = shutil.which("mkdssp") or "mkdssp"
sys.path.insert(0, str(PROJECT_DIR))

from model.StrucToxNet import EGNNModel  # noqa: E402


AA3_TO_INDEX = {
    "ALA": 0,
    "ARG": 1,
    "ASN": 2,
    "ASP": 3,
    "CYS": 4,
    "GLU": 5,
    "GLN": 6,
    "GLY": 7,
    "HIS": 8,
    "ILE": 9,
    "LEU": 10,
    "LYS": 11,
    "MET": 12,
    "PHE": 13,
    "PRO": 14,
    "SER": 15,
    "THR": 16,
    "TRP": 17,
    "TYR": 18,
    "VAL": 19,
    "LIG": 20,
}

AA3_TO_1 = {
    "ALA": "A",
    "ARG": "R",
    "ASN": "N",
    "ASP": "D",
    "CYS": "C",
    "GLN": "Q",
    "GLU": "E",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LEU": "L",
    "LYS": "K",
    "MET": "M",
    "PHE": "F",
    "PRO": "P",
    "SER": "S",
    "THR": "T",
    "TRP": "W",
    "TYR": "Y",
    "VAL": "V",
}

AA3_TO_1_PROTTRANS = {
    "ALA": "A",
    "ARG": "R",
    "ASN": "N",
    "ASP": "D",
    "CYS": "C",
    "GLN": "Q",
    "GLU": "E",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LEU": "L",
    "LYS": "K",
    "MET": "M",
    "PHE": "F",
    "PRO": "P",
    "SER": "S",
    "THR": "T",
    "TRP": "W",
    "TYR": "Y",
    "VAL": "V",
}

AA1_ALLOWED = set("ACDEFGHIKLMNPQRSTUVWYX")
SS_MAPPING = {"H": 0, "G": 1, "I": 2, "E": 3, "B": 4, "T": 5, "S": 6}
SA_THRESHOLDS = [30, 60, 90, 120, 150, 180, 210]
DISTANCE_THRESHOLD = 8.0


@dataclass
class Sample:
    sample_id: str
    original_id: str
    sequence: str
    label: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run StrucToxNet evaluation directly on a labeled CSV."
    )
    parser.add_argument("--input-csv", required=True, help="Input CSV file.")
    parser.add_argument(
        "--sequence-col", default="sequence", help="Column containing peptide sequences."
    )
    parser.add_argument(
        "--label-col", default="toxicity", help="Column containing binary labels."
    )
    parser.add_argument(
        "--id-col", default="id", help="Column used to preserve the original row id."
    )
    parser.add_argument(
        "--split-col",
        default=None,
        help="Optional split column. If provided, only rows matching --split-value are used.",
    )
    parser.add_argument(
        "--split-value",
        default=None,
        help="Value to keep from the split column, e.g. validation.",
    )
    parser.add_argument(
        "--dataset-name",
        default=None,
        help="Optional dataset name used in output files.",
    )
    parser.add_argument(
        "--run-dir",
        required=True,
        help="Working directory for generated FASTA/PDB/DSSP/cache files.",
    )
    parser.add_argument(
        "--weights",
        default=str(PROJECT_DIR / "model" / "final_model.pkl"),
        help="Path to StrucToxNet trained weights.",
    )
    parser.add_argument(
        "--prot-t5-model",
        default=str(PROJECT_DIR / "models" / "prot_t5_xl_uniref50"),
        help="Path to local ProtT5 model directory.",
    )
    parser.add_argument(
        "--esmfold-model",
        default=str(PROJECT_DIR / "models" / "esmfold_v1"),
        help="Path to local ESMFold model directory.",
    )
    parser.add_argument(
        "--mkdssp",
        default=DEFAULT_MKDSSP,
        help="Path to mkdssp executable.",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help='Device for inference. Use "cuda" or "cpu".',
    )
    parser.add_argument(
        "--eval-batch-size", type=int, default=128, help="Evaluation DataLoader batch size."
    )
    parser.add_argument(
        "--esmfold-batch-size", type=int, default=4, help="Batch size for ESMFold."
    )
    parser.add_argument(
        "--esmfold-chunk-size",
        type=int,
        default=64,
        help="Chunk size passed to ESMFold trunk when supported.",
    )
    parser.add_argument(
        "--prot-t5-batch-size", type=int, default=8, help="Batch size for ProtT5."
    )
    parser.add_argument(
        "--output-json", required=True, help="Path to save evaluation metrics JSON."
    )
    parser.add_argument(
        "--predictions-csv",
        default=None,
        help="Optional path to save row-level predictions.",
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
    return "".join(residue if residue in AA1_ALLOWED else "X" for residue in sequence)


def load_samples(args: argparse.Namespace) -> List[Sample]:
    df = pd.read_csv(args.input_csv)
    if args.split_col and args.split_value:
        if args.split_col not in df.columns:
            raise ValueError(
                f"Split column '{args.split_col}' not found in {args.input_csv}. "
                f"Available columns: {list(df.columns)}"
            )
        df = df[df[args.split_col].astype(str) == str(args.split_value)].copy()

    for col in [args.sequence_col, args.label_col]:
        if col not in df.columns:
            raise ValueError(
                f"Required column '{col}' not found in {args.input_csv}. "
                f"Available columns: {list(df.columns)}"
            )

    samples = []
    for idx, row in df.reset_index(drop=True).iterrows():
        label = int(row[args.label_col])
        sample_id = f"sample_{idx:05d}_{label}"
        original_id = str(row[args.id_col]) if args.id_col in df.columns else sample_id
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


def write_fasta(samples: Iterable[Sample], fasta_path: Path) -> None:
    with fasta_path.open("w", encoding="utf-8") as handle:
        for sample in samples:
            handle.write(f">{sample.sample_id}\n{sample.sequence}\n")


def batched(items: List[Sample], batch_size: int) -> Iterable[List[Sample]]:
    for idx in range(0, len(items), batch_size):
        yield items[idx:idx + batch_size]


def load_esmfold(model_path: Path, device: torch.device, chunk_size: int) -> EsmForProteinFolding:
    # The downloaded HF ESMFold weights can emit invalid pTM tensors on this host.
    # PDB generation only needs structural coordinates and pLDDT, so we safely bypass
    # the pTM-only branch that would otherwise abort preprocessing.
    def safe_compute_tm(logits: torch.Tensor, max_bin: int = 31, no_bins: int = 64) -> torch.Tensor:
        return torch.zeros(logits.shape[0], device=logits.device, dtype=logits.dtype)

    modeling_esmfold.compute_tm = safe_compute_tm
    modeling_esmfold.compute_predicted_aligned_error = lambda *args, **kwargs: {}

    model = EsmForProteinFolding.from_pretrained(
        str(model_path),
        low_cpu_mem_usage=True,
    )
    model = model.eval()
    if hasattr(model, "trunk") and hasattr(model.trunk, "set_chunk_size"):
        model.trunk.set_chunk_size(chunk_size)
    if device.type == "cuda":
        model = model.cuda()
        if hasattr(model, "esm"):
            model.esm = model.esm.half()
    return model


def generate_pdbs(
    samples: List[Sample],
    pdb_dir: Path,
    model_path: Path,
    device: torch.device,
    batch_size: int,
    chunk_size: int,
) -> None:
    missing = [sample for sample in samples if not (pdb_dir / f"{sample.sample_id}.pdb").exists()]
    if not missing:
        print("esmfold_missing=0")
        return

    print(f"esmfold_missing={len(missing)}")
    model = load_esmfold(model_path, device, chunk_size)

    for group in batched(missing, batch_size):
        sequences = [sample.sequence for sample in group]
        with torch.no_grad():
            pdbs = model.infer_pdbs(sequences)
        for sample, pdb_text in zip(group, pdbs):
            (pdb_dir / f"{sample.sample_id}.pdb").write_text(pdb_text, encoding="utf-8")


def write_processed_dssp(raw_dssp_path: Path, processed_dssp_path: Path) -> None:
    residue_flag = "#"
    lines = raw_dssp_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    processed = ["res\taa\tss\tsa\ttco\tkpa\talp\tphi\tpsi\tx\ty\tz"]
    passed_header = False
    for line in lines:
        if not passed_header:
            if line[2:2 + len(residue_flag)] == residue_flag:
                passed_header = True
            continue
        if not line:
            continue
        ss_seq = line[16:17] or "C"
        aa_seq = line[13]
        sa_seq = line[35:38]
        tco_seq = line[85:91]
        kpa_seq = line[91:97]
        alp_seq = line[97:103]
        phi_seq = line[103:109]
        psi_seq = line[109:115]
        x_seq = line[115:122]
        y_seq = line[122:129]
        z_seq = line[129:136]
        if not ss_seq.strip():
            ss_seq = "C"
        if line[5:10].strip():
            res_num = int(line[5:10])
            processed.append(
                "\t".join(
                    [
                        str(res_num),
                        aa_seq,
                        ss_seq,
                        sa_seq.strip(),
                        tco_seq.strip(),
                        kpa_seq.strip(),
                        alp_seq.strip(),
                        phi_seq.strip(),
                        psi_seq.strip(),
                        x_seq.strip(),
                        y_seq.strip(),
                        z_seq.strip(),
                    ]
                )
            )
    processed_dssp_path.write_text("\n".join(processed) + "\n", encoding="utf-8")


def get_distance(x1: float, x2: float, x3: float, y1: float, y2: float, y3: float) -> float:
    return math.sqrt((x1 - y1) ** 2 + (x2 - y2) ** 2 + (x3 - y3) ** 2)


def build_pdb2rr(pdb_path: Path, pdb2rr_path: Path) -> None:
    lines = pdb_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    positions = [line for line in lines if line.startswith("ATOM")]
    ca_info = {}
    cb_info = {}
    for line in positions:
        atom_name = line[12:16].strip()
        res_no = int(line[22:26].strip())
        xyz = [
            float(line[30:38].strip()),
            float(line[38:46].strip()),
            float(line[46:54].strip()),
        ]
        if atom_name == "CA":
            ca_info[res_no] = xyz
        elif atom_name == "CB":
            cb_info[res_no] = xyz

    records = []
    for res_no in ca_info:
        for res_no2 in ca_info:
            if res_no2 < res_no + 1:
                continue
            if res_no not in cb_info and res_no2 not in cb_info:
                cb_cb_distance = get_distance(*ca_info[res_no], *ca_info[res_no2])
            elif res_no in cb_info and res_no2 not in cb_info:
                cb_cb_distance = get_distance(*cb_info[res_no], *ca_info[res_no2])
            elif res_no not in cb_info and res_no2 in cb_info:
                cb_cb_distance = get_distance(*ca_info[res_no], *cb_info[res_no2])
            else:
                cb_cb_distance = get_distance(*cb_info[res_no], *cb_info[res_no2])
            if cb_cb_distance <= DISTANCE_THRESHOLD:
                records.append(f"{res_no} {res_no2} 0 8 {cb_cb_distance:.1f}")
    pdb2rr_path.write_text("\n".join(records) + ("\n" if records else ""), encoding="utf-8")


def run_mkdssp(mkdssp_path: str, pdb_path: Path, raw_dssp_path: Path) -> None:
    legacy_cmd = [mkdssp_path, "-i", str(pdb_path), "-o", str(raw_dssp_path)]
    result = subprocess.run(
        legacy_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if result.returncode == 0:
        return

    # mkdssp >=4 uses positional input/output arguments instead of -i/-o.
    if "unknown option" in (result.stderr or "") and "option 'i'" in (result.stderr or ""):
        subprocess.run(
            [mkdssp_path, str(pdb_path), str(raw_dssp_path)],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        return

    raise subprocess.CalledProcessError(
        result.returncode,
        legacy_cmd,
        output=result.stdout,
        stderr=result.stderr,
    )


def build_dssp_and_contacts(
    samples: List[Sample],
    args: argparse.Namespace,
    pdb_dir: Path,
    raw_dssp_dir: Path,
    dssp_dir: Path,
    pdb2rr_dir: Path,
) -> None:
    for sample in samples:
        pdb_path = pdb_dir / f"{sample.sample_id}.pdb"
        raw_dssp_path = raw_dssp_dir / f"{sample.sample_id}.dssp"
        processed_dssp_path = dssp_dir / f"{sample.sample_id}.dssp"
        pdb2rr_path = pdb2rr_dir / f"{sample.sample_id}.pdb2rr"

        if not processed_dssp_path.exists():
            run_mkdssp(args.mkdssp, pdb_path, raw_dssp_path)
            write_processed_dssp(raw_dssp_path, processed_dssp_path)

        if not pdb2rr_path.exists():
            build_pdb2rr(pdb_path, pdb2rr_path)


def load_pdb_atom_lines(pdb_path: Path) -> List[str]:
    atoms = []
    with pdb_path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if line.startswith("ATOM"):
                atoms.append(line.rstrip("\n"))
    return atoms


def get_tetrahedral_geom(pos: List[str]) -> List[List[float]]:
    last_res_no = int(pos[-1][22:26].strip())
    thg = [[[0, 0, 0], [0, 0, 0], [0, 0, 0]] for _ in range(last_res_no)]
    thg_val = [[0, 0, 0] for _ in range(last_res_no)]

    for line in pos:
        res_no = int(line[22:26].strip())
        atom_type = line[13:15].strip()
        xyz = [
            float(line[30:38]),
            float(line[38:46]),
            float(line[46:54]),
        ]
        if atom_type == "CA":
            thg[res_no - 1][0] = xyz
        elif atom_type == "C":
            thg[res_no - 1][1] = xyz
        elif atom_type == "N":
            thg[res_no - 1][2] = xyz

    for idx in range(len(thg_val)):
        n_atom = np.array(thg[idx][2])
        ca_atom = np.array(thg[idx][0])
        c_atom = np.array(thg[idx][1])
        n_vec = n_atom - ca_atom
        c_vec = c_atom - ca_atom
        cross = np.cross(n_vec, c_vec)
        cross_norm = cross[0] ** 2 + cross[1] ** 2 + cross[2] ** 2
        sum_vec = n_vec + c_vec
        sum_norm = sum_vec[0] ** 2 + sum_vec[1] ** 2 + sum_vec[2] ** 2
        if cross_norm == 0 or sum_norm == 0:
            thg_val[idx] = [0, 0, 0]
            continue
        t1 = cross / (cross_norm * math.sqrt(3))
        t2 = math.sqrt(2 / 3) * sum_vec / sum_norm
        value = t1 - t2
        thg_val[idx] = [0, 0, 0] if np.isnan(value).any() else value.tolist()
    return thg_val


def get_residue_area_volume(pos: List[str]) -> tuple[List[float], List[float]]:
    last_res_no = int(pos[-1][22:26].strip())
    residue_coords: List[List[List[float]]] = [[] for _ in range(last_res_no)]
    area = [0.0 for _ in range(last_res_no)]
    volume = [0.0 for _ in range(last_res_no)]
    for line in pos:
        res_no = int(line[22:26].strip())
        residue_coords[res_no - 1].append(
            [float(line[30:38]), float(line[38:46]), float(line[46:54])]
        )

    for idx, residue in enumerate(residue_coords):
        if len(residue) < 4:
            continue
        hull = ConvexHull(residue)
        volume[idx] = float(hull.volume)
        area[idx] = float(hull.area)
    return area, volume


def feature_1(pdb_path: Path, sequence: str) -> List[List[float]]:
    pos = load_pdb_atom_lines(pdb_path)
    dhg_val = get_tetrahedral_geom(pos)
    area, _ = get_residue_area_volume(pos)
    if len(dhg_val) != len(sequence) or len(area) != len(sequence):
        raise ValueError(f"feature_1 length mismatch for {pdb_path}")
    return [dhg_val[idx] + [area[idx]] for idx in range(len(sequence))]


def feature_2(pdb2rr_path: Path, sequence: str) -> List[List[float]]:
    count_list = [0.0 for _ in range(len(sequence))]
    if pdb2rr_path.exists():
        with pdb2rr_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                parts = line.split()
                if len(parts) < 2:
                    continue
                res1 = int(parts[0])
                res2 = int(parts[1])
                if res1 <= len(sequence):
                    count_list[res1 - 1] += 1
                if res2 <= len(sequence):
                    count_list[res2 - 1] += 1
    for idx, value in enumerate(count_list):
        count_list[idx] = 1.0 if value > 25 else value / 25.0
    return [[value] for value in count_list]


def ss_one_hot3(ss: str) -> List[int]:
    if ss in {"H", "G", "I"}:
        return [1, 0, 0]
    if ss in {"E", "B"}:
        return [0, 1, 0]
    return [0, 0, 1]


def sa_one_hot2(sa: str) -> List[int]:
    return [1, 0] if float(sa) > 50 else [0, 1]


def feature_3(dssp_path: Path) -> List[List[float]]:
    lines = dssp_path.read_text(encoding="utf-8").splitlines()
    length = int(lines[-1].split()[0])
    dssp_list = [[0.0 for _ in range(5)] for _ in range(length)]
    for line in lines[1:]:
        parts = line.strip().split()
        residue = int(parts[0]) - 1
        dssp_list[residue] = ss_one_hot3(parts[2]) + sa_one_hot2(parts[3])
    return [values + [1.0 / (idx + 1)] for idx, values in enumerate(dssp_list)]


def ss_one_hot8(ss: str) -> List[float]:
    one_hot = np.zeros(8, dtype=np.float32)
    one_hot[SS_MAPPING.get(ss, 7)] = 1.0
    return one_hot.tolist()


def sa_one_hot8(sa: str) -> List[float]:
    one_hot = np.zeros(8, dtype=np.float32)
    one_hot[np.searchsorted(SA_THRESHOLDS, float(sa))] = 1.0
    return one_hot.tolist()


def centroid(positions: List[List[float]]) -> tuple[float, float, float]:
    arr = np.asarray(positions, dtype=np.float32)
    return float(arr[:, 0].mean()), float(arr[:, 1].mean()), float(arr[:, 2].mean())


def dist(x1: float, y1: float, z1: float, x2: float, y2: float, z2: float) -> float:
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2 + (z1 - z2) ** 2)


def feature_4(dssp_path: Path) -> List[List[float]]:
    lines = dssp_path.read_text(encoding="utf-8").splitlines()
    length = int(lines[-1].split()[0])
    dssp_list = [[0.0 for _ in range(22)] for _ in range(length)]
    positions = []
    for line in lines[1:]:
        parts = line.strip().split()
        positions.append([float(parts[9]), float(parts[10]), float(parts[11])])
    cx, cy, cz = centroid(positions)
    for line in lines[1:]:
        parts = line.strip().split()
        residue = int(parts[0]) - 1
        x, y, z = float(parts[9]), float(parts[10]), float(parts[11])
        dssp_list[residue][0] = 1.0 / dist(x, y, z, cx, cy, cz)
        dssp_list[residue][1:9] = ss_one_hot8(parts[2])
        dssp_list[residue][9:17] = sa_one_hot8(parts[3])
        dssp_list[residue][17] = float(parts[4])
        dssp_list[residue][18] = float(parts[5]) / 360.0
        dssp_list[residue][19] = float(parts[6]) / 360.0
        dssp_list[residue][20] = float(parts[7]) / 360.0
        dssp_list[residue][21] = float(parts[8]) / 360.0
    return dssp_list


def feature_5(dssp_path: Path) -> List[List[float]]:
    lines = dssp_path.read_text(encoding="utf-8").splitlines()
    length = int(lines[-1].split()[0])
    features = [[0.0 for _ in range(6)] for _ in range(length)]
    for line in lines[1:]:
        parts = line.strip().split()
        residue = int(parts[0]) - 1
        alpha = float(parts[6])
        phi = float(parts[7])
        psi = float(parts[8])
        features[residue] = [
            math.sin(math.radians(alpha)),
            math.cos(math.radians(alpha)),
            math.sin(math.radians(phi)),
            math.cos(math.radians(phi)),
            math.sin(math.radians(psi)),
            math.cos(math.radians(psi)),
        ]
    return features


def feature_6(dssp_path: Path) -> List[List[float]]:
    lines = dssp_path.read_text(encoding="utf-8").splitlines()
    length = int(lines[-1].split()[0])
    features = [[0.0 for _ in range(6)] for _ in range(length)]
    for line_idx in range(2, len(lines) - 1):
        line = lines[line_idx].split()
        line_prev = lines[line_idx - 1].split()
        line_next = lines[line_idx + 1].split()
        residue = int(line[0]) - 1
        ca_i = np.array([float(line[9]), float(line[10]), float(line[11])], dtype=np.float32)
        ca_prev = np.array([float(line_prev[9]), float(line_prev[10]), float(line_prev[11])], dtype=np.float32)
        ca_next = np.array([float(line_next[9]), float(line_next[10]), float(line_next[11])], dtype=np.float32)
        forward = ca_next - ca_i
        backward = ca_prev - ca_i
        forward_norm = np.linalg.norm(forward)
        backward_norm = np.linalg.norm(backward)
        if forward_norm == 0 or backward_norm == 0:
            continue
        features[residue][:3] = (forward / forward_norm).tolist()
        features[residue][3:] = (backward / backward_norm).tolist()
    return features


def compute_structure_features(sample: Sample, pdb_dir: Path, dssp_dir: Path, pdb2rr_dir: Path) -> np.ndarray:
    pdb_path = pdb_dir / f"{sample.sample_id}.pdb"
    dssp_path = dssp_dir / f"{sample.sample_id}.dssp"
    pdb2rr_path = pdb2rr_dir / f"{sample.sample_id}.pdb2rr"
    features_1 = feature_1(pdb_path, sample.sequence)
    features_2 = feature_2(pdb2rr_path, sample.sequence)
    features_3 = feature_3(dssp_path)
    features_4 = feature_4(dssp_path)
    features_5 = feature_5(dssp_path)
    features_6 = feature_6(dssp_path)
    lengths = [len(features_1), len(features_2), len(features_3), len(features_4), len(features_5), len(features_6)]
    if len(set(lengths)) != 1:
        raise ValueError(f"Feature lengths mismatch for {sample.sample_id}: {lengths}")
    combined = []
    for idx in range(lengths[0]):
        combined.append(
            features_1[idx]
            + features_2[idx]
            + features_3[idx]
            + features_4[idx]
            + features_5[idx]
            + features_6[idx]
        )
    return np.asarray(combined, dtype=np.float32)


def load_prot_t5(model_path: Path, device: torch.device) -> tuple[T5Tokenizer, T5EncoderModel]:
    tokenizer = T5Tokenizer.from_pretrained(str(model_path), do_lower_case=False)
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    model = T5EncoderModel.from_pretrained(
        str(model_path),
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    )
    model = model.eval().to(device)
    return tokenizer, model


def build_plm_cache(
    samples: List[Sample],
    plm_dir: Path,
    model_path: Path,
    x_min_path: Path,
    x_max_path: Path,
    device: torch.device,
    batch_size: int,
) -> None:
    missing = [sample for sample in samples if not (plm_dir / f"{sample.sample_id}.npy").exists()]
    if not missing:
        print("prot_t5_missing=0")
        return

    print(f"prot_t5_missing={len(missing)}")
    tokenizer, model = load_prot_t5(model_path, device)
    x_max = np.load(x_max_path)
    x_min = np.load(x_min_path)
    x_range = x_max - x_min
    x_range[x_range == 0] = 1

    for group in batched(missing, batch_size):
        sequences = [" ".join(sample.sequence) for sample in group]
        tokenized = tokenizer(
            sequences,
            add_special_tokens=True,
            padding=True,
            return_tensors="pt",
        )
        input_ids = tokenized["input_ids"].to(device)
        attention_mask = tokenized["attention_mask"].to(device)
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        embeddings = outputs.last_hidden_state.detach().cpu().numpy()
        masks = attention_mask.detach().cpu().numpy()
        for idx, sample in enumerate(group):
            seq_len = int(masks[idx].sum()) - 1
            seq_emb = embeddings[idx, :seq_len, :]
            seq_emb = (seq_emb - x_min) / x_range
            np.save(plm_dir / f"{sample.sample_id}.npy", seq_emb.astype(np.float32))


def extract_ca_coordinates(pdb_path: Path) -> pd.DataFrame:
    parser = PDBParser(PERMISSIVE=1)
    structure = parser.get_structure("structure", str(pdb_path))
    ca_coords = []
    for model in structure:
        for chain in model:
            for residue in chain:
                if "CA" in residue:
                    ca_atom = residue["CA"]
                    resname = residue.get_resname()
                    ca_coords.append([resname, ca_atom.coord[0], ca_atom.coord[1], ca_atom.coord[2]])
    return pd.DataFrame(ca_coords, columns=["resname", "x", "y", "z"])


def _normalize(tensor: torch.Tensor, dim: int = -1) -> torch.Tensor:
    return torch.nan_to_num(torch.div(tensor, torch.norm(tensor, dim=dim, keepdim=True)))


def _rbf(D: torch.Tensor, D_min: float = 0.0, D_max: float = 20.0, D_count: int = 16, device: str = "cpu") -> torch.Tensor:
    D_mu = torch.linspace(D_min, D_max, D_count, device=device).view(1, -1)
    D_sigma = (D_max - D_min) / D_count
    D_expand = torch.unsqueeze(D, -1)
    return torch.exp(-((D_expand - D_mu) / D_sigma) ** 2)


def _edge_features(coords: torch.Tensor, edge_index: torch.Tensor, D_max: float = 4.5, num_rbf: int = 16, device: str = "cpu") -> tuple[torch.Tensor, torch.Tensor]:
    edge_vectors = coords[edge_index[0]] - coords[edge_index[1]]
    rbf = _rbf(edge_vectors.norm(dim=-1), D_max=D_max, D_count=num_rbf, device=device)
    edge_s = rbf
    edge_v = _normalize(edge_vectors).unsqueeze(-2)
    edge_s, edge_v = map(torch.nan_to_num, (edge_s, edge_v))
    return edge_s, edge_v


def cal_geo_feature(df: pd.DataFrame, edge_cutoff: float = 8.0, num_rbf: int = 16) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    coords = torch.as_tensor(df[["x", "y", "z"]].to_numpy(), dtype=torch.float32)
    nodes = torch.as_tensor([AA3_TO_INDEX.get(resname, 21) for resname in df.resname], dtype=torch.long)
    edge_index = torch_cluster.radius_graph(coords, edge_cutoff)
    edge_s, edge_v = _edge_features(coords, edge_index, D_max=edge_cutoff, num_rbf=num_rbf)
    return coords, nodes, edge_index, edge_s, edge_v


def build_dataset(samples: List[Sample], pdb_dir: Path, dssp_dir: Path, pdb2rr_dir: Path, plm_dir: Path) -> List[Data]:
    dataset = []
    for sample_idx, sample in enumerate(samples):
        pdb_path = pdb_dir / f"{sample.sample_id}.pdb"
        structure_features = compute_structure_features(sample, pdb_dir, dssp_dir, pdb2rr_dir)
        plm_features = np.load(plm_dir / f"{sample.sample_id}.npy")
        if len(structure_features) != len(plm_features):
            raise ValueError(
                f"Feature length mismatch for {sample.sample_id}: structure={len(structure_features)} plm={len(plm_features)}"
            )
        ca_df = extract_ca_coordinates(pdb_path)
        pdb_sequence = "".join(AA3_TO_1_PROTTRANS.get(resname, "X") for resname in ca_df["resname"])
        if pdb_sequence != sample.sequence:
            raise ValueError(
                f"PDB sequence mismatch for {sample.sample_id}: expected {sample.sequence}, got {pdb_sequence}"
            )
        coords, nodes, edge_index, edge_s, edge_v = cal_geo_feature(ca_df)
        node_features = torch.from_numpy(np.concatenate([structure_features, plm_features], axis=1)).float()
        dataset.append(
            Data(
                x=coords,
                nodes=nodes,
                edge_index=edge_index,
                edge_s=edge_s,
                edge_v=edge_v,
                plm=node_features,
                label=float(sample.label),
                sample_idx=sample_idx,
            )
        )
    return dataset


def evaluate_model(dataset: List[Data], weights_path: Path, device: torch.device, batch_size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    config = argparse.Namespace(
        in_edge_nf=16,
        num_layer=3,
        dim_embedding=256,
        device=str(device),
        dropout=0.5,
    )
    model = EGNNModel(config).to(device)
    state_dict = torch.load(weights_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=0)
    labels_all = []
    predictions_all = []
    probabilities_all = []

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            labels = batch.label.to(device)
            logits = model.get_logits(batch)
            probabilities = F.softmax(logits, dim=1)[:, 1]
            predictions = torch.argmax(logits, dim=1)
            labels_all.append(labels.cpu().numpy())
            predictions_all.append(predictions.cpu().numpy())
            probabilities_all.append(probabilities.cpu().numpy())

    return (
        np.concatenate(labels_all).astype(int),
        np.concatenate(predictions_all).astype(int),
        np.concatenate(probabilities_all).astype(np.float64),
    )


def safe_specificity(labels: np.ndarray, predictions: np.ndarray) -> tuple[float, int, int, int, int]:
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    specificity = float(tn) / float(tn + fp) if (tn + fp) else 0.0
    return specificity, int(tn), int(fp), int(fn), int(tp)


def metric_auc(labels: np.ndarray, probabilities: np.ndarray) -> float:
    return float(roc_auc_score(labels, probabilities))


def metric_aupr(labels: np.ndarray, probabilities: np.ndarray) -> float:
    return float(average_precision_score(labels, probabilities))


def save_outputs(
    samples: List[Sample],
    labels: np.ndarray,
    predictions: np.ndarray,
    probabilities: np.ndarray,
    args: argparse.Namespace,
) -> dict:
    specificity, tn, fp, fn, tp = safe_specificity(labels, predictions)
    metrics = {
        "dataset": args.dataset_name or Path(args.input_csv).stem,
        "num_samples": int(len(labels)),
        "accuracy": float(accuracy_score(labels, predictions)),
        "sensitivity": float(recall_score(labels, predictions, zero_division=0)),
        "specificity": specificity,
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "mcc": float(matthews_corrcoef(labels, predictions)),
        "auc": metric_auc(labels, probabilities),
        "aupr": metric_aupr(labels, probabilities),
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }

    output_json = Path(args.output_json)
    ensure_dir(output_json.parent)
    output_json.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    if args.predictions_csv:
        prediction_rows = []
        for sample, label, pred, prob in zip(samples, labels, predictions, probabilities):
            prediction_rows.append(
                {
                    "sample_id": sample.sample_id,
                    "original_id": sample.original_id,
                    "sequence": sample.sequence,
                    "label": int(label),
                    "prediction": int(pred),
                    "prob_toxic": float(prob),
                }
            )
        predictions_df = pd.DataFrame(prediction_rows)
        predictions_path = Path(args.predictions_csv)
        ensure_dir(predictions_path.parent)
        predictions_df.to_csv(predictions_path, index=False)

    return metrics


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    run_dir = ensure_dir(Path(args.run_dir).resolve())
    fasta_path = run_dir / "input.fasta"
    pdb_dir = ensure_dir(run_dir / "pdb")
    raw_dssp_dir = ensure_dir(run_dir / "dssp_raw")
    dssp_dir = ensure_dir(run_dir / "dssp")
    pdb2rr_dir = ensure_dir(run_dir / "pdb2rr")
    plm_dir = ensure_dir(run_dir / "plm")

    samples = load_samples(args)
    write_fasta(samples, fasta_path)

    print(f"dataset_rows={len(samples)}")
    print(f"device={device}")

    generate_pdbs(
        samples=samples,
        pdb_dir=pdb_dir,
        model_path=Path(args.esmfold_model),
        device=device,
        batch_size=args.esmfold_batch_size,
        chunk_size=args.esmfold_chunk_size,
    )
    build_dssp_and_contacts(samples, args, pdb_dir, raw_dssp_dir, dssp_dir, pdb2rr_dir)
    build_plm_cache(
        samples=samples,
        plm_dir=plm_dir,
        model_path=Path(args.prot_t5_model),
        x_min_path=PREPROCESS_DIR / "inputs" / "x_min.npy",
        x_max_path=PREPROCESS_DIR / "inputs" / "x_max.npy",
        device=device,
        batch_size=args.prot_t5_batch_size,
    )
    dataset = build_dataset(samples, pdb_dir, dssp_dir, pdb2rr_dir, plm_dir)
    labels, predictions, probabilities = evaluate_model(
        dataset=dataset,
        weights_path=Path(args.weights),
        device=device,
        batch_size=args.eval_batch_size,
    )
    metrics = save_outputs(samples, labels, predictions, probabilities, args)

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
