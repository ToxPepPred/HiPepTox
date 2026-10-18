#!/usr/bin/env python
import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
import torch
from scipy.spatial import ConvexHull


SS_MAPPING = {"H": 0, "G": 1, "I": 2, "E": 3, "B": 4, "T": 5, "S": 6}
SA_THRESHOLDS = [30, 60, 90, 120, 150, 180, 210]


def parse_args():
    parser = argparse.ArgumentParser(description="Build StrucToxNet-style residue structure features.")
    parser.add_argument("--input-csv", required=True, help="CSV with dataset ids and StrucToxNet cache ids.")
    parser.add_argument("--cache-dir", required=True, help="StrucToxNet cache directory containing pdb/dssp/pdb2rr.")
    parser.add_argument("--output-file", required=True, help="Output .pt or .json file.")
    parser.add_argument("--id-col", default="id", help="Column used as the dataset sample id.")
    parser.add_argument(
        "--cache-id-col",
        default="structoxnet_sample_id",
        help="Column containing the StrucToxNet cache sample id.",
    )
    parser.add_argument("--sequence-col", default="sequence", help="Sequence column used for length validation.")
    parser.add_argument("--allow-missing", action="store_true", help="Skip rows whose cache files are missing.")
    return parser.parse_args()


def load_pdb_atom_lines(pdb_path):
    atoms = []
    with open(pdb_path, "r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if line.startswith("ATOM"):
                atoms.append(line.rstrip("\n"))
    return atoms


def get_tetrahedral_geom(pos):
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
            continue
        t1 = cross / (cross_norm * math.sqrt(3))
        t2 = math.sqrt(2 / 3) * sum_vec / sum_norm
        value = t1 - t2
        if not np.isnan(value).any():
            thg_val[idx] = value.tolist()
    return thg_val


def get_residue_area_volume(pos):
    last_res_no = int(pos[-1][22:26].strip())
    residue_coords = [[] for _ in range(last_res_no)]
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


def feature_1(pdb_path, sequence):
    pos = load_pdb_atom_lines(pdb_path)
    dhg_val = get_tetrahedral_geom(pos)
    area, _ = get_residue_area_volume(pos)
    if len(dhg_val) != len(sequence) or len(area) != len(sequence):
        raise ValueError("feature_1 length mismatch for %s" % pdb_path)
    return [dhg_val[idx] + [area[idx]] for idx in range(len(sequence))]


def feature_2(pdb2rr_path, sequence):
    count_list = [0.0 for _ in range(len(sequence))]
    if pdb2rr_path.exists():
        with open(pdb2rr_path, "r", encoding="utf-8") as handle:
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


def ss_one_hot3(ss):
    if ss in {"H", "G", "I"}:
        return [1, 0, 0]
    if ss in {"E", "B"}:
        return [0, 1, 0]
    return [0, 0, 1]


def sa_one_hot2(sa):
    return [1, 0] if float(sa) > 50 else [0, 1]


def feature_3(dssp_path):
    lines = dssp_path.read_text(encoding="utf-8").splitlines()
    length = int(lines[-1].split()[0])
    dssp_list = [[0.0 for _ in range(5)] for _ in range(length)]
    for line in lines[1:]:
        parts = line.strip().split()
        residue = int(parts[0]) - 1
        dssp_list[residue] = ss_one_hot3(parts[2]) + sa_one_hot2(parts[3])
    return [values + [1.0 / (idx + 1)] for idx, values in enumerate(dssp_list)]


def ss_one_hot8(ss):
    one_hot = np.zeros(8, dtype=np.float32)
    one_hot[SS_MAPPING.get(ss, 7)] = 1.0
    return one_hot.tolist()


def sa_one_hot8(sa):
    one_hot = np.zeros(8, dtype=np.float32)
    one_hot[np.searchsorted(SA_THRESHOLDS, float(sa))] = 1.0
    return one_hot.tolist()


def centroid(positions):
    arr = np.asarray(positions, dtype=np.float32)
    return float(arr[:, 0].mean()), float(arr[:, 1].mean()), float(arr[:, 2].mean())


def dist(x1, y1, z1, x2, y2, z2):
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2 + (z1 - z2) ** 2)


def feature_4(dssp_path):
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


def feature_5(dssp_path):
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


def feature_6(dssp_path):
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


def compute_structure_features(pdb_path, dssp_path, pdb2rr_path, sequence):
    features_1 = feature_1(pdb_path, sequence)
    features_2 = feature_2(pdb2rr_path, sequence)
    features_3 = feature_3(dssp_path)
    features_4 = feature_4(dssp_path)
    features_5 = feature_5(dssp_path)
    features_6 = feature_6(dssp_path)
    lengths = [len(features_1), len(features_2), len(features_3), len(features_4), len(features_5), len(features_6)]
    if len(set(lengths)) != 1:
        raise ValueError("Feature lengths mismatch: %s" % lengths)
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


def main():
    args = parse_args()
    cache_dir = Path(args.cache_dir)
    pdb_dir = cache_dir / "pdb"
    dssp_dir = cache_dir / "dssp"
    pdb2rr_dir = cache_dir / "pdb2rr"

    by_id = {}
    feature_dim = None
    missing = []
    seen_ids = set()

    with open(args.input_csv, "r", encoding="utf-8") as fin:
        reader = csv.DictReader(fin)
        required_columns = {args.id_col, args.cache_id_col, args.sequence_col}
        if not required_columns.issubset(reader.fieldnames or []):
            raise ValueError("Mapping CSV is missing columns: %s" % sorted(required_columns - set(reader.fieldnames or [])))
        for line_number, row in enumerate(reader, start=2):
            sample_id = (row.get(args.id_col) or "").strip()
            cache_id = (row.get(args.cache_id_col) or "").strip()
            sequence = (row.get(args.sequence_col) or "").strip().upper()
            if not sample_id or not cache_id or not sequence:
                raise ValueError("Empty ID, cache ID or sequence at CSV line %d" % line_number)
            if sample_id in seen_ids:
                raise ValueError("Duplicate mapping ID `%s` at CSV line %d" % (sample_id, line_number))
            seen_ids.add(sample_id)

            pdb_path = pdb_dir / ("%s.pdb" % cache_id)
            dssp_path = dssp_dir / ("%s.dssp" % cache_id)
            pdb2rr_path = pdb2rr_dir / ("%s.pdb2rr" % cache_id)
            if not (pdb_path.exists() and dssp_path.exists() and pdb2rr_path.exists()):
                if args.allow_missing:
                    missing.append(sample_id)
                    continue
                raise FileNotFoundError("Missing cache files for %s (%s)" % (sample_id, cache_id))

            feature = compute_structure_features(pdb_path, dssp_path, pdb2rr_path, sequence)
            if feature.shape != (len(sequence), 45):
                raise ValueError("Invalid structure feature shape for %s: expected (%d, 45), got %s" %
                                 (sample_id, len(sequence), feature.shape))
            if not np.isfinite(feature).all():
                raise ValueError("Non-finite structure features for %s" % sample_id)
            by_id[sample_id] = torch.from_numpy(feature)
            if feature_dim is None:
                feature_dim = int(feature.shape[1])

    if not by_id:
        raise ValueError("No structure features were generated; inspect the mapping CSV and cache files")
    payload = {
        "by_id": by_id,
        "feature_dim": feature_dim or 0,
        "num_samples": len(by_id),
        "missing_ids": missing,
        "cache_dir": str(cache_dir),
    }

    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.suffix.lower() == ".json":
        json_payload = dict(payload)
        json_payload["by_id"] = {key: value.tolist() for key, value in by_id.items()}
        with open(output_path, "w", encoding="utf-8") as fout:
            json.dump(json_payload, fout, ensure_ascii=False, indent=2)
    else:
        torch.save(payload, output_path)

    print("saved=%s" % output_path)
    print("num_samples=%d" % len(by_id))
    print("feature_dim=%d" % (feature_dim or 0))
    if missing:
        print("missing=%d" % len(missing))


if __name__ == "__main__":
    main()
