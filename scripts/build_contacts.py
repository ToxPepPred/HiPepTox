import argparse
import json
import math
from collections import OrderedDict
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Build residue contact edges from PDB files.")
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--pdb-dir", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--id-col", default="id")
    parser.add_argument("--pdb-name-col", default="id",
                        help="CSV column used to resolve `<pdb-dir>/<value>.pdb`.")
    parser.add_argument("--distance-cutoff", type=float, default=8.0)
    parser.add_argument("--allow-missing", action="store_true")
    return parser.parse_args()


def load_residue_centroids(pdb_path):
    residues = OrderedDict()
    with open(pdb_path, "r", encoding="utf-8") as fin:
        for line in fin:
            if not line.startswith("ATOM"):
                continue
            chain_id = line[21].strip() or "_"
            residue_id = line[22:26].strip()
            insertion = line[26].strip()
            key = (chain_id, residue_id, insertion)
            x = float(line[30:38].strip())
            y = float(line[38:46].strip())
            z = float(line[46:54].strip())
            residues.setdefault(key, []).append((x, y, z))

    centroids = []
    for coords in residues.values():
        count = float(len(coords))
        mean_x = sum(coord[0] for coord in coords) / count
        mean_y = sum(coord[1] for coord in coords) / count
        mean_z = sum(coord[2] for coord in coords) / count
        centroids.append((mean_x, mean_y, mean_z))
    return centroids


def build_contact_edges(centroids, cutoff):
    cutoff2 = cutoff * cutoff
    edges = []
    for i in range(len(centroids)):
        x1, y1, z1 = centroids[i]
        for j in range(i + 1, len(centroids)):
            x2, y2, z2 = centroids[j]
            dist2 = (x1 - x2) ** 2 + (y1 - y2) ** 2 + (z1 - z2) ** 2
            if dist2 < cutoff2:
                edges.append([i, j])
    return edges


def main():
    args = parse_args()
    input_csv = Path(args.input_csv)
    pdb_dir = Path(args.pdb_dir)
    output_json = Path(args.output_json)

    import csv

    by_id = {}
    missing = []
    with input_csv.open("r", encoding="utf-8") as fin:
        reader = csv.DictReader(fin)
        for row in reader:
            sample_id = str(row[args.id_col]).strip()
            pdb_stem = str(row[args.pdb_name_col]).strip()
            pdb_path = pdb_dir / f"{pdb_stem}.pdb"
            if not pdb_path.exists():
                if args.allow_missing:
                    missing.append(sample_id)
                    continue
                raise FileNotFoundError(f"Missing PDB for {sample_id}: {pdb_path}")
            centroids = load_residue_centroids(pdb_path)
            by_id[sample_id] = build_contact_edges(centroids, args.distance_cutoff)

    output_json.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "distance_cutoff": args.distance_cutoff,
        "pdb_dir": str(pdb_dir),
        "by_id": by_id,
    }
    if missing:
        payload["missing_ids"] = missing
    output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"saved={output_json}")
    print(f"num_samples={len(by_id)}")
    if missing:
        print(f"missing={len(missing)}")


if __name__ == "__main__":
    main()
