"""Inventory release files and hash active inputs; no model execution.

Use --hash-all to hash every archived file, including intermediate checkpoints.
Default hashes code/docs/configs/checkpoints and active data/ESM weights, plus small
archived files. Large historical files remain indexed by path and byte size.
"""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--hash-all", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    destination = root / "provenance"
    destination.mkdir(exist_ok=True)
    excluded = {"provenance/file_manifest.csv", "provenance/release_summary.json"}
    active = {"model", "util", "scripts", "configs", "docs", "checkpoints", "data"}
    active_real_roots = [(root / name).resolve() for name in ("data/processed", "data/raw", "assets/esm")
                         if (root / name).exists()]
    rows = []
    categories = {}
    external_links = []
    for base, dirs, files in os.walk(root, followlinks=False):
        dirs[:] = sorted(d for d in dirs if d not in {".git", "__pycache__"})
        names = files + [d for d in dirs if (Path(base) / d).is_symlink()]
        for name in sorted(names):
            path = Path(base) / name
            relative = path.relative_to(root).as_posix()
            if relative in excluded or name.endswith((".pyc", ".pyo")):
                continue
            category = relative.split("/")[0] if "/" in relative else "root"
            if path.is_symlink():
                target = path.resolve()
                outside = not target.is_relative_to(root)
                if outside or not target.exists():
                    external_links.append({"path": relative, "target": str(target), "exists": target.exists()})
                rows.append({"path": relative, "kind": "symlink", "bytes": 0,
                             "sha256": "", "hash_policy": "symlink", "link_target": os.readlink(path)})
                continue
            if not path.is_file():
                continue
            size = path.stat().st_size
            hashed = (args.hash_all or category in active or size <= 1024 * 1024
                      or any(path.is_relative_to(real_root) for real_root in active_real_roots))
            rows.append({"path": relative, "kind": "file", "bytes": size,
                         "sha256": sha256(path) if hashed else "",
                         "hash_policy": "sha256" if hashed else "size_index_only",
                         "link_target": ""})
            count = categories.setdefault(category, {"files": 0, "bytes": 0})
            count["files"] += 1
            count["bytes"] += size
    with (destination / "file_manifest.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["path", "kind", "bytes", "sha256", "hash_policy", "link_target"])
        writer.writeheader()
        writer.writerows(rows)
    summary = {"root": str(root), "files": sum(r["kind"] == "file" for r in rows),
               "total_file_bytes": sum(r["bytes"] for r in rows), "categories": categories,
               "sha256_files": sum(bool(r["sha256"]) for r in rows),
               "unresolved_or_external_symlinks": external_links,
               "hash_policy": "all_files" if args.hash_all else "active_inputs_and_small_archived_files",
               "scope": "File inventory and active-input hashes; no model tests or training."}
    (destination / "release_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
