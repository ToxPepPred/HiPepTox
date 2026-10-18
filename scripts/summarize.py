#!/usr/bin/env python3
"""Rebuild historical summaries from saved scan CSVs without loading a model.

This script performs no new evaluation. External test data were used in some
historical development steps; recovering validation-selected rows cannot undo
that history. Test-selected rows are retained only as exploratory records.

Default: read all matching archived runs, never choose a rerun by its test score.
--run-list accepts a JSON list of relative run directories, or {"runs": [...]},
to add an explicitly declared cohort to summary.csv without removing other runs.
--model-manifest accepts {"models": [{"run_dir": "archive/...", ...}, ...]}
and writes a separate main_models.csv for those declared directories.
Both lists select directories only; validation epochs still come from the
best_valid_epoch_N.pth filename, never from a test metric.
Historical weight files may be excluded: --checkpoint-inventory defaults to
ROOT/provenance/historical_checkpoint_inventory.json and supplies their names.
Only packaged weights are hashed; absent historical weights are not failures.
"""

import argparse
import csv
import hashlib
import json
import math
import re
import statistics
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path


TOLERANCE = 1e-5
BEST_VALID = re.compile(r"best_valid_epoch_(\d+)\.pth$")
METRICS = ("accuracy", "sensitivity", "specificity", "precision", "binary_f1", "binary_mcc")
COUNTS = ("tp", "tn", "fp", "fn")
ALIASES = {
    "accuracy": ("binary_accuracy", "accuracy [toxicity]", "accuracy"),
    "sensitivity": ("sensitivity",),
    "specificity": ("specificity",),
    "precision": ("precision",),
    "binary_f1": ("binary_f1",),
    "binary_mcc": ("binary_mcc", "matthews correlation [toxicity]", "mcc [toxicity]"),
}
RUN_FIELDS = (
    "selection", "config_group", "legacy_group", "dataset", "dataset_source",
    "scan_protocol", "seed", "seed_source", "rerun", "run_dir", "epoch",
    "checkpoint", "checkpoint_included", "checkpoint_bytes", "checkpoint_sha256",
    "checkpoint_hash_source", "historical_checkpoint", "best_valid_checkpoint",
    "best_valid_checkpoint_included", "best_valid_checkpoint_bytes",
    "best_valid_checkpoint_sha256", "best_valid_epoch_source", "recorded_checkpoint_name", "scan_csv",
    "scan_csv_sha256", "scan_row_number", "metrics_source", "selection_criterion",
    "selection_value", "declared_run_list", "declared_main_model",
) + METRICS + COUNTS + ("num_samples",) + tuple("scan_" + x for x in METRICS)
SUMMARY_FIELDS = (
    "cohort", "selection", "config_group", "legacy_group", "dataset",
    "scan_protocol", "num_samples", "n_runs", "n_rows", "seed_labels",
    "seed_sources", "run_dirs", "historical_only", "sample_std_ddof",
) + tuple(x + suffix for x in METRICS for suffix in ("_n", "_mean", "_sample_std"))


def relative(path, root):
    return path.relative_to(root).as_posix()


def number(value):
    if value is None or str(value).strip() == "":
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def integral(value):
    parsed = number(value)
    if parsed is None or parsed < 0 or not parsed.is_integer():
        return None
    return int(parsed)


def original_metric(row, name):
    for key in ALIASES[name]:
        value = number(row.get(key))
        if value is not None:
            return value, key
    return None, None


def sha256(path, cache):
    key = str(path)
    if key not in cache:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        cache[key] = digest.hexdigest()
    return cache[key]


def warn(warnings, code, context, **details):
    warnings.append({"code": code, **context, **details})


def derive_metrics(row, context, warnings):
    """Use valid confusion counts; retain original CSV metrics for comparison."""
    raw = {name: original_metric(row, name)[0] for name in METRICS}
    counts = {name: integral(row.get(name)) for name in COUNTS}
    stated_n = integral(row.get("num_samples"))
    if any(value is None for value in counts.values()):
        warn(warnings, "confusion_counts_unavailable", context,
             detail="Using finite original scan metrics where present; counts are incomplete/invalid.")
        return {**raw, **counts, "num_samples": stated_n,
                "metrics_source": "original_scan_metrics",
                **{"scan_" + k: v for k, v in raw.items()}}

    tp, tn, fp, fn = (counts[k] for k in COUNTS)
    total = tp + tn + fp + fn
    if total == 0:
        warn(warnings, "empty_confusion_matrix", context)
        return {**{name: None for name in METRICS}, **counts, "num_samples": 0,
                "metrics_source": "empty_confusion_matrix",
                **{"scan_" + k: v for k, v in raw.items()}}

    def divide(numerator, denominator, metric):
        if denominator == 0:
            warn(warnings, "undefined_metric_denominator", context, metric=metric,
                 detail="Recorded as empty, not silently assigned a zero score.")
            return None
        return numerator / denominator

    denominator = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
    derived = {
        "accuracy": (tp + tn) / total,
        "sensitivity": divide(tp, tp + fn, "sensitivity"),
        "specificity": divide(tn, tn + fp, "specificity"),
        "precision": divide(tp, tp + fp, "precision"),
        "binary_f1": divide(2 * tp, 2 * tp + fp + fn, "binary_f1"),
        "binary_mcc": divide(tp * tn - fp * fn, math.sqrt(denominator), "binary_mcc"),
    }
    if stated_n is not None and stated_n != total:
        warn(warnings, "sample_count_mismatch", context,
             scan_num_samples=stated_n, confusion_total=total)
    for name, calculated in derived.items():
        for key in ALIASES[name]:
            observed = number(row.get(key))
            if observed is not None and calculated is not None and abs(observed - calculated) > TOLERANCE:
                warn(warnings, "metric_mismatch", context, metric=name, scan_column=key,
                     scan_value=observed, confusion_value=calculated,
                     tolerance=TOLERANCE,
                     detail="Source CSV is unchanged; both source and derived values are retained.")
    return {**derived, **counts, "num_samples": total,
            "metrics_source": "confusion_matrix",
            **{"scan_" + k: v for k, v in raw.items()}}


def config_metadata(run_dir, output_root):
    parts = run_dir.relative_to(output_root).parts
    if "by_config" in parts:
        idx = parts.index("by_config")
        if idx + 1 < len(parts):
            return parts[idx + 1], ""
    # Do not merge unknown legacy configurations simply because both lack a tag.
    return "unclassified_legacy", parts[0] if parts else "."


def dataset_metadata(scan_path, config_group, run_dir):
    pattern = re.compile(r"(?:toxipep|cd[_-]?hit)[_.-]?0?[._-]?([89])(?!\d)", re.I)
    for source, value in (("scan_filename", scan_path.name), ("config_group", config_group),
                          ("run_directory", run_dir.as_posix())):
        match = pattern.search(value)
        if match:
            return "toxipep0" + match.group(1), source
    return "unknown", "not_inferred"


def seed_metadata(run_dir):
    text = run_dir.as_posix()
    matches = list(re.finditer(r"(?:^|[/_-])seed[_-]?(\d+)(?!\d)", text, re.I))
    if matches:
        values = sorted({int(match.group(1)) for match in matches})
        if len(values) == 1:
            return "seed" + str(values[0]), "explicit_seed_in_directory"
        return "ambiguous:" + "/".join(map(str, values)), "conflicting_directory_seed_tokens"
    matches = list(re.finditer(r"(?:^|[/_-])(3407|2024)(?=[/_-]|$)", text))
    # 2024 is treated as a seed only if it is not the start of a YYYY-MM-DD date.
    matches = [match for match in matches
               if not re.match(r"-\d{2}-\d{2}", text[match.end():])]
    values = sorted({int(match.group(1)) for match in matches})
    if len(values) == 1:
        return "seed" + str(values[0]), "numeric_seed_token_in_directory"
    if len(values) > 1:
        return "ambiguous:" + "/".join(map(str, values)), "conflicting_directory_seed_tokens"
    return "seed0", "inferred_default_no_explicit_seed"


def scan_protocol(path):
    """Remove run labels, preserve experiment/dataset tags to avoid broad merging."""
    text = path.stem
    text = re.sub(r"[_-]scan$", "", text, flags=re.I)
    text = re.sub(r"[_-]seed[_-]?\d+(?:[_-]rerun)?", "", text, flags=re.I)
    text = re.sub(r"[_-](?:3407|2024)(?:[_-]rerun)?(?=[_-]|$)", "", text, flags=re.I)
    text = re.sub(r"[_-]rerun", "", text, flags=re.I)
    text = re.sub(r"[_-]\d{4}-\d{2}-\d{2}(?:-\d{2}-\d{2}-\d{2})?", "", text)
    return re.sub(r"_+", "_", text).strip("_- ")


def load_declared_runs(path, root, key):
    if path is None:
        return set()
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    entries = payload.get(key, []) if isinstance(payload, dict) else payload
    if not isinstance(entries, list):
        raise ValueError("%s must contain a JSON list or a %r list" % (path, key))
    result = set()
    for entry in entries:
        value = entry.get("run_dir") if isinstance(entry, dict) else entry
        if not isinstance(value, str) or not value.strip():
            raise ValueError("Declared run entry must be a string or contain run_dir: %r" % entry)
        candidate = Path(value)
        if candidate.is_absolute():
            raise ValueError("Declared run_dir must be relative to Paper: %s" % value)
        normalized = (root / candidate).resolve()
        try:
            normalized.relative_to(root)
        except ValueError as error:
            raise ValueError("Declared run_dir escapes Paper: %s" % value) from error
        result.add(normalized.relative_to(root).as_posix())
    return result


def load_checkpoint_inventory(path, root):
    if not path.is_file():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    entries = payload.get("checkpoints", []) if isinstance(payload, dict) else []
    if not isinstance(entries, list):
        raise ValueError("Checkpoint inventory must contain a checkpoints list")
    result = {}
    for entry in entries:
        if not isinstance(entry, dict) or not isinstance(entry.get("path"), str):
            raise ValueError("Checkpoint inventory entries must contain a path")
        candidate = Path(entry["path"])
        if candidate.is_absolute() or ".." in candidate.parts:
            raise ValueError("Inventory checkpoint paths must stay relative to Paper")
        result[candidate.as_posix()] = entry
    return result


def load_main_model_paths(path, root):
    if path is None:
        return {}
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    entries = payload.get("models", []) if isinstance(payload, dict) else []
    result = {}
    for entry in entries:
        if not isinstance(entry, dict) or not entry.get("run_dir") or not entry.get("checkpoint"):
            continue
        run_dir = Path(entry["run_dir"])
        checkpoint = Path(entry["checkpoint"])
        if run_dir.is_absolute() or checkpoint.is_absolute() or ".." in run_dir.parts or ".." in checkpoint.parts:
            raise ValueError("Main model paths must stay relative to Paper")
        result[run_dir.as_posix()] = root / checkpoint
    return result


def checkpoint_metadata(path, root, inventory, hashes):
    entry = inventory.get(relative(path, root), {})
    included = path.is_file()
    if included:
        return {"included": True, "bytes": path.stat().st_size,
                "sha256": sha256(path, hashes), "hash_source": "packaged_file"}
    inventory_hash = entry.get("sha256") or ""
    return {"included": False, "bytes": entry.get("bytes"), "sha256": inventory_hash,
            "hash_source": "inventory" if inventory_hash else "not_packaged_no_hash"}


def read_scan(path):
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or "epoch" not in reader.fieldnames:
            raise ValueError("scan CSV has no epoch header")
        result = []
        for line_number, row in enumerate(reader, start=2):
            if any(str(value or "").strip() for value in row.values()):
                result.append((line_number, row))
        return result


def write_csv(path, fields, records):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for record in records:
            writer.writerow({k: "" if record.get(k) is None else record.get(k, "") for k in fields})


def summarize(records, cohort, warnings):
    grouped = defaultdict(list)
    keys = ("selection", "config_group", "legacy_group", "dataset", "scan_protocol", "num_samples")
    for record in records:
        grouped[tuple(record.get(key) for key in keys)].append(record)
    summaries = []
    for group_key, items in sorted(grouped.items(), key=lambda pair: tuple(str(x) for x in pair[0])):
        result = {"cohort": cohort, **dict(zip(keys, group_key)), "historical_only": True,
                  "sample_std_ddof": 1, "n_rows": len(items),
                  "n_runs": len({item["run_dir"] for item in items})}
        result["seed_labels"] = ";".join(sorted({item["seed"] for item in items}))
        result["seed_sources"] = ";".join(sorted({item["seed_source"] for item in items}))
        result["run_dirs"] = ";".join(sorted({item["run_dir"] for item in items}))
        if result["n_rows"] != result["n_runs"]:
            warn(warnings, "multiple_scan_rows_for_same_run", {k: result[k] for k in keys},
                 cohort=cohort, detail="Not summarized: duplicate scan sources would overweight one run.")
            for name in METRICS:
                result[name + "_n"] = 0
                result[name + "_mean"] = None
                result[name + "_sample_std"] = None
        else:
            seed_counts = defaultdict(int)
            for item in items:
                seed_counts[item["seed"]] += 1
            duplicate_seeds = {seed: count for seed, count in seed_counts.items() if count > 1}
            if duplicate_seeds:
                warn(warnings, "repeated_seed_runs", {k: result[k] for k in keys}, cohort=cohort,
                     counts=duplicate_seeds,
                     detail="All matched historical runs retained; n_runs is not number of independent seeds.")
            for name in METRICS:
                values = [item[name] for item in items if item.get(name) is not None]
                result[name + "_n"] = len(values)
                result[name + "_mean"] = statistics.mean(values) if values else None
                result[name + "_sample_std"] = statistics.stdev(values) if len(values) > 1 else None
        summaries.append(result)
    return summaries


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, help="Default: ROOT/results/historical")
    parser.add_argument("--run-list", type=Path, help="JSON list or {runs: [...]} with relative run_dir")
    parser.add_argument("--model-manifest", type=Path, help="JSON {models: [{run_dir: ...}, ...]}")
    parser.add_argument("--checkpoint-inventory", type=Path,
                        help="Default: ROOT/provenance/historical_checkpoint_inventory.json")
    args = parser.parse_args()
    root = args.root.resolve()
    output = (args.output or root / "results" / "historical").resolve()
    output.mkdir(parents=True, exist_ok=True)
    source_root = root / "archive" / "source_project" / "output"
    declared = load_declared_runs(args.run_list, root, "runs")
    model_runs = load_declared_runs(args.model_manifest, root, "models")
    main_models = load_main_model_paths(args.model_manifest, root)
    inventory_path = args.checkpoint_inventory or root / "provenance" / "historical_checkpoint_inventory.json"
    inventory = load_checkpoint_inventory(inventory_path, root)
    missing, warnings, records = [], [], []
    hashes = {}
    best_files = sorted(source_root.rglob("best_valid_epoch_*.pth")) if source_root.is_dir() else []
    scan_files = sorted(source_root.rglob("*_scan.csv")) if source_root.is_dir() else []
    best_by_dir = defaultdict(list)
    scans_by_dir = defaultdict(list)
    for path in best_files:
        if BEST_VALID.fullmatch(path.name):
            best_by_dir[path.parent].append(path)
    for inventory_relative in inventory:
        inventory_file = root / inventory_relative
        if BEST_VALID.fullmatch(inventory_file.name):
            try:
                inventory_file.relative_to(source_root)
            except ValueError:
                continue
            if inventory_file not in best_by_dir[inventory_file.parent]:
                best_by_dir[inventory_file.parent].append(inventory_file)
    for path in scan_files:
        scans_by_dir[path.parent].append(path)
    if not source_root.is_dir():
        missing.append({"reason": "source_output_directory_absent",
                        "path": "archive/source_project/output"})
    for run_dir in sorted(set(best_by_dir) | set(scans_by_dir)):
        run_relative = relative(run_dir, root)
        best = best_by_dir[run_dir]
        scans = scans_by_dir[run_dir]
        if not best:
            missing.append({"run_dir": run_relative, "reason": "best_valid_checkpoint_absent",
                            "scan_csvs": [relative(path, root) for path in scans]})
            continue
        if len(best) != 1:
            missing.append({"run_dir": run_relative, "reason": "ambiguous_best_valid_checkpoints",
                            "checkpoints": [relative(path, root) for path in best],
                            "detail": "No epoch selected using external scores."})
            continue
        if not scans:
            missing.append({"run_dir": run_relative, "reason": "same_directory_scan_csv_absent",
                            "checkpoint": relative(best[0], root)})
            continue
        best_path = best[0]
        valid_epoch = int(BEST_VALID.fullmatch(best_path.name).group(1))
        valid_epoch_source = "packaged_best_valid_filename" if best_path.is_file() else "inventory_best_valid_filename"
        group, legacy = config_metadata(run_dir, source_root)
        seed, seed_source = seed_metadata(run_dir)
        if seed_source.startswith("conflicting"):
            warn(warnings, "ambiguous_seed", {"run_dir": run_relative}, inferred_label=seed)
        for scan_path in scans:
            scan_relative = relative(scan_path, root)
            context = {"run_dir": run_relative, "scan_csv": scan_relative}
            try:
                rows = read_scan(scan_path)
            except (OSError, ValueError, csv.Error) as error:
                missing.append({**context, "reason": "scan_read_error", "error": str(error)})
                continue
            valid_matches = [(line, row) for line, row in rows if integral(row.get("epoch")) == valid_epoch]
            if len(valid_matches) != 1:
                missing.append({**context, "reason": "validation_epoch_row_absent_or_ambiguous",
                                "validation_epoch": valid_epoch, "matching_rows": len(valid_matches)})
                continue
            eligible = []
            for line, row in rows:
                epoch = integral(row.get("epoch"))
                value = number(row.get("binary_accuracy"))
                if epoch is not None and value is not None:
                    eligible.append((value, -epoch, -line, line, row))
            if not eligible:
                missing.append({**context, "reason": "finite_binary_accuracy_for_exploratory_selection_absent"})
            dataset, dataset_source = dataset_metadata(scan_path, group, run_dir)
            try:
                best_metadata = checkpoint_metadata(best_path, root, inventory, hashes)
                scan_hash = sha256(scan_path, hashes)
            except OSError as error:
                missing.append({**context, "reason": "provenance_hash_error", "error": str(error)})
                continue
            choices = [("validation_selected", *valid_matches[0], "best_valid_filename", valid_epoch)]
            if eligible:
                choice = max(eligible, key=lambda item: item[:3])
                ties = sum(item[0] == choice[0] for item in eligible)
                if ties > 1:
                    warn(warnings, "exploratory_accuracy_tie", context, tied_rows=ties,
                         detail="Earliest epoch then earliest CSV line selected only for exploratory record.")
                choices.append(("test_selected_exploratory", choice[3], choice[4],
                                "max_scan_binary_accuracy", choice[0]))
            for selection, line, row, criterion, selection_value in choices:
                epoch = integral(row.get("epoch"))
                recorded_name = str(row.get("checkpoint", "")).replace("\\", "/").rsplit("/", 1)[-1]
                historical_checkpoint = best_path if selection == "validation_selected" else (
                    run_dir / (recorded_name or "model_epoch_%d.pth" % epoch))
                checkpoint_path = historical_checkpoint
                if selection == "validation_selected" and run_relative in main_models:
                    checkpoint_path = main_models[run_relative]
                    if not checkpoint_path.is_file():
                        missing.append({**context, "reason": "declared_main_model_file_absent",
                                        "checkpoint": relative(checkpoint_path, root)})
                try:
                    selected_metadata = checkpoint_metadata(checkpoint_path, root, inventory, hashes)
                except OSError as error:
                    warn(warnings, "checkpoint_hash_error", {**context, "epoch": epoch}, error=str(error))
                    selected_metadata = {"included": checkpoint_path.is_file(), "bytes": None,
                                         "sha256": "", "hash_source": "hash_error"}
                if not recorded_name:
                    warn(warnings, "scan_checkpoint_name_absent", {**context, "epoch": epoch})
                record = {
                    "selection": selection, "config_group": group, "legacy_group": legacy,
                    "dataset": dataset, "dataset_source": dataset_source,
                    "scan_protocol": scan_protocol(scan_path), "seed": seed, "seed_source": seed_source,
                    "rerun": bool(re.search(r"(?:^|[_/-])rerun(?:[_/-]|$)", run_dir.as_posix(), re.I)),
                    "run_dir": run_relative, "epoch": epoch,
                    "checkpoint": relative(checkpoint_path, root),
                    "checkpoint_included": selected_metadata["included"],
                    "checkpoint_bytes": selected_metadata["bytes"],
                    "checkpoint_sha256": selected_metadata["sha256"],
                    "checkpoint_hash_source": selected_metadata["hash_source"],
                    "historical_checkpoint": relative(historical_checkpoint, root),
                    "best_valid_checkpoint": relative(best_path, root),
                    "best_valid_checkpoint_included": best_metadata["included"],
                    "best_valid_checkpoint_bytes": best_metadata["bytes"],
                    "best_valid_checkpoint_sha256": best_metadata["sha256"],
                    "best_valid_epoch_source": valid_epoch_source,
                    "recorded_checkpoint_name": recorded_name,
                    "scan_csv": scan_relative, "scan_csv_sha256": scan_hash,
                    "scan_row_number": line, "selection_criterion": criterion,
                    "selection_value": selection_value,
                    "declared_run_list": run_relative in declared,
                    "declared_main_model": run_relative in model_runs,
                }
                record.update(derive_metrics(row, {**context, "epoch": epoch, "selection": selection}, warnings))
                records.append(record)
    matched_runs = {row["run_dir"] for row in records}
    for cohort, wanted in (("declared_run_list", declared), ("declared_main_models", model_runs)):
        for absent in sorted(wanted - matched_runs):
            missing.append({"run_dir": absent, "reason": "declared_run_not_matched", "cohort": cohort})
    validation = [row for row in records if row["selection"] == "validation_selected"]
    exploratory = [row for row in records if row["selection"] == "test_selected_exploratory"]
    write_csv(output / "valid_selected.csv", RUN_FIELDS, validation)
    write_csv(output / "test_exploratory.csv", RUN_FIELDS, exploratory)
    summaries = summarize(records, "all_archived_runs", warnings)
    if args.run_list is not None:
        summaries += summarize([row for row in records if row["declared_run_list"]], "declared_run_list", warnings)
    write_csv(output / "summary.csv", SUMMARY_FIELDS, summaries)
    if args.model_manifest is not None:
        main_records = [row for row in validation if row["declared_main_model"]]
        write_csv(output / "main_models.csv", SUMMARY_FIELDS,
                  summarize(main_records, "declared_main_models", warnings))
    metadata = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "historical_only": True, "new_model_evaluation": False,
        "source_output": "archive/source_project/output", "metric_tolerance": TOLERANCE,
        "best_valid_files_included": len(best_files), "scan_csvs_found": len(scan_files),
        "checkpoint_inventory_entries": len(inventory),
        "best_valid_records_found": sum(len(items) for items in best_by_dir.values()),
        "historical_weights_may_be_excluded": True,
        "validation_rows": len(validation), "exploratory_rows": len(exploratory),
        "missing_count": len(missing), "warning_count": len(warnings),
        "limitation": "Historical test selection cannot be undone by recovering validation-selected rows.",
    }
    for filename, payload in (("missing.json", {**metadata, "missing": missing}),
                              ("warnings.json", {**metadata, "warnings": warnings})):
        (output / filename).write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    readme = """# 历史结果重汇总

本目录由 scripts/summarize.py 从已有 scan.csv 生成，无新训练、无新预测。
valid_selected.csv 的epoch仅来自 best_valid_epoch_N.pth 文件名：实体文件或checkpoint inventory记录。
test_exploratory.csv 按原始scan的最大 binary_accuracy 取一行，只用于探索记录。
准确度、敏感度、特异度、Precision、F1、MCC优先由有效混淆矩阵计算；原scan值保留在scan_列。
两者相差超过1e-5写入warnings.json，原始CSV不改写，不能调某个指标来配平论文表。

summary.csv默认汇总全部匹配的历史运行，按配置组、数据协议、扫描协议和样本数区分；
样本标准差使用ddof=1，只有一个run时标准差为空。未知legacy配置按来源顶层目录区分。
重复seed会告警但不会因其测试分数选择某个rerun；同一run重复扫描会告警并留空聚合指标。
使用--run-list可添加明确声明的cohort，不覆盖全部历史统计。
使用--model-manifest读取models[].run_dir，单独生成验证选checkpoint的main_models.csv。

历史epoch权重按要求不打包，provenance/historical_checkpoint_inventory.json保留原路径、体积与选模文件名。
仅六个正式主模型在models/保留物理副本；使用model-manifest时，验证行映射到这些副本并记录实际SHA256。
所有checkpoint和scan路径相对Paper根目录；checkpoint_included说明实体是否随项目提供。
未包含的历史权重不会被当作缺失评估证据，hash只采用既有inventory记录（可为空），不会读取不存在的文件。
缺失选模文件名记录、scan或指定epoch行，以及应包含但不存在的正式模型记录在missing.json。
没有明确seed标签时seed0为推断，见seed_source；脚本只重汇总历史CSV，不反序列化权重或运行模型。
完全重复检查不证明CD-HIT同源隔离。基线数据口径和历史外部测试使用情况仍需独立核验。
历史上测试集参与过开发或选模，改用验证checkpoint不能恢复其从未使用的独立性；
这些表不能标记为新盲测，也不自动证明最终论文主结果或公平基线比较。
"""
    (output / "README.md").write_text(readme, encoding="utf-8")
    print(json.dumps(metadata, ensure_ascii=False, indent=2))
    return 0 if validation else 2


if __name__ == "__main__":
    raise SystemExit(main())
