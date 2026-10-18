"""Evaluate fixed checkpoints by averaging their toxic-class probabilities."""

import argparse
import ast
import gc
import hashlib
import json
from pathlib import Path
import re
import sys

import torch

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(PROJECT_ROOT))

from evaluate_detailed import (
    compute_binary_metrics,
    load_solver,
    predict_dataset,
    write_predictions,
)
from util import common as util


def resolve_input_path(value, label, directory=False):
    """Anchor relative paths to the release root, without implicit fallbacks."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError("%s must be a nonempty path string" % label)
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = (PROJECT_ROOT / path).resolve(strict=True)
        if not path.is_relative_to(PROJECT_ROOT):
            raise ValueError("Relative %s escapes the project root: %s" % (label, value))
    else:
        path = path.resolve(strict=True)
    valid = path.is_dir() if directory else path.is_file()
    if not valid:
        raise ValueError("%s must be an existing %s: %s" % (label, "directory" if directory else "file", path))
    return path


def resolve_output_path(value):
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = (PROJECT_ROOT / path).resolve()
        if not path.is_relative_to(PROJECT_ROOT):
            raise ValueError("Relative output path escapes the project root: %s" % value)
    else:
        path = path.resolve()
    return path


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as fin:
        for chunk in iter(lambda: fin.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_manifest(path):
    with path.open("r", encoding="utf-8-sig") as fin:
        payload = json.load(fin)
    if isinstance(payload, list):
        payload = {"models": payload}
    if not isinstance(payload, dict) or not isinstance(payload.get("models"), list) or not payload["models"]:
        raise ValueError("Ensemble JSON must contain a nonempty `models` list")
    if not all(isinstance(member, dict) for member in payload["models"]):
        raise ValueError("Every ensemble member must be a JSON object")
    return payload


def member_inputs(member, args, index):
    resolved = {}
    for key in ("config", "checkpoint"):
        resolved[key] = resolve_input_path(member.get(key), "model %d %s" % (index, key))
    for key in ("plm_embedding_file", "structure_feature_file", "contact_edge_file"):
        override = getattr(args, key)
        value = override if override is not None else member.get(key)
        resolved[key] = resolve_input_path(value, "model %d %s" % (index, key)) if value is not None else None
    return resolved


def require_matching_inputs(reference_records, reference_target, records, target, index):
    reference_ids = [record["id"] for record in reference_records]
    member_ids = [record["id"] for record in records]
    if reference_ids != member_ids:
        raise ValueError("Model %d has different sample IDs or sample order" % index)
    if not torch.equal(reference_target, target):
        raise ValueError("Model %d has different target labels" % index)
    if reference_records != records:
        raise ValueError("Model %d has different sequence/source metadata" % index)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ensemble_config", required=True, help="JSON containing fixed ensemble members")
    parser.add_argument("--dataset_path", required=True)
    parser.add_argument("--split", choices=["train", "valid", "test"], default="test")
    parser.add_argument("--gpus", default="null")
    parser.add_argument("--plm_embedding_file", default=None, help="Override PLM cache for every member")
    parser.add_argument("--structure_feature_file", default=None, help="Override structure cache for every member")
    parser.add_argument("--contact_edge_file", default=None, help="Override contact cache for every member")
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--output_predictions", required=True)
    args = parser.parse_args()

    gpus = None if args.gpus == "null" else ast.literal_eval(args.gpus)
    util.require_single_process_inference(gpus)
    manifest_path = resolve_input_path(args.ensemble_config, "ensemble_config")
    dataset_path = resolve_input_path(args.dataset_path, "dataset_path", directory=True)
    manifest = load_manifest(manifest_path)
    models = manifest["models"]

    # A manifest may declare the dataset associated with its default feature
    # caches. New datasets must then provide their own required feature inputs.
    expected_value = manifest.get("expected_dataset_path", manifest.get("dataset_path"))
    if expected_value is not None:
        expected_path = resolve_input_path(expected_value, "expected_dataset_path", directory=True)
        if dataset_path != expected_path:
            missing_overrides = [
                key for key in ("plm_embedding_file", "structure_feature_file", "contact_edge_file")
                if any(member.get(key) is not None for member in models) and getattr(args, key) is None
            ]
            if missing_overrides:
                raise ValueError("New dataset requires explicit feature overrides: %s" % ", ".join("--" + key for key in missing_overrides))

    output_json = resolve_output_path(args.output_json)
    output_predictions = resolve_output_path(args.output_predictions)
    if output_json == output_predictions:
        raise ValueError("JSON metrics and CSV predictions require different output paths")
    resolved_models = [member_inputs(member, args, index) for index, member in enumerate(models, start=1)]
    input_paths = {manifest_path, (dataset_path / "peptide_toxicity.csv").resolve()}
    for inputs in resolved_models:
        input_paths.update(path for path in inputs.values() if path is not None)
    if output_json in input_paths or output_predictions in input_paths:
        raise ValueError("Evaluation outputs must not overwrite the ensemble manifest, dataset, configuration, checkpoint or feature inputs")

    util.get_root_logger(file=False)
    reference_records = None
    reference_target = None
    probabilities = []
    member_metadata = []
    seen_members = set()
    for index, (member, inputs) in enumerate(zip(models, resolved_models), start=1):
        signature = tuple(str(inputs[key]) for key in (
            "config", "checkpoint", "plm_embedding_file", "structure_feature_file", "contact_edge_file"
        ))
        if signature in seen_members:
            raise ValueError("Model %d duplicates a previous ensemble member" % index)
        seen_members.add(signature)
        solver = load_solver(
            str(inputs["config"]), str(dataset_path), gpus,
            str(inputs["plm_embedding_file"]) if inputs["plm_embedding_file"] else None,
            str(inputs["contact_edge_file"]) if inputs["contact_edge_file"] else None,
            str(inputs["structure_feature_file"]) if inputs["structure_feature_file"] else None,
        )
        solver.load(str(inputs["checkpoint"]), load_optimizer=False)
        pred, target, records = predict_dataset(solver, args.split, return_records=True)
        metrics = compute_binary_metrics(pred, target)
        if reference_records is None:
            reference_records = records
            reference_target = target
        else:
            require_matching_inputs(reference_records, reference_target, records, target, index)
        probabilities.append(pred.softmax(dim=-1)[:, 1].double())
        member_metadata.append({
            "index": index,
            "name": member.get("name", "model_%d" % index),
            "declared_member": member,
            "resolved_inputs": {key: str(value) if value is not None else None for key, value in inputs.items()},
            "checkpoint_sha256": sha256_file(inputs["checkpoint"]),
            "config_sha256": sha256_file(inputs["config"]),
            "validation_selected_filename": bool(re.fullmatch(r"best_valid_epoch_\d+\.pth", inputs["checkpoint"].name)),
            "metrics": metrics,
        })
        del solver, pred, target
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    mean_probability = torch.stack(probabilities).mean(dim=0)
    epsilon = torch.finfo(mean_probability.dtype).eps
    probability = mean_probability.clamp(min=epsilon, max=1 - epsilon)
    log_probability = torch.stack((1 - probability, probability), dim=-1).log()
    metrics = compute_binary_metrics(log_probability, reference_target)
    write_predictions(reference_records, log_probability, reference_target, str(output_predictions))
    result = {
        "ensemble": "mean_probability",
        "ensemble_config": str(manifest_path),
        "ensemble_config_sha256": sha256_file(manifest_path),
        "dataset_path": str(dataset_path),
        "split": args.split,
        "positive_label": 1,
        "auprc_definition": "average precision (sklearn.average_precision_score)",
        "auroc_definition": "ROC area with tied scores handled by sklearn",
        "decision_rule": "argmax of [1-p, p]; exact ties predict class 0",
        "num_models": len(models),
        "probability_clamp_epsilon": epsilon,
        "csv_logit_fields": "logit_0 and logit_1 contain log probabilities of the ensemble, not raw member logits",
        "checkpoint_evaluation_policy": "Evaluate the fixed supplied checkpoints; no member or epoch is selected using this evaluation split",
        "intended_checkpoint_selection_split": "validation",
        "checkpoint_selection_evidence": "Consult checkpoints/manifest.json and original training logs; filenames alone do not establish selection provenance",
        "historical_test_selection_limit": (
            "External results in the source project were inspected during model development. "
            "Re-evaluating fixed validation-selected checkpoints does not establish a new blinded external test."
        ),
        "feature_overrides": {key: getattr(args, key) for key in (
            "plm_embedding_file", "structure_feature_file", "contact_edge_file"
        )},
        "members": member_metadata,
        "metrics": metrics,
        "output_predictions": str(output_predictions),
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open("w", encoding="utf-8") as fout:
        json.dump(result, fout, ensure_ascii=False, indent=2)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
