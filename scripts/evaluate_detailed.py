import argparse
import ast
import csv
import json
import math
import os
import sys

import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils import data as torch_data

from torchdrug import core, data, utils
from torchdrug.utils import comm

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import model
from util import datasets, tasks
from util import common as util


def load_solver(config_path, dataset_path, gpus, plm_embedding_file=None, contact_edge_file=None,
                structure_feature_file=None):
    util.require_single_process_inference(gpus)
    cfg = util.load_config(
        config_path,
        context={
            "output_dir": "./outputs_eval_binary_metrics",
            "dataset_path": dataset_path,
            "gpus": gpus,
            "plm_embedding_file": plm_embedding_file,
            "contact_edge_file": contact_edge_file,
            "structure_feature_file": structure_feature_file,
        },
    )
    cfg.engine.gpus = gpus
    cfg.train.num_epoch = 0
    if cfg.dataset.get("augment", None):
        cfg.dataset.augment = "none"
    if cfg.dataset.get("augment_prob", None) is not None:
        cfg.dataset.augment_prob = 0.0
    dataset_obj = core.Configurable.load_config_dict(cfg.dataset)
    solver, _ = util.build_solver(cfg, dataset_obj)
    return solver


def predict_dataset(solver, split, truncate_node_feature_dim=None, return_records=False):
    util.require_single_process_inference(world_size=solver.world_size)
    dataset_obj = getattr(solver, "%s_set" % split)
    if not len(dataset_obj):
        raise ValueError("Cannot evaluate an empty `%s` split" % split)
    sampler = torch_data.SequentialSampler(dataset_obj)
    dataloader = data.DataLoader(dataset_obj, solver.batch_size, sampler=sampler, num_workers=solver.num_worker)
    model = solver.model
    model.split = split
    model.eval()

    preds = []
    targets = []
    ids = []
    records = []
    with torch.no_grad():
        for batch in dataloader:
            if solver.device.type == "cuda":
                batch = utils.cuda(batch, device=solver.device)
            if truncate_node_feature_dim is not None:
                graph = batch["graph"]
                graph.unit_feature = graph.unit_feature[:, :truncate_node_feature_dim]
            pred, target = model.predict_and_target(batch)
            preds.append(pred.detach().cpu())
            targets.append(target.detach().cpu())
            batch_ids = batch.get("id")
            if batch_ids is not None:
                ids.extend(batch_ids)
                sequences = batch.get("sequence", [""] * len(batch_ids))
                sources = batch.get("source", [""] * len(batch_ids))
                records.extend({"id": row_id, "sequence": sequence, "source": source}
                               for row_id, sequence, source in zip(batch_ids, sequences, sources))

    pred = utils.cat(preds)
    target = utils.cat(targets)
    if len(ids) != len(pred) or len(set(ids)) != len(ids):
        raise ValueError("Evaluation requires one unique sample ID for every prediction")
    return pred, target, records if return_records else ids


def compute_binary_metrics(pred, target):
    if pred.ndim != 2 or pred.shape[1] != 2 or len(pred) == 0:
        raise ValueError("Binary evaluation requires two logits per sample")
    if not torch.isfinite(pred).all() or not torch.isfinite(target).all():
        raise ValueError("Predictions and labels must contain only finite values")
    target = target.squeeze(-1)
    if len(target) != len(pred) or not torch.all((target == 0) | (target == 1)):
        raise ValueError("Binary targets must be 0 or 1 with one target per prediction")
    target = target.long()
    pred_label = pred.argmax(dim=-1).long()
    prob = pred.softmax(dim=-1)[:, 1]

    tp = int(((pred_label == 1) & (target == 1)).sum().item())
    tn = int(((pred_label == 0) & (target == 0)).sum().item())
    fp = int(((pred_label == 1) & (target == 0)).sum().item())
    fn = int(((pred_label == 0) & (target == 1)).sum().item())
    total = tp + tn + fp + fn

    accuracy = (tp + tn) / total if total else 0.0
    sensitivity = tp / (tp + fn) if (tp + fn) else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    f1 = (2 * precision * sensitivity / (precision + sensitivity)) if (precision + sensitivity) else 0.0

    denom = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = ((tp * tn - fp * fn) / denom) if denom else 0.0
    f1_negative = (2 * tn / (2 * tn + fp + fn)) if (2 * tn + fp + fn) else 0.0
    has_both_classes = bool((target == 0).any() and (target == 1).any())
    probability_array = prob.detach().cpu().numpy()
    target_array = target.detach().cpu().numpy()
    auroc = float(roc_auc_score(target_array, probability_array)) if has_both_classes else None
    auprc = float(average_precision_score(target_array, probability_array)) if bool((target == 1).any()) else None

    return {
        "accuracy": accuracy,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "precision": precision,
        "f1": f1,
        "macro_f1": (f1 + f1_negative) / 2,
        "balanced_accuracy": (sensitivity + specificity) / 2,
        "mcc": mcc,
        "auroc": auroc,
        "auprc": auprc,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "num_samples": total,
        "mean_positive_probability": float(prob.mean().item()),
    }


def write_predictions(records, pred, target, output_path):
    probabilities = pred.softmax(dim=-1)[:, 1]
    predicted_labels = pred.argmax(dim=-1)
    targets = target.squeeze(-1).long()
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", encoding="utf-8", newline="") as fout:
        writer = csv.DictWriter(fout, fieldnames=[
            "id", "sequence", "source", "target", "predicted_label",
            "toxicity_probability", "logit_0", "logit_1",
        ])
        writer.writeheader()
        for index, record in enumerate(records):
            writer.writerow({
                **record,
                "target": int(targets[index]),
                "predicted_label": int(predicted_labels[index]),
                "toxicity_probability": float(probabilities[index]),
                "logit_0": float(pred[index, 0]),
                "logit_1": float(pred[index, 1]),
            })


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--dataset_path", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="test", choices=["train", "valid", "test"])
    parser.add_argument("--plm_embedding_file", default=None)
    parser.add_argument("--contact_edge_file", default=None)
    parser.add_argument("--structure_feature_file", default=None)
    parser.add_argument("--truncate_node_feature_dim", type=int, default=None)
    parser.add_argument("--output_json", default=None)
    parser.add_argument("--output_predictions", default=None,
                        help="Write ordered per-sample labels, probabilities and logits to CSV")
    parser.add_argument("--gpus", default="null")
    args = parser.parse_args()

    gpus = None if args.gpus == "null" else ast.literal_eval(args.gpus)
    util.require_single_process_inference(gpus)

    util.get_root_logger(file=False)
    solver = load_solver(
        args.config,
        args.dataset_path,
        gpus,
        args.plm_embedding_file,
        args.contact_edge_file,
        args.structure_feature_file,
    )
    solver.load(args.checkpoint, load_optimizer=False)
    pred, target, records = predict_dataset(solver, args.split, args.truncate_node_feature_dim, return_records=True)
    metrics = compute_binary_metrics(pred, target)

    result = {
        "config": args.config,
        "dataset_path": args.dataset_path,
        "checkpoint": args.checkpoint,
        "split": args.split,
        "plm_embedding_file": args.plm_embedding_file,
        "structure_feature_file": args.structure_feature_file,
        "contact_edge_file": args.contact_edge_file,
        "truncate_node_feature_dim": args.truncate_node_feature_dim,
        "positive_label": 1,
        "decision_rule": "argmax of logits; class 1 is toxic",
        "auprc_definition": "average precision (sklearn.average_precision_score)",
        "auroc_definition": "ROC area with tied scores handled by sklearn",
        "output_predictions": args.output_predictions,
        "metrics": metrics,
    }

    if args.output_predictions:
        write_predictions(records, pred, target, args.output_predictions)
    if args.output_json:
        os.makedirs(os.path.dirname(os.path.abspath(args.output_json)), exist_ok=True)
        with open(args.output_json, "w", encoding="utf-8") as fout:
            json.dump(result, fout, ensure_ascii=False, indent=2)

    if comm.get_rank() == 0:
        print(json.dumps(result, ensure_ascii=False, indent=2))
        if args.output_json:
            print("Wrote %s" % args.output_json)


if __name__ == "__main__":
    main()
