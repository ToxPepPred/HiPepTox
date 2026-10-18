import argparse
import ast
import csv
import json
import os
import sys
import tempfile

import torch
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
            "output_dir": "./outputs_predict_unlabeled",
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


def build_temp_dataset(input_csv, workdir):
    rows = []
    seen_ids = set()
    with open(input_csv, "r", encoding="utf-8-sig") as fin:
        reader = csv.DictReader(fin)
        if "sequence" not in (reader.fieldnames or []):
            raise ValueError("Column `sequence` is required in `%s`" % input_csv)
        for idx, row in enumerate(reader):
            sequence = (row.get("sequence") or "").strip().upper()
            if not sequence:
                raise ValueError("Empty sequence at CSV row %d in `%s`" % (idx + 2, input_csv))
            row_id = (
                row.get("row_id")
                or row.get("id")
                or row.get("dfbp_id")
                or "row_%06d" % (idx + 1)
            ).strip()
            if not row_id:
                raise ValueError("Empty sample ID at CSV row %d in `%s`" % (idx + 2, input_csv))
            if row_id in seen_ids:
                raise ValueError("Duplicate sample ID `%s` in `%s`; prediction rows require unique IDs" % (row_id, input_csv))
            seen_ids.add(row_id)
            source = (row.get("category") or row.get("source") or "").strip()
            rows.append({
                "row_id": row_id,
                "sequence": sequence,
                "source": source,
                "raw": row,
            })

    if not rows:
        raise ValueError("No peptide rows found in `%s`" % input_csv)

    dataset_csv = os.path.join(workdir, "peptide_toxicity.csv")
    with open(dataset_csv, "w", encoding="utf-8", newline="") as fout:
        writer = csv.DictWriter(
            fout,
            fieldnames=["id", "sequence", "toxicity", "split", "source"],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow({
                "id": row["row_id"],
                "sequence": row["sequence"],
                "toxicity": 0,
                "split": "test",
                "source": row["source"],
            })

    return rows


def predict_dataset(solver, split):
    util.require_single_process_inference(world_size=solver.world_size)
    dataset_obj = getattr(solver, "%s_set" % split)
    sampler = torch_data.SequentialSampler(dataset_obj)
    dataloader = data.DataLoader(dataset_obj, solver.batch_size, sampler=sampler, num_workers=solver.num_worker)
    model = solver.model
    model.split = split
    model.eval()

    outputs = []
    with torch.no_grad():
        for batch in dataloader:
            if solver.device.type == "cuda":
                batch = utils.cuda(batch, device=solver.device)
            pred, _ = model.predict_and_target(batch)
            prob = pred.softmax(dim=-1)[:, 1].detach().cpu()
            label = pred.argmax(dim=-1).detach().cpu()
            ids = batch.get("id", [])
            sequences = batch.get("sequence", [])
            sources = batch.get("source", [])
            for idx in range(len(ids)):
                outputs.append({
                    "row_id": ids[idx],
                    "sequence": sequences[idx],
                    "source": sources[idx],
                    "toxicity_probability": float(prob[idx].item()),
                    "predicted_label": int(label[idx].item()),
                })
    return outputs


def index_predictions(input_rows, predictions):
    pred_by_id = {}
    for row in predictions:
        row_id = row["row_id"]
        if row_id in pred_by_id:
            raise ValueError("Duplicate prediction for sample ID `%s`" % row_id)
        pred_by_id[row_id] = row
    input_ids = {row["row_id"] for row in input_rows}
    if input_ids != set(pred_by_id):
        raise ValueError("Predicted IDs do not match input IDs; refusing to write incomplete predictions")
    return pred_by_id


def write_prediction_csv(input_rows, predictions, output_csv):
    pred_by_id = index_predictions(input_rows, predictions)
    base_fields = list(input_rows[0]["raw"].keys()) if input_rows else []
    fieldnames = list(dict.fromkeys(base_fields + ["toxicity_probability", "predicted_label"]))
    os.makedirs(os.path.dirname(os.path.abspath(output_csv)), exist_ok=True)

    with open(output_csv, "w", encoding="utf-8", newline="") as fout:
        writer = csv.DictWriter(fout, fieldnames=fieldnames)
        writer.writeheader()
        for row in input_rows:
            record = dict(row["raw"])
            pred = pred_by_id[row["row_id"]]
            record["toxicity_probability"] = "%.6f" % pred["toxicity_probability"]
            record["predicted_label"] = pred["predicted_label"]
            writer.writerow(record)


def summarize_predictions(input_rows, predictions):
    pred_by_id = index_predictions(input_rows, predictions)
    total_rows = len(input_rows)
    total_unique_sequences = len({row["sequence"] for row in input_rows})
    toxic_rows = 0
    toxic_sequences = set()
    for row in input_rows:
        pred = pred_by_id[row["row_id"]]
        if pred["predicted_label"] == 1:
            toxic_rows += 1
            toxic_sequences.add(row["sequence"])

    top_rows = sorted(predictions, key=lambda row: row["toxicity_probability"], reverse=True)[:20]

    return {
        "num_rows": total_rows,
        "num_unique_sequences": total_unique_sequences,
        "predicted_toxic_rows": toxic_rows,
        "predicted_non_toxic_rows": total_rows - toxic_rows,
        "predicted_toxic_unique_sequences": len(toxic_sequences),
        "top20_highest_probability": top_rows,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--input_csv", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--plm_embedding_file", default=None)
    parser.add_argument("--contact_edge_file", default=None)
    parser.add_argument("--structure_feature_file", default=None)
    parser.add_argument("--output_csv", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--gpus", default="null")
    args = parser.parse_args()

    gpus = None if args.gpus == "null" else ast.literal_eval(args.gpus)
    util.require_single_process_inference(gpus)

    util.get_root_logger(file=False)

    with tempfile.TemporaryDirectory(prefix="hipeptox_predict_") as tempdir:
        input_rows = build_temp_dataset(args.input_csv, tempdir)
        solver = load_solver(
            args.config,
            tempdir,
            gpus,
            args.plm_embedding_file,
            args.contact_edge_file,
            args.structure_feature_file,
        )
        solver.load(args.checkpoint, load_optimizer=False)
        predictions = predict_dataset(solver, "test")

    write_prediction_csv(input_rows, predictions, args.output_csv)
    summary = summarize_predictions(input_rows, predictions)

    result = {
        "config": args.config,
        "input_csv": args.input_csv,
        "checkpoint": args.checkpoint,
        "plm_embedding_file": args.plm_embedding_file,
        "contact_edge_file": args.contact_edge_file,
        "structure_feature_file": args.structure_feature_file,
        "output_csv": args.output_csv,
        "summary": summary,
    }

    os.makedirs(os.path.dirname(os.path.abspath(args.output_json)), exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as fout:
        json.dump(result, fout, ensure_ascii=False, indent=2)

    if comm.get_rank() == 0:
        print(json.dumps(result, ensure_ascii=False, indent=2))
        print("Wrote %s" % args.output_csv)
        print("Wrote %s" % args.output_json)


if __name__ == "__main__":
    main()
