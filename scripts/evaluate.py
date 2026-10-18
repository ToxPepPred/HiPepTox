import argparse
import ast
import json
import os
import sys

from torchdrug import core
from torchdrug.utils import comm

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import model
from util import datasets, tasks
from util import common as util


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--dataset_path", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="test", choices=["train", "valid", "test"])
    parser.add_argument("--output_dir", default="./outputs_eval")
    parser.add_argument("--plm_embedding_file", default=None)
    parser.add_argument("--contact_edge_file", default=None)
    parser.add_argument("--structure_feature_file", default=None)
    parser.add_argument("--gpus", default="null")
    args = parser.parse_args()

    gpus = None if args.gpus == "null" else ast.literal_eval(args.gpus)
    util.require_single_process_inference(gpus)
    cfg = util.load_config(
        args.config,
        context={
            "output_dir": args.output_dir,
            "dataset_path": args.dataset_path,
            "gpus": gpus,
            "plm_embedding_file": args.plm_embedding_file,
            "contact_edge_file": args.contact_edge_file,
            "structure_feature_file": args.structure_feature_file,
        },
    )
    cfg.checkpoint = args.checkpoint
    util.resolve_config_paths(cfg)
    cfg.engine.gpus = gpus
    cfg.train.num_epoch = 0
    if cfg.dataset.get("augment", None):
        cfg.dataset.augment = "none"
    if cfg.dataset.get("augment_prob", None) is not None:
        cfg.dataset.augment_prob = 0.0

    util.get_root_logger(file=False)
    dataset = core.Configurable.load_config_dict(cfg.dataset)
    solver, _ = util.build_solver(cfg, dataset)
    metric = solver.evaluate(args.split)

    if comm.get_rank() == 0:
        serializable = {k: float(v.item()) if hasattr(v, "item") else v for k, v in metric.items()}
        print(json.dumps(serializable, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
