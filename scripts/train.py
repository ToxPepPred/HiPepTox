import math
import csv
import json
import os
import pprint
import random
import shutil
import sys

import numpy as np
import torch
import yaml
from torch.optim import lr_scheduler

from torchdrug import core
from torchdrug.utils import comm

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import model
from util import datasets, tasks
from util import common as util


def lower_better(metric_name, lower_metric_list=("root mean squared error", "mean absolute error")):
    return any(metric_name.startswith(cur_metric) for cur_metric in lower_metric_list)


def train_and_validate(cfg, solver, scheduler):
    if cfg.train.num_epoch == 0:
        return solver
    if not len(solver.valid_set):
        raise ValueError("Training requires a nonempty validation split for checkpoint selection")

    # 每个epoch都进行验证
    best_result = float("inf") if lower_better(cfg.metric) else float("-inf")
    best_epoch = -1

    for epoch in range(1, cfg.train.num_epoch + 1):
        # 训练一个epoch
        solver.train(num_epoch=1)
        
        # 保存当前epoch的模型
        solver.save("model_epoch_%d.pth" % solver.epoch)
        
        # 每个epoch都进行验证
        metric = solver.evaluate("valid")
        result = float(metric[cfg.metric])
        if not math.isfinite(result):
            raise ValueError("Validation selection metric `%s` is not finite at epoch %d" % (cfg.metric, solver.epoch))
        if comm.get_rank() == 0:
            values = {key: float(value) for key, value in metric.items()}
            history_exists = os.path.exists("validation_history.csv")
            with open("validation_history.csv", "a", encoding="utf-8", newline="") as fout:
                writer = csv.DictWriter(fout, fieldnames=["epoch"] + list(values))
                if not history_exists:
                    writer.writeheader()
                writer.writerow({"epoch": solver.epoch, **values})

        # 更新最佳模型
        if lower_better(cfg.metric):
            if result < best_result:
                best_result = result
                best_epoch = solver.epoch
        else:
            if result > best_result:
                best_result = result
                best_epoch = solver.epoch

        if isinstance(scheduler, lr_scheduler.ReduceLROnPlateau):
            scheduler.step(result)
        torch.cuda.empty_cache()

    if comm.get_rank() == 0:
        logger.warning("Best epoch: %d, Best %s: %f", best_epoch, cfg.metric, best_result)
        # 保留所有模型，不删除
        # 额外复制最佳模型
        shutil.copy("model_epoch_%d.pth" % best_epoch, "best_valid_epoch_%d.pth" % best_epoch)
        with open("best_checkpoint.json", "w", encoding="utf-8") as fout:
            json.dump({
                "checkpoint": "best_valid_epoch_%d.pth" % best_epoch,
                "selection_split": "validation",
                "selection_metric": cfg.metric,
                "selection_value": best_result,
                "epoch": best_epoch,
            }, fout, ensure_ascii=False, indent=2)
    comm.synchronize()
    torch.cuda.empty_cache()
    solver.load("best_valid_epoch_%d.pth" % best_epoch, load_optimizer=False)

    return solver


def test(solver):
    solver.evaluate("valid")
    if len(solver.test_set) == 0:
        if comm.get_rank() == 0:
            logger.warning("Skip test evaluation because the test split is empty.")
        return None
    return solver.evaluate("test")


if __name__ == "__main__":
    args, dynamic_vars = util.parse_args()
    args.config = os.path.abspath(os.path.expanduser(args.config))
    cfg = util.load_config(args.config, context=dynamic_vars)
    util.create_working_directory(cfg)

    seed = args.seed
    torch.manual_seed(seed + comm.get_rank())
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    logger = util.get_root_logger()
    if comm.get_rank() == 0:
        logger.warning("Config file: %s", args.config)
        logger.warning(pprint.pformat(cfg))

    if cfg.engine.get("gpus") and not torch.cuda.is_available():
        logger.warning("CUDA is unavailable on this machine. Falling back to CPU execution.")
        cfg.engine.gpus = None

    if comm.get_rank() == 0:
        with open("resolved_config.yaml", "w", encoding="utf-8") as fout:
            yaml.safe_dump(json.loads(json.dumps(cfg)), fout, sort_keys=False)
        with open("run_metadata.json", "w", encoding="utf-8") as fout:
            json.dump({
                "seed": seed,
                "config": args.config,
                "checkpoint_selection": "validation_only",
                "torch_version": torch.__version__,
                "gpus": cfg.engine.get("gpus"),
            }, fout, ensure_ascii=False, indent=2)

    dataset = core.Configurable.load_config_dict(cfg.dataset)
    solver, scheduler = util.build_solver(cfg, dataset)

    solver = train_and_validate(cfg, solver, scheduler)
    test_metric = test(solver)
    if comm.get_rank() == 0 and test_metric is not None:
        logger.warning("Test metric: %s", pprint.pformat(test_metric))
