import argparse
import copy
import logging
import os
import time
import uuid

import easydict
import jinja2
import torch
import yaml
from jinja2 import meta
from torch import distributed as dist
from torch.optim import lr_scheduler

from torchdrug import core, utils
from torchdrug.utils import comm

logger = logging.getLogger(__file__)


def get_root_logger(file=True):
    root_logger = logging.getLogger("")
    root_logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)-10s %(message)s", "%H:%M:%S")

    if file:
        handler = logging.FileHandler("log.txt")
        handler.setFormatter(formatter)
        root_logger.addHandler(handler)

    return root_logger


def create_working_directory(cfg):
    resolve_config_paths(cfg)
    world_size = comm.get_world_size()
    if world_size > 1 and not dist.is_initialized():
        comm.init_process_group("nccl", init_method="env://")

    run_name = time.strftime("%Y-%m-%d-%H-%M-%S") + "-" + uuid.uuid4().hex[:8]
    working_dir = os.path.join(
        os.path.expanduser(cfg.output_dir),
        cfg.task["class"],
        cfg.dataset["class"],
        cfg.task.model["class"],
        run_name,
    )

    # Different distributed jobs use different rendezvous ports. Single-process
    # runs do not need a shared temporary file at all.
    rendezvous = os.environ.get("MASTER_PORT", "default")
    file_name = os.path.join(cfg.output_dir, ".working_dir_%s.tmp" % rendezvous)

    if comm.get_rank() == 0:
        os.makedirs(working_dir, exist_ok=True)
        if world_size > 1:
            with open(file_name, "w", encoding="utf-8") as fout:
                fout.write(working_dir)
    comm.synchronize()
    if comm.get_rank() != 0:
        with open(file_name, "r", encoding="utf-8") as fin:
            working_dir = fin.read()
    comm.synchronize()
    if world_size > 1 and comm.get_rank() == 0 and os.path.exists(file_name):
        os.remove(file_name)

    os.chdir(working_dir)
    return working_dir


def detect_variables(cfg_file):
    with open(cfg_file, "r", encoding="utf-8") as fin:
        raw = fin.read()
    env = jinja2.Environment()
    ast = env.parse(raw)
    return meta.find_undeclared_variables(ast)


def _jinja_finalize(value):
    if value is None:
        return "null"
    if value is True:
        return "true"
    if value is False:
        return "false"
    return value


def load_config(cfg_file, context=None):
    with open(cfg_file, "r", encoding="utf-8") as fin:
        raw = fin.read()
    template = jinja2.Environment(
        finalize=_jinja_finalize, undefined=jinja2.StrictUndefined
    ).from_string(raw)
    instance = template.render(context or {})
    cfg = yaml.safe_load(instance)
    if not isinstance(cfg, dict):
        raise ValueError("Configuration `%s` must contain a YAML mapping" % cfg_file)
    cfg = easydict.EasyDict(cfg)
    resolve_config_paths(cfg)
    validate_feature_configuration(cfg)
    return cfg


def resolve_config_paths(cfg, base_dir=None):
    """Resolve input paths before a training run changes its working directory.

    Relative paths use the directory from which the command was launched, as in
    the README examples. Already absolute paths are preserved.
    """
    base_dir = os.path.abspath(base_dir or os.getcwd())
    for section, keys in (
        (cfg, ("output_dir", "checkpoint")),
        (cfg.get("dataset", {}), ("path", "plm_embedding_file", "structure_feature_file", "contact_edge_file")),
    ):
        for key in keys:
            value = section.get(key)
            if value is None:
                continue
            value = os.path.expanduser(os.fspath(value))
            section[key] = value if os.path.isabs(value) else os.path.join(base_dir, value)
    return cfg


def validate_feature_configuration(cfg):
    dataset = cfg.get("dataset", {})
    model = cfg.get("task", {}).get("model", {})
    if model.get("plm_dual_branch", False) and not dataset.get("plm_embedding_file"):
        raise ValueError("This model requires --plm_embedding_file; a null PLM input is not supported")
    if "structure_feature_file" in dataset and dataset.get("structure_feature_strict", True):
        if not dataset["structure_feature_file"]:
            raise ValueError("This configuration requires --structure_feature_file; use a non-spatial configuration to omit it")


def require_single_process_inference(gpus=None, world_size=None):
    """Reject distributed inference until ordered ID gathering is implemented."""
    if gpus is not None:
        if not isinstance(gpus, (list, tuple)) or not all(isinstance(gpu, int) for gpu in gpus):
            raise ValueError("--gpus must be null for CPU or a list containing one GPU index, e.g. '[0]'")
        if len(gpus) != 1:
            raise ValueError("Inference supports CPU or exactly one GPU; pass --gpus null or --gpus '[0]'")
    if world_size is None:
        world_size = comm.get_world_size()
    if world_size != 1:
        raise ValueError("Distributed inference is not supported; run this command as a single process")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", help="yaml configuration file", required=True)
    parser.add_argument("-s", "--seed", help="random seed for PyTorch", type=int, default=0)

    args, unparsed = parser.parse_known_args()
    dynamic_vars = detect_variables(args.config)

    parser = argparse.ArgumentParser()
    for var in sorted(dynamic_vars):
        parser.add_argument(f"--{var}", required=True)
    parsed = parser.parse_args(unparsed)
    dynamic_vars = {k: utils.literal_eval(v) for k, v in parsed._get_kwargs()}

    return args, dynamic_vars


def build_solver(cfg, dataset):
    if cfg.get("split", False):
        splits = dataset.split(**cfg.split)
    else:
        splits = dataset.split()
    train_set, valid_set, test_set = splits

    if comm.get_rank() == 0:
        logger.warning("#train: %d, #valid: %d, #test: %d", len(train_set), len(valid_set), len(test_set))

    cfg.task.task = dataset.target_fields
    task = core.Configurable.load_config_dict(cfg.task)

    cfg.optimizer.params = task.parameters()
    optimizer = core.Configurable.load_config_dict(cfg.optimizer)

    engine_cfg = copy.deepcopy(cfg.engine)
    if "scheduler" not in cfg:
        scheduler = None
    else:
        scheduler_cfg = copy.deepcopy(cfg.scheduler)
        if scheduler_cfg["class"] == "ReduceLROnPlateau":
            scheduler_cfg.pop("class")
            scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, **scheduler_cfg)
        elif hasattr(lr_scheduler, scheduler_cfg["class"]):
            scheduler_name = scheduler_cfg.pop("class")
            scheduler_cls = getattr(lr_scheduler, scheduler_name)
            scheduler = scheduler_cls(optimizer, **scheduler_cfg)
        else:
            scheduler_cfg.optimizer = optimizer
            scheduler = core.Configurable.load_config_dict(scheduler_cfg)
    engine_cfg.scheduler = scheduler

    solver = core.Engine(task, train_set, valid_set, test_set, optimizer, **engine_cfg)

    if cfg.get("checkpoint") is not None:
        solver.load(cfg.checkpoint, load_optimizer=False)

    return solver, scheduler
