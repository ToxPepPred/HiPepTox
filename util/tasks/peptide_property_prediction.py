from collections.abc import Sequence

import torch
from torch.nn import functional as F

from torchdrug import layers, metrics, tasks
from torchdrug.core import Registry as R
from torchdrug.layers import functional

from util.metric import macro_f1


@R.register("tasks.PeptidePropertyPrediction")
class PeptidePropertyPrediction(tasks.PropertyPrediction):

    def __init__(self, *args, mixup_mode="none", mixup_alpha=0.4, mixup_prob=1.0,
                 mlp_hidden_dims=None, **kwargs):
        super().__init__(*args, **kwargs)
        mixup_mode = (mixup_mode or "none").strip().lower()
        if mixup_mode in {"graph_embedding", "hidden_feature"}:
            mixup_mode = "graph_feature"
        if mixup_mode not in {"none", "graph_feature"}:
            raise ValueError("Unknown mixup mode `%s`" % mixup_mode)
        self.mixup_mode = mixup_mode
        self.mixup_alpha = float(mixup_alpha)
        self.mixup_prob = float(mixup_prob)
        if mlp_hidden_dims is None:
            self.mlp_hidden_dims = None
        elif isinstance(mlp_hidden_dims, Sequence):
            self.mlp_hidden_dims = [int(dim) for dim in mlp_hidden_dims]
        else:
            self.mlp_hidden_dims = [int(mlp_hidden_dims)]

    def preprocess(self, train_set, valid_set, test_set):
        super().preprocess(train_set, valid_set, test_set)
        if self.mlp_hidden_dims:
            self.mlp = layers.MLP(
                self.model.output_dim,
                self.mlp_hidden_dims + [sum(self.num_class)],
                batch_norm=self.mlp_batch_norm,
                dropout=self.mlp_dropout,
            )

    def _mixup_enabled(self, graph_feature):
        return (
            self.training
            and getattr(self, "split", None) == "train"
            and self.mixup_mode != "none"
            and self.mixup_alpha > 0
            and self.mixup_prob > 0
            and graph_feature.shape[0] > 1
            and torch.rand((), device=graph_feature.device) < self.mixup_prob
        )

    def _sample_mixup_lambda(self, device, dtype):
        alpha = torch.tensor(self.mixup_alpha, device=device, dtype=dtype)
        return torch.distributions.Beta(alpha, alpha).sample()

    def _apply_graph_feature_mixup(self, graph_feature, target, labeled):
        perm = torch.randperm(graph_feature.shape[0], device=graph_feature.device)
        lam = self._sample_mixup_lambda(graph_feature.device, graph_feature.dtype)
        mixed_feature = lam * graph_feature + (1 - lam) * graph_feature[perm]
        return mixed_feature, target[perm], labeled[perm], lam

    def _criterion_loss(self, pred, target, criterion):
        if criterion == "mse":
            if self.normalization:
                return F.mse_loss((pred - self.mean) / self.std, (target - self.mean) / self.std, reduction="none")
            return F.mse_loss(pred, target, reduction="none")
        if criterion == "bce":
            return F.binary_cross_entropy_with_logits(pred, target, reduction="none")
        if criterion == "ce":
            return F.cross_entropy(pred, target.long().squeeze(-1), reduction="none").unsqueeze(-1)
        raise ValueError("Unknown criterion `%s`" % criterion)

    def _predict_from_graph_feature(self, graph_feature):
        pred = self.mlp(graph_feature)
        if self.normalization:
            pred = pred * self.std + self.mean
        return pred

    def forward(self, batch):
        all_loss = torch.tensor(0, dtype=torch.float32, device=self.device)
        metric = {}

        graph = batch["graph"]
        if self.graph_construction_model:
            graph = self.graph_construction_model(graph)
        output = self.model(graph, graph.node_feature.float(), all_loss=all_loss, metric=metric)
        graph_feature = output["graph_feature"]

        if all([t not in batch for t in self.task]):
            return all_loss, metric

        target = self.target(batch)
        labeled = ~torch.isnan(target)
        target[~labeled] = 0

        mix_target = None
        mix_labeled = None
        mix_lambda = None
        if self._mixup_enabled(graph_feature):
            graph_feature, mix_target, mix_labeled, mix_lambda = self._apply_graph_feature_mixup(
                graph_feature, target, labeled
            )

        pred = self._predict_from_graph_feature(graph_feature)

        for criterion, weight in self.criterion.items():
            if mix_target is None:
                loss = self._criterion_loss(pred, target, criterion)
                effective_labeled = labeled
            else:
                loss_a = self._criterion_loss(pred, target, criterion)
                loss_b = self._criterion_loss(pred, mix_target, criterion)
                loss = mix_lambda * loss_a + (1 - mix_lambda) * loss_b
                effective_labeled = labeled & mix_labeled
            loss = functional.masked_mean(loss, effective_labeled, dim=0)

            name = tasks._get_criterion_name(criterion)
            if self.verbose > 0:
                for t, l in zip(self.task, loss):
                    metric["%s [%s]" % (name, t)] = l
            loss = (loss * self.weight).sum() / self.weight.sum()
            metric[name] = loss
            all_loss += loss * weight

        return all_loss, metric

    def _class_sizes(self):
        if isinstance(self.num_class, Sequence):
            return list(self.num_class)
        return [self.num_class]

    def _evaluate(self, pred, target):
        labeled = ~torch.isnan(target)
        metric = {}
        class_sizes = self._class_sizes()

        for metric_name in self.metric:
            scores = []
            class_offset = 0

            for task_index, class_size in enumerate(class_sizes):
                task_pred = pred[:, class_offset:class_offset + class_size]
                task_target = target[:, task_index]
                task_labeled = labeled[:, task_index]

                if metric_name == "acc":
                    score = metrics.accuracy(task_pred[task_labeled], task_target[task_labeled].long())
                elif metric_name == "mcc":
                    score = metrics.matthews_corrcoef(task_pred[task_labeled], task_target[task_labeled].long())
                elif metric_name == "macrof1":
                    score = macro_f1(task_pred[task_labeled], task_target[task_labeled].long())
                elif metric_name == "auroc":
                    if class_size != 2:
                        raise ValueError("`auroc` is only implemented for binary classification in this scaffold")
                    positive_score = task_pred[task_labeled].softmax(dim=-1)[:, 1]
                    score = metrics.area_under_roc(positive_score, task_target[task_labeled].long())
                elif metric_name == "auprc":
                    if class_size != 2:
                        raise ValueError("`auprc` is only implemented for binary classification in this scaffold")
                    positive_score = task_pred[task_labeled].softmax(dim=-1)[:, 1]
                    score = metrics.area_under_prc(positive_score, task_target[task_labeled].long())
                else:
                    raise ValueError("Unknown metric `%s`" % metric_name)

                scores.append(score)
                class_offset += class_size

            score_tensor = torch.stack(scores)
            formatted_name = tasks._get_metric_name(metric_name)
            for task_name, score in zip(self.task, score_tensor):
                metric["%s [%s]" % (formatted_name, task_name)] = score

        return metric

    def evaluate(self, pred, target):
        metric = self._evaluate(pred, target)
        for metric_name in self.metric:
            formatted_name = tasks._get_metric_name(metric_name)
            metric["%s [average]" % formatted_name] = sum(
                metric["%s [%s]" % (formatted_name, task_name)] for task_name in self.task
            ) / len(self.task)
        return metric
