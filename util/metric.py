import torch


def macro_f1(pred, target, eps=1e-10):
    pred_label = pred.argmax(dim=-1)
    num_class = pred.size(-1)
    scores = []

    for class_id in range(num_class):
        tp = ((pred_label == class_id) & (target == class_id)).sum().float()
        fp = ((pred_label == class_id) & (target != class_id)).sum().float()
        fn = ((pred_label != class_id) & (target == class_id)).sum().float()

        precision = tp / (tp + fp + eps)
        recall = tp / (tp + fn + eps)
        f1 = 2 * precision * recall / (precision + recall + eps)
        scores.append(f1)

    return torch.stack(scores).mean()
