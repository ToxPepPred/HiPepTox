# -*- coding: utf-8 -*-
# @Time    : 2024/8/16 9:00
# @Author  : Jiao Shihu
# @Email   : shihujiao@163.com
# @IDE     : PyCharm
# @FileName: util_metric.py



from sklearn.metrics import roc_curve, precision_recall_curve, average_precision_score
from sklearn.metrics import auc
import numpy as np
import torch


def caculate_metric(pred_y, labels, pred_prob):
#     print('labels', labels) # [n_sample, num_class]
#     print('pred_y', pred_y) # [n_sample, num_class]
#     print('pred_prob', pred_prob) # [n_sample, num_class]

    test_num = len(labels)
    tp = 0
    fp = 0
    tn = 0
    fn = 0

    for index in range(test_num):
        if labels[index] == 1:
            if labels[index] == pred_y[index]:
                tp = tp + 1
            else:
                fn = fn + 1
        else:
            if labels[index] == pred_y[index]:
                tn = tn + 1
            else:
                fp = fp + 1

    # print('tp\tfp\ttn\tfn')
    # print('{}\t{}\t{}\t{}'.format(tp, fp, tn, fn))

    ACC = float(tp + tn) / test_num

    # precision
    if tp + fp == 0:
        Precision = 0
    else:
        Precision = float(tp) / (tp + fp)

    # SE
    if tp + fn == 0:
        Recall = Sensitivity = 0
    else:
        Recall = Sensitivity = float(tp) / (tp + fn)

    # SP
    if tn + fp == 0:
        Specificity = 0
    else:
        Specificity = float(tn) / (tn + fp)

    # MCC
    if (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn) == 0:
        MCC = 0
    else:
        MCC = float(tp * tn - fp * fn) / (np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))

    # F1-score
    if Recall + Precision == 0:
        F1 = 0
    else:
        F1 = 2 * Recall * Precision / (Recall + Precision)

    labels = labels.cpu()
    pred_prob = pred_prob.cpu()
    labels = labels.numpy().tolist()
    pred_prob = pred_prob.numpy().tolist()

    # ROC and AUC
    fpr, tpr, thresholds = roc_curve(labels, pred_prob, pos_label=1)  # 默认1就是阳性
    AUC = auc(fpr, tpr)
# Balanced Accuracy
    BACC = (Sensitivity + Specificity) / 2


    # PRC and AP
    precision, recall, thresholds = precision_recall_curve(labels, pred_prob, pos_label=1)
    AP = average_precision_score(labels, pred_prob, average='macro', pos_label=1, sample_weight=None)

    metric1 = torch.tensor([BACC*100, AUC, Sensitivity*100, Specificity*100, MCC, F1, Precision])

    metric2 = np.around(metric1,decimals=3)

    # ROC(fpr, tpr, AUC)
    # PRC(recall, precision, AP)
    roc_data = [fpr, tpr, AUC]
    prc_data = [recall, precision, AP]
    return metric2, roc_data, prc_data



