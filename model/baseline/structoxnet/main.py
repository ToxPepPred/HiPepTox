# -*- coding: utf-8 -*-
# @Time    : 2024/8/16 9:00
# @Author  : Jiao Shihu
# @Email   : shihujiao@163.com
# @IDE     : PyCharm
# @FileName: main.py

import re
import os
import torch
import pickle
import pandas as pd
import numpy as np
from model import StrucToxNet
from configuration import config as cf
import torch.nn.functional as F
from preprocess import  data_processing

def load_text_file(fast_file):
    with open(fast_file) as f:
        lines = f.read()
        records = lines.split('>')[1:]
        seq_data = []
        for line in records:
            array = line.split('\n')
            sequence = re.sub('[^ACDEFGHIKLMNPQRSTUVWYX]','-',''.join(array[1:]).upper())
            seq_data.append(sequence)
        return seq_data

config = cf.get_train_config()

sequences_list = load_text_file('./Preprocessing/inputs/test_sequences.txt')

with open('./Preprocessing/outputs/combined_test.pkl', 'rb') as f:
    test_data = pickle.load(f)

test_loader = data_processing.create_data_loaders(test_data,config)




def model_eval(data_iter, model, config):
    label_pred = torch.empty([0], device=config.device)
    label_true = torch.empty([0], device=config.device)
    pred_prob = torch.empty([0], device=config.device)
    step = 0
    model.eval()
    with torch.no_grad():
        for batch in data_iter:
            batch = batch.to(config.device)
            labels = batch.label.to(config.device)
            outputs = model.get_logits(batch)

            pred_prob_all = F.softmax(outputs, dim=1)
            pred_prob_positive = pred_prob_all[:, 1]
            pred_prob_sort = torch.max(outputs, 1)
            pred_class = pred_prob_sort[1]

            label_pred = torch.cat([label_pred, pred_class])
            label_true = torch.cat([label_true, labels])
            pred_prob = torch.cat([pred_prob, pred_prob_positive])

            step += 1

    return label_pred.cpu().numpy(), pred_prob.cpu().numpy()


model = StrucToxNet.EGNNModel(config).to(config.device)
model.load_state_dict(torch.load('model/final_model.pkl', map_location=torch.device('cpu')))

y_pred, y_pred_prob = model_eval(test_loader, model, config)

for i, seq in enumerate(sequences_list):
    sequences_list[i] = ''.join(seq)

results = pd.DataFrame(np.zeros([len(y_pred), 4]), columns=["Seq_ID", "Sequences", "Prediction", "Confidence"])
for i in range(len(y_pred)):
    if y_pred[i] == 1:
        y_prob = str(round(y_pred_prob[i] * 100, 2)) + "%"
        results.iloc[i, :] = [round(i + 1), sequences_list[i], "Toxic peptide", y_prob]
    else:
        y_prob = str(round((1-y_pred_prob[i]) * 100, 2)) + "%"
        results.iloc[i, :] = [round(i + 1), sequences_list[i], "Non toxic peptide", y_prob]
os.chdir("result")
results.to_csv("Prediction_results", index=False)
print("job finished!")


