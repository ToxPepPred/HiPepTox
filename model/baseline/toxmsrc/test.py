from keras.utils import to_categorical as labelEncoding
from sklearn.metrics import average_precision_score, f1_score, recall_score
from sklearn.metrics import (confusion_matrix, matthews_corrcoef, roc_curve, auc, accuracy_score)
from model import ourmodel
import numpy as np


my_seed = 42
np.random.seed(my_seed)
import random

random.seed(my_seed)
import tensorflow as tf

tf.random.set_seed(my_seed)
import pandas as pd
import numpy as np

# 加载数据
data1 = np.load('data/X2.npz')
X2 = data1['x_train']
y_2 = pd.read_csv('data/y_train.csv').to_numpy()
y2 = labelEncoding(y_2, dtype=int)


model = ourmodel()
model.load_weights('ToxMSRC.h5')
y_tes = y_2
y_p = model.predict([X2])
y_pred1 = y_p.argmax(axis=1)
acc = accuracy_score(y_tes, y_pred1)
sn = recall_score(y_tes, y_pred1)
mcc = matthews_corrcoef(y_tes, y_pred1)
tn, fp, fn, tp = confusion_matrix(y_tes, y_pred1).ravel()
sp = tn / (tn + fp)
fpr, tpr, _ = roc_curve(y_tes, y_p[:, 1])
rocauc = auc(fpr, tpr)
aupr = average_precision_score(y_tes, y_pred1)
f1 = f1_score(y_tes, np.round(y_pred1.reshape(-1)))
print("ACC : ", acc)
print("SN : ", sn)
print("SP : ", sp)
print("MCC : ", mcc)
print("AUC : ", rocauc)

