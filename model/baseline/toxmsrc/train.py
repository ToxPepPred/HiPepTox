# -*- coding: utf-8 -*-
from keras.optimizers import Adam
import pandas as pd
import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.utils import to_categorical as labelEncoding

from sklearn.metrics import (confusion_matrix,  matthews_corrcoef, precision_score, roc_curve, auc)
from sklearn.model_selection import KFold
from model import ourmodel
import numpy as np
from smote import datapro

my_seed = 42
np.random.seed(my_seed)
import random

random.seed(my_seed)
import tensorflow as tf

tf.random.set_seed(my_seed)

model = ourmodel()
model.summary()


data1 = np.load('word2vec_model/example_X.npz')
X1 = data1['x_train']
X2 = data1['x_test']

y_1 = pd.read_csv('word2vec_model/y_train.csv').to_numpy()
y_2 = pd.read_csv('word2vec_model/y_test.csv').to_numpy()
y2 = labelEncoding(y_2, dtype=int)

X1, y_1 = datapro(X1, y_1)
y1 = labelEncoding(y_1, dtype=int)

if __name__ == '__main__':
    config = tf.compat.v1.ConfigProto()

    config.gpu_options.allow_growth = True

    session = tf.compat.v1.Session(config=config)
    # set epoch and batch_size
    setEpochNumber = 100
    setBatchSizeNumber = 32
    ####################################################

    cv = KFold(n_splits=8, shuffle=True, random_state=42)

    Accuracy = []
    Sensitivity = []
    Specificity = []
    Precision = []
    MCC = []

    # ROC Curve:
    fig1 = plt.figure(figsize=[12, 12])

    TPR = []
    meanFPR = np.linspace(0, 1, 100)

    i = 1

    names = ['first']
    name = names[0]
    nn = 1

    for train, test in cv.split(y1):

        # Compile Model:
        model = ourmodel()
        model.compile(optimizer=Adam(learning_rate=0.0001),
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])

        filepath = 'ToxMSRC/%sModel%d.tf' % (name, nn)

        checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True,
                                     save_weights_only=True, mode='max')
        callbacks_list = [checkpoint]
        back = EarlyStopping(monitor='val_accuracy', patience=20, verbose=1, mode='auto')
        # Run Model:
        results = model.fit(x=[X1[train, :]],
                            y=y1[train, :],
                            validation_data=([X1[test, :]], y1[test, :]),
                            batch_size=setBatchSizeNumber, epochs=setEpochNumber,  # batch_size???
                            verbose=1,
                            callbacks=[callbacks_list, back])
        model.save('ToxMSRC/%sModel%d.h5' % (name, nn))

        nn += 1
        accuracy = model.evaluate(x=[X2], y=y2)
        Accuracy.append(accuracy[1])

        # Performance Metices:
        Yactual = y_2
        Yp = model.predict([X2])
        v = Yp
        Yp = Yp.argmax(axis=1)

        CM = confusion_matrix(y_pred=Yp, y_true=Yactual)
        TN, FP, FN, TP = CM.ravel()

        MCC.append(matthews_corrcoef(y_true=Yactual, y_pred=Yp))
        Sensitivity.append(TP / (TP + FN))
        Specificity.append(TN / (TN + FP))
        Precision.append(precision_score(y_true=Yactual, y_pred=Yp))

        # ROC Curve
        fpr, tpr, _ = roc_curve(Yactual, v[:, 1])
        TPR.append(np.interp(meanFPR, fpr, tpr))
        rocauc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, alpha=0.3, label='ROC fold %d (AUC = %0.4f)' % (i, rocauc))
        i = i + 1
        # plt.legend()
        # plt.show()

        print('AUC:', rocauc)
        print('Accuracy:', Accuracy)
        print('Sensitivity: ', Sensitivity)
        print('Specificity: ', Specificity)
        print('MCC:', MCC)
        print('Precision: ', Precision)

    # end-for
