from keras.layers import Dropout, Bidirectional, LSTM
from tensorflow.keras.layers import *
from tensorflow.keras.models import *

import numpy as np
from keras.layers import (Add, BatchNormalization, Dense, Input, Convolution1D, Flatten, MaxPooling1D)
from keras.models import Model

my_seed = 42
np.random.seed(my_seed)
import random

random.seed(my_seed)
import tensorflow as tf

tf.random.set_seed(my_seed)


def ourmodel():
    in_put = Input(shape=(49, 96))
    x1 = Convolution1D(128, 3, activation='relu', padding='same')(in_put)
    x1 = BatchNormalization()(x1)
    x1 = Dropout(0.2)(x1)

    x2 = Convolution1D(128, 5, activation='relu', padding='same')(in_put)
    x2 = BatchNormalization()(x2)
    x2 = Dropout(0.2)(x2)

    x3 = Convolution1D(128, 7, activation='relu', padding='same')(in_put)
    x3 = BatchNormalization()(x3)
    x3 = Dropout(0.2)(x3)

    x = Add()([x1, x2, x3])

    x = MaxPooling1D(pool_size=3, strides=1, padding='valid')(x)
    x = Dropout(0.4)(x)

    x = Convolution1D(256, 3, activation='relu', padding='valid')(x)

    x1 = Bidirectional(LSTM(128, return_sequences=True))(x)

    x = Add()([x1, x])

    x = Flatten()(x)
    x = Dense(128, activation='relu', name='FC3')(x)
    x = Dense(64, activation='relu', name='FC2')(x)
    x = Dropout(rate=0.6)(x)
    x = Dense(32, activation='relu', name='FC4')(x)

    output = Dense(2, activation='softmax', name='Output')(x)

    return Model(inputs=[in_put], outputs=[output])
