# =============================================
# -*- coding: utf-8 -*-
# @Time    : 2021/1/26 下午3:12
# @Author  : xiaoguang
# @Email   : 1549380550@qq.com
# @File    : network.py
# @Software: PyCharm
# @Discript:
# ============================================
from tensorflow.keras.models import Model
from tensorflow.keras.layers import *
import tensorflow as tf

from tensorflow.keras.layers import Activation
from tensorflow.keras.utils import get_custom_objects


class Mish(Activation):
    def __init__(self, activate, **kwargs):
        super(Mish, self).__init__(activate, **kwargs)
        self.__name__ = "Mish"

def mish(inputs):
    return inputs * tf.math.tanh(tf.math.softplus(inputs))


def LeNet5(input_shape):
    get_custom_objects().update({'Mish': Mish(mish)})

    inputs = Input(shape=input_shape)

    conv1 = Conv2D(6, 5, padding='same')(inputs)
    pool1 = MaxPooling2D((2, 2))(conv1)
    conv2 = Conv2D(16, 5, activation="Mish", padding='same')(pool1)
    pool2 = MaxPooling2D((2, 2))(conv2)
    conv2 = Conv2D(120, 5, activation="Mish", padding='same')(pool2)
    fc = Flatten()(conv2)

    fc1 = Dense(120, activation="relu")(fc)
    fc2 = Dense(10, activation="softmax")(fc1)

    model = Model(inputs, fc2)

    return model
