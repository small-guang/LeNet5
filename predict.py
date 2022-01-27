# =============================================
# -*- coding: utf-8 -*-
# @Time    : 2021/1/26 下午5:23
# @Author  : xiaoguang
# @Email   : 1549380550@qq.com
# @File    : predict.py
# @Software: PyCharm
# @Discript:
# ============================================
import matplotlib.pyplot as plt
import tensorflow as tf
import os
import numpy as np


model_filepath = os.path.join(os.path.dirname(__file__),"output/LeNet5")
print(model_filepath)

mnist = tf.keras.datasets.mnist
trian, test = mnist.load_data()

model = tf.keras.models.load_model(model_filepath)

img = test[0][899]
img = (np.expand_dims(img,0))
print(np.argmax(model.predict(img)), test[1][899])

plt.grid(False)
plt.imshow(test[0][899], cmap=plt.cm.binary)


if np.argmax(model.predict(img))==test[1][899]:
    color = 'blue'
else:
    color = 'red'
plt.xlabel("{}".format(np.argmax(model.predict(img))), color=color)
plt.show()

