# =============================================
# -*- coding: utf-8 -*-
# @Time    : 2021/1/26 下午1:32
# @Author  : xiaoguang
# @Email   : 1549380550@qq.com
# @File    : plt.py
# @Software: PyCharm
# @Discript:
# ============================================
import matplotlib.pyplot as plt
import dataset
import tensorflow as tf


def m(train_images, train_labels):
    plt.figure(figsize=(10,10))
    for i in range(25):
        plt.subplot(5,5,i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(train_images[i], cmap=plt.cm.binary)
        plt.xlabel(train_labels[i])
    plt.show()


mnist = tf.keras.datasets.mnist
train, test = mnist.load_data()
for data in dataset.MNISTData(train, True):
    m(*data)
