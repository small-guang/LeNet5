# =============================================
# -*- coding: utf-8 -*-
# @Time    : 2021/1/26 下午2:53
# @Author  : xiaoguang
# @Email   : 1549380550@qq.com
# @File    : dataset.py
# @Software: PyCharm
# @Discript:
# ============================================
import numpy as np


class MNISTData:
    def __init__(self, data, need_shuffle, batch_size=128):
        self._data = data[0]
        self._labels = data[1]
        self.num_examples = self._data.shape[0]
        self._need_shuffle = need_shuffle
        self._indicator = 0
        self._batch_size = batch_size
        if self._need_shuffle:
            self._shuffle_data()

    def __iter__(self):
        return self

    def _shuffle_data(self):
        p = np.random.permutation(self.num_examples)
        self._data = self._data[p]
        self._labels = self._labels[p]

    def next_batch(self):
        end_indicator = self._indicator + self._batch_size
        if end_indicator > self.num_examples:
            if self._need_shuffle:
                self._shuffle_data()
                self._indicator = 0
                end_indicator = self._batch_size
            else:
                self._indicator = 0
                end_indicator = self._batch_size
        if end_indicator > self.num_examples:
            raise StopIteration
        batch_data = self._data[self._indicator: end_indicator] / 255.0
        batch_labels = self._labels[self._indicator: end_indicator]
        self._indicator = end_indicator

        return batch_data, batch_labels

    def __next__(self):
        return self.next_batch()



