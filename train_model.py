# =============================================
# -*- coding: utf-8 -*-
# @Time    : 2021/1/26 下午4:12
# @Author  : xiaoguang
# @Email   : 1549380550@qq.com
# @File    : train_model.py
# @Software: PyCharm
# @Discript:
# ============================================
import os
import network
import dataset
import tensorflow as tf
import logging
from distutils.util import strtobool
from configparser import ConfigParser
from export_frozen_graph import export_frozen_graph


config = ConfigParser()
config.read("./config.cfg")

input_size = int(config.get("INPUT_SIZE", "input_size"))
learning_rate = float(config.get("MODEL", 'learning_rate'))
batch_size = int(config.get("MODEL", 'batch_size'))
epochs = int(config.get("MODEL", 'epochs'))

model_filepath = config.get("FILE_PATH", "model_filepath")
checkpoint_filepath = model_filepath + config.get("FILE_PATH", "checkpoint_filepath")
log_path = model_filepath + config.get("FILE_PATH", "log_path")
logfile = config.get("FILE_PATH", "logfile")
savefile = config.get("FILE_PATH", "savefile")

use_gpu = strtobool(config.get("GPU", "use_gpu"))

if not os.path.exists(os.path.dirname(logfile)):
    os.makedirs(os.path.dirname(logfile))

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(levelname)s %(message)s',
                    datefmt='%a, %d %b %Y %H:%M:%S', filename=logfile, filemode='w+')

log = tf.keras.callbacks.TensorBoard(log_dir=log_path)

cp_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,            # 文件路径
    save_best_only=True,                     # 保存最好的
    save_weights_only=True,                  # 只保存参数
    monitor='val_accuracy',                  # 需要监视的值
    mode='max',                              # 模式
    period=1,                                # CheckPoint之间的间隔的epoch数
    verbose=1)


mnist = tf.keras.datasets.mnist
train, test = mnist.load_data()

train_dataset = dataset.MNISTData(train, True)
test_dateset = dataset.MNISTData(test, False)

model = network.LeNet5(input_shape=[input_size, input_size, 1])
model.summary()

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),  # 优化器
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),             # 函数
    metrics=['accuracy']
)


tf.debugging.set_log_device_placement(True)
if use_gpu:
    gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(device=gpu, enable=True)
            tf.print(gpu)

    else:
        os.environ["CUDA_VISIBLE_DEVICE"] = "-1"
        logging.info("not found gpu device,convert to use cpu")

else:
    os.environ["CUDA_VISIBLE_DEVICE"] = "-1"


history = model.fit_generator(
    train_dataset,
    epochs=epochs,
    steps_per_epoch=train_dataset.num_examples // batch_size + 1,
    validation_data=test_dateset,
    validation_steps=test_dateset.num_examples // batch_size + 1,
    callbacks=[cp_callback, log]
)

model.load_weights(checkpoint_filepath)
model.save(model_filepath + savefile)
logging.info("model has been saved")
export_frozen_graph(model, model_filepath + 'frozen_graph.pb', (input_size, input_size, 1))
logging.info("model has been frozen")

