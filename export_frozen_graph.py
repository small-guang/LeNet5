# =============================================
# -*- coding: utf-8 -*-
# @Time    : 2021/1/26 下午4:59
# @Author  : xiaoguang
# @Email   : 1549380550@qq.com
# @File    : export_frozen_graph.py
# @Software: PyCharm
# @Discript:
# ============================================
import tensorflow as tf
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2

def export_frozen_graph(model, name, input_size) :
	f = tf.function(lambda x: model(x))
	f = f.get_concrete_function(x=tf.TensorSpec(shape=[None, input_size[0], input_size[1], input_size[2]], dtype=tf.float32))
	f2 = convert_variables_to_constants_v2(f)
	graph_def = f2.graph.as_graph_def()

	# Export frozen graph
	with tf.io.gfile.GFile(name, 'wb') as f:
		f.write(graph_def.SerializeToString())


