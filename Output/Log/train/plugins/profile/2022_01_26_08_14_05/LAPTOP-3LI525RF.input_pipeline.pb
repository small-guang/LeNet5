	??? ??C@??? ??C@!??? ??C@	.;??J??.;??J??!.;??J??"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$??? ??C@R???Q??A????ƛC@Y/?$???*	?????4?@2~
GIterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap[0]::Generatora??+e??!˺C?GRX@)a??+e??1˺C?GRX@:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::Prefetch??0?*??!??q????)??0?*??1??q????:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelismn????!???=_@)?q??????17? 6H???:Preprocessing2F
Iterator::Model??ͪ?զ?!H3x??@)??_vOv?1&=??
%??:Preprocessing2p
9Iterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap?f??j+??!e>4?aX@)/n??r?1>4??2??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.2% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9/;??J??I@b?Z?X@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	R???Q??R???Q??!R???Q??      ??!       "      ??!       *      ??!       2	????ƛC@????ƛC@!????ƛC@:      ??!       B      ??!       J	/?$???/?$???!/?$???R      ??!       Z	/?$???/?$???!/?$???b      ??!       JCPU_ONLYY/;??J??b q@b?Z?X@