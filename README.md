# LeNet5_tensorflow2
使用tensorflow2 搭建LeNet5网络，训练MNIST数据集，详细讲解已上传之CSDN [https://blog.csdn.net/qq_45723275/article/details/122667592?spm=1001.2014.3001.5501](https://blog.csdn.net/qq_45723275/article/details/122667592?spm=1001.2014.3001.5501) ，最后使用c++ opencv4.5.5完成模型部署。


## 综述
![245855274f404d02bdf94981c0158d20](README.assets/245855274f404d02bdf94981c0158d20.png)
LeNet5 这个网络包含了深度学习的基本模块：卷积层，池化层，全链接层。是其他深度学习模型的基础。LeNet-5共有7层，不包含输入，每层都包含可训练参数；每个层有多个Feature Map，每个FeatureMap通过一种卷积滤波器提取输入的一种特征，然后每个FeatureMap有多个神经元。本人将激活函数更换为Mish
## 环境配置
本人环境配置如下：
> pycharm2021
> 
> vs2022
> 
> Anaconda3
> 
> tensorflow=2.4

## 预测结果
![c7b4514569674b61b74878e479b0f2d4](README.assets/c7b4514569674b61b74878e479b0f2d4.png)


## 文件说明
- [x] dattset.py 加载数据集并划分训练集和测试集
- [x] export_frozen_graph.py 模型固化
- [x] network.py 网络结构
- [x] plt.py 查看数据集
- [x] train_model.py 模型训练
- [x] predict.py 模型预测    
