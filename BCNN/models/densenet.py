import math
from collections import OrderedDict

import mindspore.nn as nn
from mindspore.ops import operations as P
from mindspore.common import initializer as init


from models.var_init import default_recurisive_init, KaimingNormal



class GlobalAvgPooling(nn.Cell):
    """
    GlobalAvgPooling function.
    """
    def __init__(self):
        super(GlobalAvgPooling, self).__init__()
        self.mean = P.ReduceMean(True)
        self.shape = P.Shape()
        self.reshape = P.Reshape()

    def construct(self, x):
        x = self.mean(x, (2, 3))
        b, c, _, _ = self.shape(x)
        x = self.reshape(x, (b, c))
        return x

class CommonHead(nn.Cell):
    def __init__(self, num_classes, out_channels):
        super(CommonHead, self).__init__()
        self.avgpool = GlobalAvgPooling()
        self.fc = nn.Dense(out_channels, num_classes, has_bias=True)

    def construct(self, x):
        x = self.avgpool(x)
        x = self.fc(x)
        return x

def conv7x7(in_channels, out_channels, stride=1, padding=3, has_bias=False):
    return nn.Conv2d(in_channels, out_channels, kernel_size=7, stride=stride, has_bias=has_bias,
                     padding=padding, pad_mode="pad")


def conv3x3(in_channels, out_channels, stride=1, padding=1, has_bias=False):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, has_bias=has_bias,
                     padding=padding, pad_mode="pad")


def conv1x1(in_channels, out_channels, stride=1, padding=0, has_bias=False):
    return nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, has_bias=has_bias,
                     padding=padding, pad_mode="pad")


class _DenseLayer(nn.Cell):
    """
    the dense layer, include 2 conv layer
    """
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate):
        super(_DenseLayer, self).__init__()
        self.norm1 = nn.BatchNorm2d(num_input_features)
        self.relu1 = nn.ReLU()
        self.conv1 = conv1x1(num_input_features, bn_size*growth_rate)

        self.norm2 = nn.BatchNorm2d(bn_size*growth_rate)
        self.relu2 = nn.ReLU()
        self.conv2 = conv3x3(bn_size*growth_rate, growth_rate)

        # nn.Dropout in MindSpore use keep_prob, diff from Pytorch
        self.keep_prob = 1.0 - drop_rate
        self.dropout = nn.Dropout(keep_prob=self.keep_prob)

    def construct(self, features):
        bottleneck = self.conv1(self.relu1(self.norm1(features)))
        new_features = self.conv2(self.relu2(self.norm2(bottleneck)))
        if self.keep_prob < 1:
            new_features = self.dropout(new_features)
        return new_features

class _DenseBlock(nn.Cell):
    """
    the dense block
    """
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate):
        super(_DenseBlock, self).__init__()
        self.cell_list = nn.CellList()
        for i in range(num_layers):
            layer = _DenseLayer(
                num_input_features + i * growth_rate,
                growth_rate=growth_rate,
                bn_size=bn_size,
                drop_rate=drop_rate
            )
            self.cell_list.append(layer)

        self.concate = P.Concat(axis=1)

    def construct(self, init_features):
        features = init_features
        for layer in self.cell_list:
            new_features = layer(features)
            features = self.concate((features, new_features))
        return features

class _Transition(nn.Cell):
    """
    the transition layer
    """
    def __init__(self, num_input_features, num_output_features, avgpool=False):
        super(_Transition, self).__init__()
        if avgpool:
            poollayer = nn.AvgPool2d(kernel_size=2, stride=2)
        else:
            poollayer = nn.MaxPool2d(kernel_size=2, stride=2)
        self.features = nn.SequentialCell(OrderedDict([
            ('norm', nn.BatchNorm2d(num_input_features)),
            ('relu', nn.ReLU()),
            ('conv', conv1x1(num_input_features, num_output_features)),
            ('pool', poollayer)
        ]))

    def construct(self, x):
        x = self.features(x)
        return x

class Densenet(nn.Cell): # num_classes1000
    """
    the densenet architecture
    """
    __constants__ = ['features']

    def __init__(self, growth_rate, block_config, num_init_features=None, bn_size=4, drop_rate=0):
        super(Densenet, self).__init__()

        layers = OrderedDict()
        if num_init_features:
            layers['conv0'] = conv7x7(3, num_init_features, stride=2, padding=3)  # down 2
            layers['norm0'] = nn.BatchNorm2d(num_init_features)
            layers['relu0'] = nn.ReLU()
            layers['pool0'] = nn.MaxPool2d(kernel_size=3, stride=2, pad_mode='same')  # down 2  取消maxpool
            num_features = num_init_features
        else:
            layers['conv0'] = conv3x3(3, growth_rate*2, stride=1, padding=1)
            layers['norm0'] = nn.BatchNorm2d(growth_rate*2)
            layers['relu0'] = nn.ReLU()
            num_features = growth_rate * 2

        # Each denseblock
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(
                num_layers=num_layers,
                num_input_features=num_features,
                bn_size=bn_size,
                growth_rate=growth_rate,
                drop_rate=drop_rate
            )
            layers['denseblock%d'%(i+1)] = block
            num_features = num_features + num_layers*growth_rate

            if i != len(block_config)-1:   # down 2
                if num_init_features:
                    trans = _Transition(num_input_features=num_features, num_output_features=num_features // 2,  
                                        avgpool=False)
                else:
                    trans = _Transition(num_input_features=num_features, num_output_features=num_features // 2,  
                                        avgpool=True)
                layers['transition%d'%(i+1)] = trans
                num_features = num_features // 2

        # Final batch norm
        layers['norm5'] = nn.BatchNorm2d(num_features)
        layers['relu5'] = nn.ReLU()

        self.features = nn.SequentialCell(layers)
        self.out_channels = num_features

    def construct(self, x):
        x = self.features(x)
        return x

    def get_out_channels(self):
        return self.out_channels


def _densenet100(**kwargs):
    return Densenet(growth_rate=12, block_config=(16, 16, 16), **kwargs)


def _densenet121(**kwargs):
    return Densenet(growth_rate=32, block_config=(6, 12, 24, 16), num_init_features=64, **kwargs)


def _densenet161(**kwargs):
    return Densenet(growth_rate=48, block_config=(6, 12, 36, 24), num_init_features=96, **kwargs)


def _densenet169(**kwargs):
    return Densenet(growth_rate=32, block_config=(6, 12, 32, 32), num_init_features=64, **kwargs)


def _densenet201(**kwargs):
    return Densenet(growth_rate=32, block_config=(6, 12, 48, 32), num_init_features=64, **kwargs)


def _densenet1(**kwargs):
    return Densenet(growth_rate=32, block_config=(6, 12, 8), num_init_features=64, **kwargs)

├── BCNN
    ├── README.md                    // DenseNet相关说明
    ├── README_CN.md                 // DenseNet相关说明
    ├── ascend310_infer              // 实现310推理源代码
    ├── scripts
    │   ├── run_distribute_train.sh             // Ascend分布式shell脚本
    │   ├── run_distribute_train_gpu.sh             // GPU分布式shell脚本
    │   ├── run_distribute_eval.sh              // Ascend评估shell脚本
    │   ├── run_infer_310.sh                    // Ascend 310 推理shell脚本
    │   ├── run_distribute_eval_gpu.sh              // GPU评估shell脚本
    │   ├── run_eval_cpu.sh              // CPU训练shell脚本
    │   ├── run_train_cpu.sh              // CPU评估shell脚本
    ├── src
    │   ├── datasets             // 数据集处理函数
    │   ├── losses
    │       ├──crossentropy.py            // DenseNet损失函数
    ├── utils
        ├──lr_scheduler.py            // 学习率调度函数
    ├── train.py                    // 训练脚本
    ├── eval.py                     //  评估脚本
    ├── model.py            // BCNN模型架构
    ├── cub200.py             // 数据集处理函数
