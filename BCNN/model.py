# -*- coding: utf-8 -*-
"""Mean field B-CNN model."""

import math
import random

import numpy as np

from mindspore import Tensor, nn, context, Parameter, ParameterTuple
from mindspore.common.initializer import initializer
from mindspore.ops import operations as P
import mindspore.ops as ops
import mindspore.ops.functional as F
import mindspore.common.dtype as mstype
import mindspore
from mindspore.train.serialization import load_param_into_net, load_checkpoint

from models.vgg import vgg16
from models.densenet import _densenet1,_densenet121
from models.xception import xception
from models.inceptionv4 import Inceptionv4




class BCNN(nn.Cell):
    """Mean field B-CNN model.

    The B-CNN model is illustrated as follows.
    conv1^2 (64) -> pool1 -> conv2^2 (128) -> pool2 -> conv3^3 (256) -> pool3
    -> conv4^3 (512) -> pool4 -> conv5^3 (512) -> mean field bilinear pooling
    -> fc.

    The network accepts a 3*448*448 input, and the relu5-3 activation has shape
    512*28*28 since we down-sample 4 times.

    Attributes:
        _is_all, bool: In the all/fc phase.
        features, torch.nn.Module: Convolution and pooling layers.
        bn, torch.nn.Module.
        gap_pool, torch.nn.Module.
        mf_relu, torch.nn.Module.
        mf_pool, torch.nn.Module.
        fc, torch.nn.Module.
    """
    def __init__(self, num_classes,phase='train'):
        super().__init__()

        # /home/nls1/.mscache/mindspore/ascend/1.1/预训练
        # args = {
        #     'batch_norm':True,  # True False
        #     'has_dropout':True,
        #     'initialize_mode':'KaimingNormal',  # XavierUniform KaimingNormal
        #     'padding':0,
        #     'pad_mode':'same',
        #     'has_bias':False
        # }
        # self.features = vgg16(num_classes=num_classes, args=args, phase=phase)
        # if phase=='train':
        #     ckpt = load_checkpoint('pretrained/vgg16_ascend_v111_cifar10_offical_cv_bs64_acc93.ckpt')
        #     load_param_into_net(self.features, ckpt)
        #     self.features.set_train(True)

        # features = xception()
        # if phase=='train':
        #     ckpt = load_checkpoint('pretrained/xception_ascend_v111_imagenet_offcial_cv_bs128_1top79.ckpt')
        #     load_param_into_net(features, ckpt)
        #     features.set_train(True)
        # conv = nn.Conv2d(2048, 512, kernel_size=3, stride=1, has_bias=False,padding=0, pad_mode="same")
        # norm = nn.BatchNorm2d(512)
        # relu = nn.ReLU()
        # self.features = nn.SequentialCell([features, conv, norm, relu])

        features = _densenet121()
        if phase=='train':
            ckpt = load_checkpoint('pretrained/densenet121_ascend_v111_imagenet2012_offical_cv_bs32_acc75.ckpt')
            load_param_into_net(features, ckpt)
            features.set_train(True)
        conv = nn.Conv2d(1024, 512, kernel_size=3, stride=1, has_bias=False,padding=0, pad_mode="same")
        norm = nn.BatchNorm2d(512)
        relu = nn.ReLU()
        self.features = nn.SequentialCell([features, conv, norm, relu])


        # Classification layer.
        self.fc = nn.Dense(512 * 512, num_classes)



    def construct(self, X):
        """Forward pass of the network.

        Args:
            X, torch.Tensor (bs,3,448,448).

        Returns:
            score, torch.Tensor (bs*200).
        """
        # Input. (bs, 3, 448, 448)
        
        X = self.features(X) # (bs, 512, 28, 28)


        # Classical bilinear pooling.
        shape = X.shape
        X = ops.reshape(X, (shape[0], shape[1], shape[2]*shape[3]))  # (bs, 512, 512)
        X = ops.BatchMatMul(False,True)(X, X) / (shape[2]*shape[3]) # bmm BatchMatMul
        X = ops.reshape(X, (shape[0], shape[1]*shape[1]))  # (bs, 512*512)

        # Normalization.
        # X = torch.sign(X) * torch.sqrt(torch.abs(X) + 1e-5)
        X = ops.Sqrt()(X + 1e-5)
        X = ops.L2Normalize()(X) # axis=0  x / L2

        # Classification.
        X = self.fc(X)
        # print(X.asnumpy()[0].argmax(),X[0,int(X.asnumpy()[0].argmax())])

        return X
