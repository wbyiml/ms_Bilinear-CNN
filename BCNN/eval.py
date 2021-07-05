# Copyright 2020 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""
#################train lstm example on aclImdb########################
"""
import argparse
import os

import numpy as np

from cub200 import create_dataset
from model import BCNN

from mindspore import Tensor, nn, Model, context
from mindspore.nn import Accuracy, Recall, F1
from mindspore.train.serialization import load_checkpoint, load_param_into_net

from mindspore.nn.metrics.metric import Metric

class RegAccuracy(Metric):
    def __init__(self):
        super(RegAccuracy, self).__init__()
        self.clear()

    def clear(self):
        """Clears the internal evaluation result."""
        self._correct_num = 0
        self._total_num = 0


    def update(self, *inputs):
        """
        Updates the internal evaluation result :math:`y_{pred}` and :math:`y`.

        Args:
            inputs: Input `y_pred` and `y`. `y_pred` and `y` are a `Tensor`, a list or an array.
                For the 'classification' evaluation type, `y_pred` is in most cases (not strictly) a list
                of floating numbers in range :math:`[0, 1]`
                and the shape is :math:`(N, C)`, where :math:`N` is the number of cases and :math:`C`
                is the number of categories. Shape of `y` can be :math:`(N, C)` with values 0 and 1 if one-hot
                encoding is used or the shape is :math:`(N,)` with integer values if index of category is used.
                For 'multilabel' evaluation type, `y_pred` and `y` can only be one-hot encoding with
                values 0 or 1. Indices with 1 indicate the positive category. The shape of `y_pred` and `y`
                are both :math:`(N, C)`.

        Raises:
            ValueError: If the number of the inputs is not 2.
        """


        if len(inputs) != 2:
            raise ValueError('Accuracy need 2 inputs (y_pred, y), but got {}'.format(len(inputs)))

        y_pred = self._convert_data(inputs[0])
        y = self._convert_data(inputs[1])

    


        indices = np.round(y_pred)

        result = (np.equal(indices, y) * 1).reshape(-1)


        self._correct_num += result.sum()
        self._total_num += result.shape[0]


    def eval(self):
        """
        Computes the accuracy.

        Returns:
            Float, the computed result.

        Raises:
            RuntimeError: If the sample size is 0.
        """
        if self._total_num == 0:
            raise RuntimeError('Accuary can not be calculated, because the number of samples is 0.')
        return self._correct_num / self._total_num


# CUDA_VISIBLE_DEVICES=0 python eval.py --dataset_path data/cub200 --device_target GPU --ckpt_path outputs/bcnn-2_502.ckpt

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', help='data/cub200')
    parser.add_argument('--batch_size', type=int, default=16) # 64
    parser.add_argument('--ckpt_path', type=str, default=None, help='the checkpoint file path used to evaluate model.')
    parser.add_argument('--device_target', type=str, default="CPU", choices=['GPU', 'CPU'])
    args = parser.parse_args()

    context.set_context(
        mode=context.PYNATIVE_MODE, # GRAPH_MODE
        save_graphs=False,
        device_target=args.device_target)

    

    dataset = create_dataset(args.dataset_path,batch_size=args.batch_size,is_train=False)

    network = BCNN(num_classes=200,phase='test')
    


    # loss = nn.BCELoss(reduction='mean')
    loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')

    # model = Model(network, loss, metrics={'acc': RegAccuracy()})
    model = Model(network, loss, metrics={'acc': Accuracy(), 'recall': Recall(), 'f1': F1()})


    print("============== Starting Testing ==============")
    param_dict = load_checkpoint(args.ckpt_path)
    load_param_into_net(network, param_dict)
    if args.device_target == "CPU":
        acc = model.eval(dataset, dataset_sink_mode=False)
    else:
        acc = model.eval(dataset)
    print("============== {} ==============".format(acc))
