#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import time
import argparse

import numpy as np

import mindspore
from mindspore import Tensor, nn, Model, context
from mindspore.train.callback import LossMonitor, TimeMonitor, CheckpointConfig, ModelCheckpoint

from cub200 import create_dataset
from model import BCNN
from utils.lr_scheduler import CosineAnnealingLR

import mindspore.dataset.core.config as config  # ds.config 或 ds.core.config
config.set_num_parallel_workers(2)





# CUDA_VISIBLE_DEVICES=1 python train.py --dataset_path data/cub200 --device_target GPU
# TODO: loss softmax改为sigmoid(output sigmoid,dataset onehot)   数据增强可用cutmix MixUpBatch    focal loss
# TODO：backbone预训练
def main():
    """The main function."""
    parser = argparse.ArgumentParser(description='Train mean field bilinear CNN on CUB200.')
    parser.add_argument('--dataset_path', help='data/cub200')
    parser.add_argument('--batch_size', type=int, default=16) # 64 
    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--save_dir', default='outputs2')
    parser.add_argument('--device_target', type=str, default="CPU", choices=['GPU', 'CPU'])
    parser.add_argument('--pretrained', dest='pretrained', type=str, required=False, help='Pre-trained model.')
    args = parser.parse_args()

    context.set_context(
        # mode=context.GRAPH_MODE,
        mode=context.PYNATIVE_MODE,
        save_graphs=False,
        device_target=args.device_target)

    print("Loading dataset...")
    dataset = create_dataset(args.dataset_path,batch_size=args.batch_size,is_train=True)
    print("done")

    network = BCNN(num_classes=200,phase='train')

    print('created network')
    if args.pretrained is not None:
        print('loaded network')
        pass  


    # loss = nn.BCELoss(reduction='mean')
    loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')


    cfg = {
        'base_lr': 1e-4 , #  1e-4 
        'eta_min': 1e-5,
        'warmup_epochs': 1,
        'momentum':0.9,
        'weight_decay':1e-4
    }
    # lr_scheduler = CosineAnnealingLR(cfg['base_lr'], args.epoch, dataset.get_dataset_size(), args.epoch, warmup_epochs=cfg['warmup_epochs'], eta_min=cfg['eta_min'])
    # lr_schedule = lr_scheduler.get_lr()
    
    # 阶梯式学习率
    # def generate_steps_lr(lr_init, steps_per_epoch, total_epochs):
    #     total_steps = total_epochs * steps_per_epoch
    #     decay_epoch_index = [0.3*total_steps, 0.6*total_steps, 0.8*total_steps]
    #     lr_each_step = []
    #     for i in range(total_steps):
    #         if i < decay_epoch_index[0]:
    #             lr = lr_init
    #         elif i < decay_epoch_index[1]:
    #             lr = lr_init * 0.1
    #         elif i < decay_epoch_index[2]:
    #             lr = lr_init * 0.01
    #         else:
    #             lr = lr_init * 0.001
    #         lr_each_step.append(lr)
    #     return np.array(lr_each_step).astype(np.float32)
    # lr_schedule = generate_steps_lr(lr_init=0.001, steps_per_epoch=dataset.get_dataset_size(), total_epochs=args.epoch)

    # optimizer = nn.Momentum(network.trainable_params(), learning_rate=Tensor(lr_schedule), momentum=cfg['momentum'], weight_decay=cfg['weight_decay']) # 0.9 
    optimizer = nn.Momentum(network.trainable_params(), cfg['base_lr'], momentum=cfg['momentum'], weight_decay=cfg['weight_decay']) # 0.9 
    # optimizer = nn.Momentum(network.trainable_params(), cfg['base_lr'], momentum=cfg['momentum']) # 0.9 

    model = Model(network, loss, optimizer, {'acc': nn.Accuracy()})
    print("============== Starting Training ==============")
    config_ck = CheckpointConfig(save_checkpoint_steps=374*2, keep_checkpoint_max=5)
    ckpoint_cb = ModelCheckpoint(prefix="bcnn", directory=args.save_dir, config=config_ck)
    time_cb = TimeMonitor(data_size=dataset.get_dataset_size())
    loss_cb = LossMonitor()
    if args.device_target == "CPU":
        model.train(args.epoch, dataset, callbacks=[time_cb, loss_cb, ckpoint_cb], dataset_sink_mode=False)
    else:
        model.train(args.epoch, dataset, callbacks=[time_cb, loss_cb, ckpoint_cb])
    print("============== Training Success ==============")


if __name__ == '__main__':
    main()
