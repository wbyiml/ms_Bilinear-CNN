# ms_task3

## 利用MindSpore实现细粒度分类网络双线性网络Bilinear CNN

数据集CUB_200_2011存放于 data/cub200目录下：data/cub200/CUB_200_2011


densenet121预训练模型（可从mindspore hub下载）：https://pan.baidu.com/s/1uAqXgnD75K4xmZq7KHsg_g

提取码：v3ui

训练epoch120结果：0.802

结果模型：https://pan.baidu.com/s/1m9uo-6po92kwmDFUK8fdvA

提取码：jid3


训练：
CUDA_VISIBLE_DEVICES=0 python train.py --dataset_path data/cub200 --device_target GPU --epoch 120

评估：
CUDA_VISIBLE_DEVICES=1 python eval.py --dataset_path data/cub200 --device_target GPU --ckpt_path outputs/bcnn-120_374.ckpt


