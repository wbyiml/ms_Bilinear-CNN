# -*- coding: utf-8 -*
"""This module is served as torchvision.datasets to load CUB200-2011.

CUB200-2011 dataset has 11,788 images of 200 bird species. The project page
is as follows.
    http://www.vision.caltech.edu/visipedia/CUB-200-2011.html
- Images are contained in the directory data/cub200/raw/images/,
  with 200 subdirectories.
- Format of images.txt: <image_id> <image_name>
- Format of train_test_split.txt: <image_id> <is_training_image>
- Format of classes.txt: <class_id> <class_name>
- Format of iamge_class_labels.txt: <image_id> <class_id>

This file is modified from:
    https://github.com/vishwakftw/vision.
"""


import os
import pickle

import numpy as np
import PIL.Image
import mindspore.dataset as ds
import mindspore.dataset.vision.c_transforms as VT
from mindspore.dataset.vision import Inter

class CUB200:
    def __init__(self, root, is_train=True):
        self._root = os.path.expanduser(root)  # Replace ~ by the complete dir  
        self.is_train = is_train


        if os.path.isfile(os.path.join(self._root, 'processed/train.pkl')) and os.path.isfile(os.path.join(self._root, 'processed/test.pkl')):
            print('datasets already cached.')
        else:
            self.__build_cache()


        # Now load the picked data.
        if self.is_train:
            self.datas, self.labels = pickle.load(open(os.path.join(self._root, 'processed/train.pkl'), 'rb'))
            assert (len(self.datas) == 5994 and len(self.labels) == 5994)
        else:
            self.datas, self.labels = pickle.load(open(os.path.join(self._root, 'processed/test.pkl'), 'rb'))
            assert (len(self.datas) == 5794 and len(self.labels) == 5794)

    def __getitem__(self, index):
        """
        Args:
            index, int: Index.

        Returns:
            image, PIL.Image: Image of the given index.
            target, str: target of the given index.
        """
        image, target = self.datas[index], self.labels[index]

        # Doing this so that it is consistent with all other datasets.
        image = PIL.Image.fromarray(image)

        return image, target

    def __len__(self):
        """Length of the dataset.

        Returns:
            length, int: Length of the dataset.
        """
        return len(self.labels)



    def __build_cache(self):
        """Prepare the data for train/test split and save onto disk."""
        image_path = os.path.join(self._root, 'CUB_200_2011/images/')
        # Format of images.txt: <image_id> <image_name>
        id2name = np.genfromtxt(os.path.join(self._root, 'CUB_200_2011/images.txt'), dtype=str)
        # Format of train_test_split.txt: <image_id> <is_training_image>
        id2train = np.genfromtxt(os.path.join(self._root, 'CUB_200_2011/train_test_split.txt'), dtype=int)

        train_data = []
        train_labels = []
        test_data = []
        test_labels = []
        for id_ in range(id2name.shape[0]):
            image = PIL.Image.open(os.path.join(image_path, id2name[id_, 1]))
            label = int(id2name[id_, 1][:3]) - 1  # Label starts with 0

            # Convert gray scale image to RGB image.
            if image.getbands()[0] == 'L':
                image = image.convert('RGB')
            image_np = np.array(image) # ,dtype='uint8'
            image.close()

            if id2train[id_, 1] == 1:
                train_data.append(image_np)
                train_labels.append(label)
            else:
                test_data.append(image_np)
                test_labels.append(label)

        os.makedirs(os.path.join(self._root, 'processed'), exist_ok=True)
        pickle.dump((train_data, train_labels),
                    open(os.path.join(self._root, 'processed/train.pkl'), 'wb'))
        pickle.dump((test_data, test_labels),
                    open(os.path.join(self._root, 'processed/test.pkl'), 'wb'))


def create_dataset(root,batch_size,is_train=True):
    ds.config.set_seed(1)

    dataset_generator = CUB200(root, is_train=is_train)
    
    dataset = ds.GeneratorDataset(dataset_generator, ["image", "label"], shuffle=True)

    resize_crop_op = VT.RandomResizedCrop(size=256, scale=(0.8, 1.0), interpolation=Inter.BICUBIC)  # 448  256 缩放+裁剪
    h_flip_op = VT.RandomHorizontalFlip(0.2)  # 水平翻转
    cutout_op = VT.CutOut(20, num_patches=5)  # 涂黑
    random_affine_op = VT.RandomAffine(degrees=10, translate=(-0.1, 0.1, -0.1, 0.1), scale=(0.95, 1.05), resample=Inter.BICUBIC, fill_value=125) # 放射
    coloradjust_op = VT.RandomColorAdjust(brightness=(0.8, 1.1), contrast=(0.8, 1.1), saturation=(0.8, 1.1))  # 色彩调整

    resize_op = VT.Resize([256, 256], Inter.BICUBIC)
    normalize_op = VT.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    trans_op = VT.HWC2CHW()

    selectpolicy_op = VT.RandomSelectSubpolicy([[(cutout_op, 0.5)]])
    

    if is_train: # RandomResizedCrop，随机翻转
        dataset = dataset.map(operations=resize_crop_op, input_columns="image")
        dataset = dataset.map(operations=h_flip_op, input_columns="image")
        # dataset = dataset.map(operations=cutout_op, input_columns="image")
        dataset = dataset.map(operations=selectpolicy_op,input_columns=["image"])
        dataset = dataset.map(operations=random_affine_op, input_columns="image")
        dataset = dataset.map(operations=coloradjust_op, input_columns="image")
    else: # resize
        dataset = dataset.map(operations=resize_op, input_columns="image")
    dataset = dataset.map(operations=normalize_op, input_columns="image")
    dataset = dataset.map(operations=trans_op, input_columns="image")

    dataset = dataset.shuffle(buffer_size=dataset.get_dataset_size())
    dataset = dataset.batch(batch_size=batch_size, drop_remainder=True)
    dataset = dataset.repeat(count=1)

    return dataset
