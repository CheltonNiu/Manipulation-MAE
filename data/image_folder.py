"""A modified image folder class

We modify the official PyTorch image folder (https://github.com/pytorch/vision/blob/master/torchvision/datasets/folder.py)
so that this class can load images from both current directory and its subdirectories.
"""

import torch.utils.data as data
import numpy as np
from PIL import Image
import os, torch
import matplotlib.pyplot as plt
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torchvision import transforms
from data import siamese_transforms as st
IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
    '.tif', '.TIF', '.tiff', '.TIFF',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def make_dataset(dir, max_dataset_size=float("inf")):
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)
    return images[:min(max_dataset_size, len(images))]


def mask_to_onehot(mask, palette=[[255]]):
    """
    Converts a segmentation mask (H, W, C) to (H, W, K) where the last dim is a one
    hot encoding vector, C is usually 1 or 3, and K is the number of class.
    """
    semantic_map = []
    for colour in palette:
        equality = np.equal(mask, colour)
        class_map = np.all(equality, axis=-1)
        semantic_map.append(class_map)
    semantic_map = np.stack(semantic_map, axis=-1).astype(np.float32)
    return semantic_map


def default_loader(path):
    return Image.open(path).convert('RGB')


def build_transform(is_train, args):
    mean = IMAGENET_DEFAULT_MEAN
    std = IMAGENET_DEFAULT_STD
    # train transform
    if is_train:
        transform = st.Compose([
            st.Resize((256, 256)),
            st.ToTensor(),
            st.Normalize(mean, std),
            st.RandomCrop((args.input_size, args.input_size)),
            st.RandomHorizontalFlip(flip_prob=0.3),
            st.RandomRotation([-60, 60]),
            st.GaussianBlur()
        ])
    # test transform
    else:
        transform = st.Compose([
            st.Resize((args.input_size, args.input_size)),
            st.ToTensor(),
            st.Normalize(mean, std)
        ])
    return transform


class ImageFolder(data.Dataset):

    def __init__(self, args, is_train, loader=default_loader):
        self.root = args.train_data_path if is_train else args.test_data_path
        self.imgs = make_dataset(os.path.join(self.root, 'image'))
        self.masks = make_dataset(os.path.join(self.root, 'mask'))
        if len(self.imgs) == 0 :
            raise (RuntimeError("Found 0 images in: " + self.root + "\n"
                                                               "Supported image extensions are: " + ",".join(
                IMG_EXTENSIONS)))

        self.transform = build_transform(is_train=is_train, args=args)
        self.loader = loader

    def __getitem__(self, index):
        img_path = self.imgs[index]
        img = self.loader(img_path)

        '''
        """Coverage dataset"""
        if 't.tif' in img_path:
            mask_path = os.path.join(self.root, 'mask', os.path.basename(img_path).replace('t.tif','forged.tif'))
            '''
    
        """CASIA 1.0 or CASIA 2.0"""
        if 'Tp' in img_path or 'Sp' in img_path:
            mask_path = os.path.join(self.root, 'mask', os.path.basename(img_path).replace('.jpg','_gt.png').replace('.tif','_gt.png'))  # CASIA 2.0 dataset

            mask = self.loader(mask_path)
            mask = mask_to_onehot(mask).squeeze()
        else:
            mask = np.uint8(np.zeros((img.size)))
        mask = Image.fromarray((mask * 255).astype(np.uint8)).convert('L')

        img, mask = self.transform(img, mask)

        mask = mask.ceil()

        return img, mask, img_path


    def __len__(self):
        return len(self.imgs)



