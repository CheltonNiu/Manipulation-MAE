import math
import numpy as np
import random
from PIL import Image

import torch
from torchvision import transforms as T
from torchvision.transforms import functional as F


class Resize(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, image, mask=None):
        image = F.resize(image, self.size)
        if mask is not None:
            mask = F.resize(mask, self.size, interpolation=F.InterpolationMode.BICUBIC)
        return image, mask


class RandomHorizontalFlip(object):
    def __init__(self, flip_prob):
        self.flip_prob = flip_prob

    def __call__(self, image, mask=None):
        if random.random() < self.flip_prob:
            image = F.hflip(image)
            if mask is not None:
                mask = F.hflip(mask)
        return image, mask


class RandomCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, image, mask):
        crop_params = T.RandomCrop.get_params(image, self.size)
        image = F.crop(image, *crop_params)
        if mask is not None:
            mask = F.crop(mask, *crop_params)
        return image, mask


class RandomRotation(object):
    def __init__(self, degrees):
        self.degrees = degrees

    def __call__(self, image, mask):
        rotation_params = T.RandomRotation.get_params(self.degrees)
        image = F.rotate(img=image,angle=rotation_params)
        if mask is not None:
            mask = F.rotate(img=mask, angle=rotation_params)
        return image, mask


class CenterCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, image, mask):
        image = F.center_crop(image, self.size)
        if mask is not None:
            mask = F.center_crop(mask, self.size)
        return image, mask


class GaussianBlur(object):
    def __init__(self, kernel_size=5):
        self.kernel_size = kernel_size

    def __call__(self, image, mask):
        image = F.gaussian_blur(img=image, kernel_size=self.kernel_size)
        return image, mask


class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image, mask):
        image = F.normalize(image, mean=self.mean, std=self.std)
        return image, mask


class Pad(object):
    def __init__(self, padding_n, padding_fill_value=0, padding_fill_target_value=0):
        self.padding_n = padding_n
        self.padding_fill_value = padding_fill_value
        self.padding_fill_target_value = padding_fill_target_value

    def __call__(self, image, mask):
        image = F.pad(image, self.padding_n, self.padding_fill_value)
        if mask is not None:
            mask = F.pad(mask, self.padding_n, self.padding_fill_target_value)
        return image, mask


class ToTensor(object):
    def __call__(self, image, mask):
        image = F.to_tensor(image)
        if mask is not None:
        #   mask = torch.as_tensor(np.array(mask), dtype=torch.int64)
            mask = F.to_tensor(mask)
        return image, mask


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, mask=None):
        for t in self.transforms:
            image, mask = t(image, mask)
        return image, mask