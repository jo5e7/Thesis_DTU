from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from PIL import Image
from torchvision.transforms import functional as F
from torch import nn
import collections
import sys
from torchvision import transforms
from matplotlib import pyplot as plt
import numpy as np

if sys.version_info < (3, 3):
    Sequence = collections.Sequence
    Iterable = collections.Iterable
else:
    Sequence = collections.abc.Sequence
    Iterable = collections.abc.Iterable


class PadChestDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, csv_file, labels, root_dir, transform=None, testing=False):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.pad_chest_df = pd.read_csv(csv_file)
        if testing:
            zero_df = self.pad_chest_df["ImageDir"] == 0
            self.pad_chest_df = self.pad_chest_df[zero_df]
            self.pad_chest_df = self.pad_chest_df.reset_index(drop=True)

        self.root_dir = root_dir
        self.transform = transform
        self.labels = labels

    def __len__(self):
        return len(self.pad_chest_df)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir,
                                self.pad_chest_df.loc[idx, "ImageID"])

        labels_output = self.pad_chest_df.loc[idx, self.labels]
        labels_output = labels_output.tolist()
        #print(idx, labels_output)

        #image = io.imread(img_name)
        #image = np.stack((image,) * 3, axis=-1)

        image = Image.open(img_name).convert("RGB")

        #image.show()
        #plt.imshow(image, interpolation='nearest')
        #plt.show()



        if self.transform:
            image = self.transform(image)

        #sample = {'image': image, 'labels': labels_output}
        return image, torch.FloatTensor(labels_output)



class PadChestDataset_loc(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, csv_file, labels, position_labels, root_dir, transform=None, testing=False):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.pad_chest_df = pd.read_csv(csv_file)
        if testing:
            zero_df = self.pad_chest_df["ImageDir"] == 0
            self.pad_chest_df = self.pad_chest_df[zero_df]
            self.pad_chest_df = self.pad_chest_df.reset_index(drop=True)

        self.root_dir = root_dir
        self.transform = transform
        self.labels = labels
        self.position_labels = position_labels

    def __len__(self):
        return len(self.pad_chest_df)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir,
                                self.pad_chest_df.loc[idx, "ImageID"])

        labels_output = self.pad_chest_df.loc[idx, self.labels]
        labels_output = labels_output.tolist()

        position_labels_output = self.pad_chest_df.loc[idx, self.position_labels]
        position_labels_output = position_labels_output.tolist()

        # Positional labels only if there is a radiographical finding
        if 0 in labels_output:
            for i in range(len(position_labels_output)):
                position_labels_output[i] = 0
        #print(idx, labels_output)
        #print(idx, position_labels_output)

        #image = io.imread(img_name)
        #image = np.stack((image,) * 3, axis=-1)

        image = Image.open(img_name).convert("RGB")

        #image.show()
        #plt.imshow(image, interpolation='nearest')
        #plt.show()


        sample = {'image': image, 'labels_output': labels_output, 'position_labels_output': position_labels_output}
        if self.transform:
            sample = self.transform(sample)

        #sample = {'image': image, 'labels': labels_output}
        return sample['image'], torch.FloatTensor(sample['labels_output']), torch.FloatTensor(sample['position_labels_output'])

# Transforms

class RandomHorizontalFlip_loc(object):
    """Horizontally flip the given PIL Image randomly with a given probability.

    Args:
        p (float): probability of the image being flipped. Default value is 0.5
    """

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, sample):
        """
        Args:
            img (PIL Image): Image to be flipped.

        Returns:
            PIL Image: Randomly flipped image.
        """

        if random.random() < self.p:
            img = sample['image']
            radiograhical_findings = sample['labels_output']
            locations = sample['position_labels_output']

            #print(locations)
            left = locations[0]
            right = locations[1]
            locations[0] = right
            locations[1] = left
            #print(locations)

            sample = {'image': F.hflip(img), 'labels_output': radiograhical_findings, 'position_labels_output': locations}
            return sample

        return sample

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)


class Resize_loc(object):
    """Resize the input PIL Image to the given size.

    Args:
        size (sequence or int): Desired output size. If size is a sequence like
            (h, w), output size will be matched to this. If size is an int,
            smaller edge of the image will be matched to this number.
            i.e, if height > width, then image will be rescaled to
            (size * height / width, size)
        interpolation (int, optional): Desired interpolation. Default is
            ``PIL.Image.BILINEAR``
    """

    def __init__(self, size, interpolation=Image.BILINEAR):
        assert isinstance(size, int) or (isinstance(size, Iterable) and len(size) == 2)
        self.size = size
        self.interpolation = interpolation

    def __call__(self, sample):
        """
        Args:
            img (PIL Image): Image to be scaled.

        Returns:
            PIL Image: Rescaled image.
        """
        img = sample['image']
        radiograhical_findings = sample['labels_output']
        locations = sample['position_labels_output']
        #print(img)
        sample = {'image': F.resize(img, self.size, self.interpolation), 'labels_output': radiograhical_findings, 'position_labels_output': locations}

        return sample


import numbers
import random

class RandomRotation_loc(object):
    """Rotate the image by angle.

    Args:
        degrees (sequence or float or int): Range of degrees to select from.
            If degrees is a number instead of sequence like (min, max), the range of degrees
            will be (-degrees, +degrees).
        resample ({PIL.Image.NEAREST, PIL.Image.BILINEAR, PIL.Image.BICUBIC}, optional):
            An optional resampling filter. See `filters`_ for more information.
            If omitted, or if the image has mode "1" or "P", it is set to PIL.Image.NEAREST.
        expand (bool, optional): Optional expansion flag.
            If true, expands the output to make it large enough to hold the entire rotated image.
            If false or omitted, make the output image the same size as the input image.
            Note that the expand flag assumes rotation around the center and no translation.
        center (2-tuple, optional): Optional center of rotation.
            Origin is the upper left corner.
            Default is the center of the image.

    .. _filters: https://pillow.readthedocs.io/en/latest/handbook/concepts.html#filters

    """

    def __init__(self, degrees, resample=False, expand=False, center=None):
        if isinstance(degrees, numbers.Number):
            if degrees < 0:
                raise ValueError("If degrees is a single number, it must be positive.")
            self.degrees = (-degrees, degrees)
        else:
            if len(degrees) != 2:
                raise ValueError("If degrees is a sequence, it must be of len 2.")
            self.degrees = degrees

        self.resample = resample
        self.expand = expand
        self.center = center

    @staticmethod
    def get_params(degrees):
        """Get parameters for ``rotate`` for a random rotation.

        Returns:
            sequence: params to be passed to ``rotate`` for random rotation.
        """
        angle = random.uniform(degrees[0], degrees[1])

        return angle

    def __call__(self, sample):
        """
            img (PIL Image): Image to be rotated.

        Returns:
            PIL Image: Rotated image.
        """
        #print('rotation')
        img = sample['image']
        radiograhical_findings = sample['labels_output']
        locations = sample['position_labels_output']
        angle = self.get_params(self.degrees)

        sample = {'image': F.rotate(img, angle, self.resample, self.expand), 'labels_output': radiograhical_findings, 'position_labels_output': locations}

        return sample


class ToTensor_loc(object):
    """Convert a ``PIL Image`` or ``numpy.ndarray`` to tensor.

    Converts a PIL Image or numpy.ndarray (H x W x C) in the range
    [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0]
    if the PIL Image belongs to one of the modes (L, LA, P, I, F, RGB, YCbCr, RGBA, CMYK, 1)
    or if the numpy.ndarray has dtype = np.uint8

    In the other cases, tensors are returned without scaling.
    """

    def __call__(self, sample):
        """
        Args:
            pic (PIL Image or numpy.ndarray): Image to be converted to tensor.

        Returns:
            Tensor: Converted image.
        """
        img = sample['image']
        radiograhical_findings = sample['labels_output']
        locations = sample['position_labels_output']

        sample = {'image': F.to_tensor(img), 'labels_output': radiograhical_findings, 'position_labels_output': locations}

        return sample

    def __repr__(self):
        return self.__class__.__name__ + '()'


class Normalize_loc(object):
    """Normalize a tensor image with mean and standard deviation.
    Given mean: ``(M1,...,Mn)`` and std: ``(S1,..,Sn)`` for ``n`` channels, this transform
    will normalize each channel of the input ``torch.*Tensor`` i.e.
    ``input[channel] = (input[channel] - mean[channel]) / std[channel]``

    .. note::
        This transform acts out of place, i.e., it does not mutates the input tensor.

    Args:
        mean (sequence): Sequence of means for each channel.
        std (sequence): Sequence of standard deviations for each channel.
    """

    def __init__(self, mean, std, inplace=False):
        self.mean = mean
        self.std = std
        self.inplace = inplace

    def __call__(self, sample):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.

        Returns:
            Tensor: Normalized Tensor image.
        """

        img = sample['image']
        radiograhical_findings = sample['labels_output']
        locations = sample['position_labels_output']

        sample = {'image': F.normalize(img, self.mean, self.std, self.inplace), 'labels_output': radiograhical_findings,
                  'position_labels_output': locations}

        return sample


    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)
