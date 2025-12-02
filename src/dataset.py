import os
from typing import List, Tuple

import numpy as np
import PIL
import torch
import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms


def make_dataset(path: str) -> Tuple[List[str], List[str]]:
    """
    Creates a dataset of paired images from a directory.

    The dataset should be partitioned into two sets: one contains images that
    will have the low pass filter applied, and the other contains images that
    will have the high pass filter applied.

    Args:
        path: string specifying the directory containing images
    Returns:
        images_a: list of strings specifying the paths to the images in set A,
           in lexicographically-sorted order
        images_b: list of strings specifying the paths to the images in set B,
           in lexicographically-sorted order
    """

    images_a, images_b = [], []
    
    for filename in os.listdir(path):
        if os.path.isdir(path + "/" + filename):
            continue
        if filename[1] == "a":
            images_a.append(path + "/" + filename)
        else:
            images_b.append(path + "/" + filename)
    images_a, images_b = sorted(images_a), sorted(images_b)
    return images_a, images_b


def get_cutoff_frequencies(path: str) -> List[Tuple[int]]:
    """
    Gets the cutoff frequencies corresponding to each pair of images.

    Args:
        path: string specifying the path to the .txt file with cutoff frequency
        values
    Returns:
        cutoff_frequencies: numpy array of ints. The array should have the same
            length as the number of image pairs in the dataset
    """

    cutoff_frequencies = []
    f = open(path)
    lines = f.readlines()
    for line in lines:
        cutoff_frequencies.append((int(line[0]), int(line[2])))

    f.close()
    cutoff_frequencies = np.array(cutoff_frequencies)

    return cutoff_frequencies


class HybridImageDataset(data.Dataset):
    """Hybrid images dataset."""

    def __init__(self, image_dir: str, cf_file: str) -> None:
        """
        HybridImageDataset class constructor.

        Args:
            image_dir: string specifying the directory containing images
            cf_file: string specifying the path to the .txt file with cutoff
            frequency values
        """
        images_a, images_b = make_dataset(image_dir)
        cutoff_frequencies = get_cutoff_frequencies(cf_file)

        self.transform = transforms.ToTensor()

        self.images_a = images_a
        self.images_b = images_b
        self.cutoff_frequencies = cutoff_frequencies

    def __len__(self) -> int:
        """Returns number of pairs of images in dataset."""

        return len(self.images_a)


    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """
        Returns the pair of images and corresponding cutoff frequency value at
        index `idx`.

        Args:
            idx: int specifying the index at which data should be retrieved
        Returns:
            image_a: Tensor of shape (c, m, n)
            image_b: Tensor of shape (c, m, n)
            cutoff_frequency: int specifying the cutoff frequency corresponding
               to (image_a, image_b) pair
        """
        image_a = self.transform(PIL.Image.open(self.images_a[idx]))
        image_b = self.transform(PIL.Image.open(self.images_b[idx]))
        cutoff_frequency = self.cutoff_frequencies[idx]

        return image_a, image_b, cutoff_frequency
