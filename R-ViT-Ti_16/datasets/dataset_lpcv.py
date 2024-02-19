import os
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import random

from typing import List, Tuple, BinaryIO
from torchvision.transforms import transforms
import cv2
import numpy
from cv2.mat_wrapper import Mat
from imageio.core.util import Array
from imageio import imread
from numpy import ndarray
import sys
import os
import glob

def random_rot_flip(image, label):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    label = np.rot90(label, k)
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    label = np.flip(label, axis=axis).copy()
    return image, label


def random_rotate(image, label):
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, order=0, reshape=False)
    label = ndimage.rotate(label, angle, order=0, reshape=False)
    return image, label


class RandomGenerator(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        if random.random() > 0.5:
            image, label = random_rot_flip(image, label)
        elif random.random() > 0.5:
            image, label = random_rotate(image, label)
        x, y = image.shape
        if x != self.output_size[0] or y != self.output_size[1]:
            image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=3)  # why not 3?
            label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
        label = torch.from_numpy(label.astype(np.float32))
        sample = {'image': image, 'label': label.long()}
        return sample

def loadImageToTensor(imagePath: str) -> torch.Tensor:
    MEAN: Tuple[float, float, float] = (0.485, 0.456, 0.406)
    STANDARD_DEVIATION: Tuple[float, float, float] = (0.229, 0.224, 0.225)

    image: Array = imread(uri=imagePath)
    resizedImage: Mat = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)
    imageTensor: Tensor = transforms.ToTensor()(resizedImage)
    imageTensor: Tensor = transforms.Normalize(mean=MEAN, std=STANDARD_DEVIATION)(
        imageTensor
    )
    # imageTensor: Tensor = imageTensor.unsqueeze(0)

    return imageTensor.numpy()

def loadGroundTruthImage(imagePath: str) -> ndarray:
    image: Array = imread(uri=imagePath).astype(numpy.uint8)

    if len(image.shape) == 3:
        image = image[:, :, 0]

    resizedImage: Mat = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)
    resizedImage: Mat = cv2.resize(
        resizedImage, (224, 224), interpolation=cv2.INTER_NEAREST
    )
    outputImage: ndarray = resizedImage[numpy.newaxis, :, :]

    return outputImage

class LPCV_dataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, gt_data_dir, split, transform=None, target_transform=None, augmentation=None):
        
        self.split = split
        self.transform = transform
        self.target_transform = target_transform
        self.augmentation = augmentation
        
        if self.split == "train":
            self.data_dir = os.path.join(data_dir, split)
            self.gt_data_dir = os.path.join(gt_data_dir, split)
            self.img_list = [file for file in os.listdir(self.data_dir) if file.startswith('train_')]
        
        else:
            self.data_dir = os.path.join(data_dir, split)
            self.gt_data_dir = os.path.join(gt_data_dir, split)
            self.img_list = [file for file in os.listdir(self.gt_data_dir) if file.startswith('val_')]
        
    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        img_name = self.img_list[idx]
        img_path = os.path.join(self.data_dir, img_name)
        
        if self.split == "train":
            gt_img_path = os.path.join(self.gt_data_dir, f"train_{img_name[6:]}")
        else:
            gt_img_path = os.path.join(self.gt_data_dir, f"val_{img_name[4:]}")

        # Load images
        img = loadImageToTensor(img_path)
        gt_img: Array = loadGroundTruthImage(gt_img_path)

        # Apply transformations if provided
        # if self.transform:
        #     img = np.array(self.transform(img))
        #     gt_img = np.array(self.transform(gt_img))
        
        sample = {'image' : img, 'label' : gt_img}
        
        if self.augmentation:
            sample = self.augmentation(sample)
            
        sample['case_name'] = img_name

        return sample
