import glob
import random
import os
import sys
import numpy as np
import pickle
from PIL import Image
import cv2
import torch
import torch.nn.functional as F
from torchvision.transforms import Compose, CenterCrop, ToTensor, Resize,Normalize, Grayscale, RandomHorizontalFlip, RandomVerticalFlip
# from utils.augmentations import horisontal_flip
from torch.utils.data import Dataset
import torchvision.transforms as transforms


def target_transform(crop_size):
    return Compose([
        # CenterCrop(crop_size),
        # Resize(crop_size),
        ToTensor()
    ])

def train_transform(img_size, rgb_rags = False):
    if rgb_rags:
        return Compose([
        Resize(img_size),
        ToTensor()
        ])
    else:
        return Compose([
            # RandomHorizontalFlip(),
            Grayscale(),
            # Resize([img_size,img_size]),
            Resize(img_size),
            ToTensor(),
            # RandomVerticalFlip()
        ])

def input_transform(img_size, rgb_rags = False):
    if rgb_rags:
        return Compose([
        Resize(img_size),
        ToTensor()
        ])
    else:
        return Compose([
            Grayscale(),
            Resize([img_size,img_size]),
            ToTensor()
        ])




def downsample_transform(img_size,ratio=2):

    return Compose([
            Resize([img_size//ratio,img_size//ratio]),
            Resize([img_size,img_size]),
        ])