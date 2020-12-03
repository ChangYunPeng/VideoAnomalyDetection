import glob
import random
import os
import sys
import tqdm 
import numpy as np
import pickle
from PIL import Image
import cv2
import torch
import torch.nn.functional as F
from torchvision.transforms import Compose, CenterCrop, ToTensor, Resize,Normalize, Grayscale, RandomHorizontalFlip, RandomVerticalFlip
# from utils.augmentations import horisontal_flip
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms


from .transform import train_transform, input_transform, target_transform
from .datasets_labels import gather_datasets_labels


def generate_clips_list( frame_list, clips_length = 5, frame_interval = 1 ):
    batch_path_list = []
    
    for frame_idx in range(0, len(frame_list)-clips_length*frame_interval):
        single_batch_list = []
        for inputs_idx in range(0, clips_length):
            single_batch_list.append( frame_list[frame_idx+(inputs_idx*frame_interval) ] )
        batch_path_list.append(single_batch_list)

    return batch_path_list


def video_path_list(dataset_path):
    dataset_path = os.path.join(dataset_path,'testing/frames')
    video_list = os.listdir( dataset_path )
    video_list.sort()
    video_path_list = []
    idx = 0
    for video_path_iter in video_list:
        img_list = os.listdir( os.path.join( dataset_path,video_path_iter ) ) 
        img_list = [os.path.join(dataset_path,video_path_iter,var) for var in img_list]
        # print(video_path_iter)
        # print(idx,len(img_list))
        idx = idx+1
        img_list.sort(key = lambda x: int(os.path.basename(x).split('.')[0]))
        video_path_list.append(img_list)
    
    return video_path_list

class sliding_datasets(object):
    def __init__(self, path_list = [], img_size=256, rgb_rags = False, bacthsize=24 ):
        self.files = path_list
        self.dataset_length = len(path_list)
        self.img_size = img_size
        self.transform = train_transform(self.img_size, rgb_rags=rgb_rags)
        self.batchsize = bacthsize
        self.moving_idx = 0
        self.finished_tag = False
        self.fetch_nums = np.int( np.ceil(  self.dataset_length / self.batchsize  ))
    

    def clip(self,frame_idx):
        img_path = self.files[frame_idx]
        # print(img_path[0])
        batch = []
        for idx in range(len(img_path)):
            batch.append( self.transform(Image.open(img_path[idx]) ))
        batch = torch.cat(batch, 0)
        return batch


    def fetch(self):
        batches = []
        for idx in range(self.batchsize):
            batch = self.clip(self.moving_idx)
            batches.append( torch.unsqueeze(batch,0) )
            self.moving_idx += 1
            if self.moving_idx == self.dataset_length:
                self.finished_tag = True
                self.moving_idx = 0
                break        
        batches = torch.cat(batches,0)
        return batches

class sliding_whole_dataset(object):
    def __init__(self, config , termporal_tag= False):
        self.config = config
        self.termporal_tag = termporal_tag
        self.frame_path_list = video_path_list(config.dataset_path)        
        self.videos_num =  np.int(len(self.frame_path_list))
        self.clips_length = config.clips_length
        self.label_list = gather_datasets_labels(config.dataset_path, config.dataset_name)
        self.rgb_tags = config.rgb_tags
        self.img_size = config.img_size
        self.transform = train_transform(self.img_size, self.rgb_tags)
    
    def generate_video_sequence(self):
        dataset_sequnces = []
        cropped_label_list = []

        for video_idx in range(self.videos_num):
            video_sequnces = generate_clips_list(self.frame_path_list[video_idx] , clips_length = self.clips_length  , frame_interval = 1 )
            video_length = len(video_sequnces)
            label_length = len(self.label_list[video_idx])
            drops = (label_length - video_length)//2
            labels = self.label_list[video_idx][-len(video_sequnces):]
            
            dataset_sequnces += video_sequnces
            cropped_label_list.append(labels)
        
        dataset = sliding_datasets(dataset_sequnces, self.img_size , self.rgb_tags,bacthsize=self.config.eval_batches)
        return dataset, cropped_label_list


if __name__ == "__main__":
    label_list = gather_datasets_labels('/mnt/data/DataSet/datasets/')[1]
    for label_iter in label_list:
        print(label_iter)
