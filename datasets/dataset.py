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


from .transform import train_transform, input_transform, target_transform

def read_img_list(img_path_list, img_size =(256,256), rgb_tags = False):
    frame_concate = []
    for img_path_iter in img_path_list:
        if rgb_tags:
            cur_frame = cv2.imread(img_path_iter)
        else:
            cur_frame = cv2.cvtColor(cv2.imread(img_path_iter), cv2.COLOR_BGR2GRAY)
        
        if not img_size == None:
            cur_frame = cv2.resize(cur_frame, (img_size[0], img_size[1]))

        cur_frame_np = np.array(cur_frame, dtype=np.float) / np.float(255.0)
        if len(cur_frame_np.shape)==2:
            cur_frame_np = cur_frame_np[np.newaxis, : , : , np.newaxis]
        if len(cur_frame_np.shape)==3:
            cur_frame_np = cur_frame_np[np.newaxis, : , : , :]
        frame_concate.append(cur_frame_np)
    frame_concate = np.concatenate(frame_concate, axis=0)
    return frame_concate

def read_original_img_list(img_path_list, img_size =(256,256), rgb_tags = False):
    frame_concate = []
    for img_path_iter in img_path_list:
        frame_concate.append(Image.open(img_path_iter))
    # frame_concate = np.concatenate(frame_concate, axis=0)
    return frame_concate

def video_path_list(dataset_path):
    video_list = os.listdir(dataset_path)
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

def generate_trainfile( dataset_root_path, video_path, video_num = 5, frame_interval = 1 ):
    # path = '/mnt/data/DataSet/datasets/' + 'ped1/training/frames/'
    path = os.path.join(dataset_root_path, video_path)
    frame_path_list = video_path_list(path)
    # frame_path_list.sort()
    batch_path_list = []
    for video_iter in frame_path_list:
        for frame_idx in range(0, len(video_iter)-video_num*frame_interval):
            single_batch_list = []
            for inputs_idx in range(0, video_num):
                single_batch_list.append( video_iter[frame_idx+(inputs_idx*frame_interval) ] )
            batch_path_list.append(single_batch_list)

    return batch_path_list

class ImageFolder(Dataset):
    def __init__(self,dataset_root_path, folder_path,video_num = 5, frame_interval = 1, img_size=256, rgb_rags = False):
        self.files = generate_trainfile( dataset_root_path, video_num=video_num)
        # import pdb;pdb.set_trace()
        self.img_size = img_size
        self.transform = train_transform(self.img_size, rgb_rags=rgb_rags)
        self.video_num = video_num

    def __getitem__(self, index):
        img_path = self.files[index % len(self.files)]
        # Extract image as PyTorch tensor
        batch = []
        for idx in range(len(img_path)):
            # tmp =  self.transform(Image.open(img_path[idx]).convert('L'))
            # print(tmp.shape)
            batch.append( self.transform(Image.open(img_path[idx]) ))
        # batch = [torch.expand var for var in batch ]
        # import pdb;pdb.set_trace()
        batch = torch.cat(batch, 0)
        return batch

    def __len__(self):
        return len(self.files)

class ImageFolder3D(Dataset):
    def __init__(self,dataset_root_path, folder_path,video_num = 5, frame_interval = 1, img_size=256, rgb_rags = False):
        self.files = generate_trainfile( dataset_root_path, folder_path,video_num=video_num)
        self.img_size = img_size
        self.transform = train_transform(self.img_size, rgb_rags=rgb_rags)
        self.video_num = video_num

    def __getitem__(self, index):
        img_path = self.files[index % len(self.files)]
        batch = []
        for idx in range(len(img_path)):
            batch.append( torch.unsqueeze(self.transform(Image.open(img_path[idx]) ) , 1))
        batch = torch.cat(batch, 1)
        return batch

    def __len__(self):
        return len(self.files)


class sliding_basic_dataset(object):
    def __init__(self, dataset_path ,path, label_list, rgb_tags = False, img_size =256):
        # super(sliding_basic_dataset, self).__init__(dataset_path ,path, vn_len, type_name,label_list)
        self.path = os.path.join( dataset_path, path)
        print(self.path)
        # self.frame_path_list = video_path_list(self.path,vn_len, type_name)
        # self.frame_path_list.sort()
        dataset_name = dataset_path.split('/')[-2]
        print(dataset_name)
        # import pdb;pdb.set_trace()

        with open( dataset_name + '.pickle','rb') as pf:
            self.frame_path_list = pickle.load(pf)
        
        self.frame_path_list.sort()
        print(len(self.frame_path_list))
        self.video_clips_num =  np.int(len(self.frame_path_list))
        # self.video_num =  np.int(len(self.frame_path_list))
        self.label_list = label_list
        self.rgb_tags = rgb_tags
        self.img_size = (img_size,img_size)
        self.transform = input_transform(img_size, rgb_tags)

    def init_video_sequence(self, selected_video_idx = 'random', video_num = 4, frame_interval = 2, is_frame = True, is_Optical = True,crop_imgsize = 4 ,img_size=256):
        self.videos_end = False
        if selected_video_idx == 'random':
            self.seletced_video_idx = random.randint(0, len(self.frame_path_list) - 1)
        else :
            self.seletced_video_idx =  selected_video_idx
            
        target_frame_idx = []
        selected_label = []

        self.videos_list = read_original_img_list(self.frame_path_list[self.seletced_video_idx], img_size=self.img_size, rgb_tags=self.rgb_tags)

        sample_num = np.int(len(self.frame_path_list[self.seletced_video_idx]) - 1 - video_num* frame_interval)
        print('frame num :', len(self.frame_path_list[self.seletced_video_idx]))
        print('label num :',self.label_list[self.seletced_video_idx].shape)
        for sample_idx in range(sample_num):
            target_frame_idx.append(sample_idx)
            selected_label.append(self.label_list[self.seletced_video_idx][sample_idx ])
            # + (video_num* frame_interval)//2
        
        self.moving_idx = 0
        self.seletced_frame_idx = 0 
        self.target_frame_list = target_frame_idx

        selected_label = np.stack(selected_label)
        self.video_num = video_num
        self.frame_interval = frame_interval
        self.is_frame = is_frame
        self.is_Optical = is_Optical
        self.crop_size = crop_imgsize
        # self.img_size = img_size
        return selected_label

    def get_targetd_video_batches(self, batch_size = 4):
        batches = []
        if not self.videos_end:
            if self.moving_idx + batch_size <len(self.target_frame_list):
                range_0 = self.moving_idx
                range_1 = self.moving_idx + batch_size
                self.moving_idx += batch_size
            else:
                range_0 = self.moving_idx
                range_1 = len(self.target_frame_list)
                self.videos_end = True
                self.moving_idx = 0
            batches = []
            for sample_idx in range(range_0, range_1):
                target_idx = self.target_frame_list[sample_idx]
                # tmp_clips = self.videos_np[target_idx:target_idx+self.video_num,:,:,0]
                tmp_clips = []
                for idx in range(self.video_num):
                    tmp_clips.append( self.transform(self.videos_list[target_idx+idx]) )
                tmp_clips = torch.cat(tmp_clips, 0)
                # import pdb;pdb.set_trace()

                batches.append( torch.unsqueeze( tmp_clips,0 ) )
            batches = torch.cat(batches, 0)
            # print(batches.shape)
            return batches, True
        else:
            return batches, False
        return 
    
    def get_targetd_video_batches_3d(self, batch_size = 4):
        batches = []
        if not self.videos_end:
            if self.moving_idx + batch_size <len(self.target_frame_list):
                range_0 = self.moving_idx
                range_1 = self.moving_idx + batch_size
                self.moving_idx += batch_size
            else:
                range_0 = self.moving_idx
                range_1 = len(self.target_frame_list)
                self.videos_end = True
                self.moving_idx = 0
            batches = []
            for sample_idx in range(range_0, range_1):
                target_idx = self.target_frame_list[sample_idx]
                # tmp_clips = self.videos_np[target_idx:target_idx+self.video_num,:,:,0]
                tmp_clips = []
                for idx in range(self.video_num):
                    tmp_clips.append(  torch.unsqueeze( self.transform(self.videos_list[target_idx+idx]), 1) )
                tmp_clips = torch.cat(tmp_clips, 1)
                # import pdb;pdb.set_trace()

                batches.append( torch.unsqueeze( tmp_clips,0 ) )
            batches = torch.cat(batches, 0)
            # print(batches.shape)
            return batches, True
        else:
            return batches, False
        return 

if __name__ == "__main__":
    batch_path_list = generate_trainfile( '/mnt/data/DataSet/datasets/', 'avenue/training/frames')
    print(len(batch_path_list))
    import pdb; pdb.set_trace()