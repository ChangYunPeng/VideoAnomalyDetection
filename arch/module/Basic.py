import torch
import torch.nn as nn
import numpy as np


def psnr(pred, gt):
    return 10 * log10(1 / torch.sum((pred - gt) ** 2).item())


def video_static_motion(frames, img_channel, frames_num):
    """0-th frame for appearance feature 1-th to last frames for motion feature"""
    motion_in = frames[:,0*img_channel:(frames_num-1)*img_channel,:,:]
    static_in = frames[:,0*img_channel:1*img_channel,:,:]
    motion_target = frames[:,(frames_num-1)*img_channel:,:,:]-static_in
    static_target = static_in

    return static_in, motion_in, static_target, motion_target


def video_split_static_and_motion_seq(frames, img_channel, frames_num):
    """0-th frame for appearance feature 1-th to last frames for motion feature"""
    motion_in = frames[:,0*img_channel:(frames_num-1)*img_channel,:,:]
    static_in = frames[:,(frames_num-1)*img_channel:,:,:]
    motion_target = motion_in - static_in.repeat(1,frames_num-1,1,1)
    static_target = static_in
    return static_in, motion_in, static_target, motion_target



def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv_up3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""

    downsample = [
        nn.PixelShuffle(upscale_factor=stride),
        nn.Conv2d(in_planes//(stride**2), out_planes,kernel_size=3, stride=1,padding=1, bias=False),
    ]
    downsample = nn.Sequential(*downsample)

    return downsample



class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        if stride == -2:
            self.conv1 = conv_up3x3(inplanes, planes, -1*stride)
            # import pdb;pdb.set_trace()
        else:
            self.conv1 = conv3x3(inplanes, planes, stride)

        self.relu = nn.LeakyReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)

        if self.downsample is not None:
            residual = self.downsample(x)
        
        out += residual
        out = self.relu(out)
        return out


class BlurFunc(nn.Module):
    def __init__(self,ratio=2):
        super(BlurFunc, self).__init__()
        self.down =  nn.AvgPool2d(ratio,ratio)
        self.up = nn.Upsample(scale_factor=ratio, mode='bilinear',align_corners=False)

    def forward(self, x):
        x = self.down(x)
        x = self.up(x)
        return x
    
if __name__ == "__main__":
    
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '2'
    print('#### Test Case ###')
    model = BlurFunc( ).cuda()
    x = torch.rand(2,12,256,256).cuda()
    out = model(x)
    print(out.shape)