import sys
import os
# from datasets_sequence import multi_train_datasets, multi_test_datasets
import torch
import time
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
import numpy as np
import torch.nn.init as init




def softmax_normalization(x, func):
    b,c,h,w = x.shape
    x_re = x.view([b,c,-1])
    x_norm = func(x_re)
    x_norm = x_norm.view([ b,c,h,w ])
    return x_norm

class Variance_Attention(nn.Module):
    def __init__(self, depth_in, depth_embedding,maxpool=1):
        super(Variance_Attention, self).__init__()     
        self.flow = nn.Sequential()
        self.flow.add_module('proj_conv', nn.Conv2d(depth_in, depth_embedding, kernel_size=1,padding=False, bias=False))
        self.maxpool = maxpool
        if not maxpool == 1:
            self.flow.add_module('pool', nn.AvgPool2d( kernel_size=maxpool, stride = maxpool ))
            self.unpool = nn.Upsample(scale_factor = maxpool)
        self.norm_func = nn.Softmax(-1)

    def forward(self,x):
        proj_x = self.flow(x)
        mean_x = torch.mean(proj_x, dim=1, keepdim=True)
        variance_x = torch.sum(torch.pow(proj_x - mean_x, 2) , dim = 1 , keepdim=True)
        var_norm = softmax_normalization(variance_x, self.norm_func)
        if not self.maxpool == 1:
            var_norm = self.unpool(var_norm)
        
        return torch.exp(var_norm)*x 
    
if __name__ == "__main__":
    xx = torch.rand(1,8,16,16)
    var_att = Variance_Attention(8,16,maxpool=2)
    yy = var_att(xx)