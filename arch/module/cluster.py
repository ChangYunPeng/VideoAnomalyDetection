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


def cluster_alpha(max_n = 40):

    constant_value = 1  # specs.embedding_size # Used to modify the range of the alpha scheme
    # max_n = 40  # Number of alpha values to consider
    alphas = np.zeros(max_n, dtype=float)
    alphas[0] = 0.1
    for i in range(1, max_n):
        alphas[i] = (2 ** (1 / (np.log(i + 1)) ** 2)) * alphas[i - 1]
    alphas = alphas / constant_value
    # print(alphas)
    return alphas


class PosSoftAssign(nn.Module):
    def __init__(self, dims=1,alpha=1.0):
        super(PosSoftAssign, self).__init__()    
        self.dims = dims
        self.alpha = alpha

    def forward(self,x, alpha=None):
        if not alpha==None:
            self.alpha = alpha
        x_max,_ = torch.max(x,self.dims,keepdim=True)
        exp_x = torch.exp(self.alpha*(x-x_max))
        soft_x = exp_x/(exp_x.sum(self.dims,keepdim=True))
        return soft_x

class NegSoftAssign(nn.Module):
    def __init__(self, dims=1,alpha=32.0):
        super(NegSoftAssign, self).__init__()    
        self.dims = dims
        self.alpha = alpha

    def forward(self,x, alpha=None):
        if not alpha==None:
            self.alpha = alpha
        
        x_min,_ = torch.min(x,self.dims,keepdim=True)
        exp_x = torch.exp((-self.alpha)*(x-x_min))
        soft_x = exp_x/(exp_x.sum(self.dims,keepdim=True))
        return soft_x

class EuclidDistance_Assign_Module(nn.Module):
    def __init__(self, feature_dim, cluster_num = 64, maxpool=1, soft_assign_alpha=32.0):
        super(EuclidDistance_Assign_Module, self).__init__()   
        self.euclid_dis = torch.cdist 
        self.act = nn.Sigmoid()
        self.feature_dim = feature_dim
        self.cluster_num = cluster_num
        self.norm = F.normalize

        self.assign_func = NegSoftAssign(-1,soft_assign_alpha)
        self.register_param()
     
    def register_param(self,):    
        cluster_center = nn.Parameter(  torch.rand( self.cluster_num, self.feature_dim ) , requires_grad = True )
        identity_matrix = nn.Parameter( torch.eye(self.cluster_num) , requires_grad = False )
        self.register_parameter('cluster_center', cluster_center)
        self.register_parameter('identity_matrix', identity_matrix )
        return
    
    def self_similarity(self):
        return self.euclid_dis(self.cluster_center, self.cluster_center)
    
    def forward(self,x, alpha=None):
        x = self.norm(x,p=2,dim=1)
        x_transpose = x.permute(0,2,3,1).contiguous() 
        x_re = x_transpose.view( x.shape[0] , -1 , x.shape[1] )
        soft_assign = self.euclid_dis(x_re, self.cluster_center.unsqueeze(0))
        
        x_distance = soft_assign.view(x_transpose.shape[0], x_transpose.shape[1], x_transpose.shape[2], self.cluster_num)
        x_distance_assign = self.assign_func(x_distance,alpha)
        return x_distance, x_distance_assign




if __name__ == "__main__":
    soft_assign = PosSoftAssign(0,8)
    xx  =  torch.rand(10)
    soft_xx = soft_assign(xx)
    print(xx)
    print(soft_xx)