import os

import torch.nn as nn
import torch
import math
import numpy as np
import tqdm

from torchvision.utils import save_image
import torch.optim as optim
import torch.nn.functional as F

from arch.module.ResNet import  ResEncoder, ResDecoder
from arch.module.ResUNet import UResEncoder,UResDecoder
from arch.module.Basic import video_static_motion, BasicBlock, BlurFunc, video_split_static_and_motion_seq
from arch.module.cluster import EuclidDistance_Assign_Module

from arch.module.loss_utils import gradient_loss, gradient_metric
from arch.module.eval_utils import psnr,batch_psnr, l1_metric, l2_metric , min_max_np, calcu_result, reciprocal_metric, log_metric,tpsnr


class PredRes_AE_Cluster_Model(nn.Module):
    def __init__(self, static_channel_in=3,static_channel_out=3, static_layer_struct=[2,2,2,2], static_layer_nums = 4 ,motion_channel_in=12, motion_channel_out=3, motion_layer_struct=[2,2,2,2], motion_layer_nums = 4, img_channel=3,frame_nums=5, cluster_num=128, blur_ratio=3, seq_tag=False, model_type='res',):
        super(PredRes_AE_Cluster_Model, self).__init__()
        self.img_channel = img_channel
        self.frame_nums = frame_nums
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(1)
        self.tanh = nn.Tanh()
        self.cluster_num = cluster_num
        self.blur_ratio = blur_ratio
        self.seq_tag = seq_tag

        self.neck_planes  = 32
        self.inter_planes =  self.neck_planes*(2**static_layer_nums)
        #  256
        self.bulr_function = BlurFunc(ratio=blur_ratio)

        self.model_type = model_type
        
        self.motion_encoder_func = UResEncoder(BasicBlock, input_channels=motion_channel_in ,  layers=motion_layer_struct, layer_num=motion_layer_nums , neck_planes  = self.neck_planes, att_tag=True, last_layer_softmax = True)
        self.motion_decoder_func = UResDecoder(BasicBlock, output_channels=motion_channel_out ,  layers=motion_layer_struct[::-1], layer_num=motion_layer_nums , neck_planes  = self.neck_planes)
        self.static_encoder_func = ResEncoder(BasicBlock, input_channels=static_channel_in ,  layers=static_layer_struct, layer_num=static_layer_nums , neck_planes  = self.neck_planes , last_layer_softmax = True )
        self.static_decoder_func = ResDecoder(BasicBlock, output_channels=static_channel_out ,  layers=static_layer_struct[::-1], layer_num=static_layer_nums , neck_planes  = self.neck_planes)
        
        self.cluster = EuclidDistance_Assign_Module( self.inter_planes+self.inter_planes, cluster_num=self.cluster_num, soft_assign_alpha=25.0 )
        
        self.ae_par = list(self.motion_encoder_func.parameters()) \
                    + list(self.motion_decoder_func.parameters()) \
                    + list(self.static_encoder_func.parameters()) \
                    + list(self.static_decoder_func.parameters()) 
        
        self.motion_par = list(self.motion_encoder_func.parameters()) + list(self.motion_decoder_func.parameters())
        self.static_par = list(self.static_encoder_func.parameters()) + list(self.static_decoder_func.parameters()) 


        self.cluster_par = list( self.cluster.parameters() )  + list( self.motion_encoder_func.layer_list[-1].last_layer.parameters() ) + list(  self.static_encoder_func.proj.parameters() )


        self.upfunc = nn.Upsample(scale_factor=2**motion_layer_nums)

        self.l2_criterion = nn.MSELoss()
        self.l1_criterion = nn.L1Loss()

    def forward(self,x ,alpha=None, stage=['G']):
        if self.seq_tag:
            static_in, motion_in, static_target, motion_target  = video_split_static_and_motion_seq(x,self.img_channel, self.frame_nums)
        else:
            static_in, motion_in, static_target, motion_target  = video_static_motion(x,self.img_channel, self.frame_nums)
        

        if 'S' in stage or 'E' in stage or 'F' in stage:
            static_encoder = self.static_encoder_func( static_in )
            
            
            static_decoder =  self.static_decoder_func(static_encoder)

            if 'G' in stage:
                loss_deblur = self.l2_criterion( static_decoder , static_in )
                grad_deblur = gradient_loss( static_decoder , static_in )
                return loss_deblur,grad_deblur
            elif 'S'  in stage and 'E'  in stage:
                deblur_psnr = tpsnr( static_decoder , static_in )
                return deblur_psnr
                

        if 'M' in stage or 'E' in stage or 'F' in stage:
            motion_encoder = self.motion_encoder_func( motion_in )
            motion_decoder = self.motion_decoder_func(motion_encoder) 
            if self.seq_tag:
                static_in =  static_in.repeat(1,self.frame_nums-1,1,1)
            pred_target = static_in + motion_target

            if 'G' in stage:
                loss_predict = self.l2_criterion( motion_decoder + static_in, pred_target )
                grad_predict = gradient_loss( motion_decoder + static_in, pred_target )
                return loss_predict,grad_predict
            elif 'M'  in stage and 'E'  in stage:
                predict_psnr = tpsnr( motion_decoder + static_in, pred_target )
                return predict_psnr
        

        if 'ini' in stage:
            static_encoder_rep = static_encoder.permute(0,2,3,1).contiguous()
            static_encoder_rep = static_encoder_rep.reshape(-1, static_encoder_rep.shape[-1]).contiguous()
            motion_encoder_rep = motion_encoder[0].permute(0,2,3,1).contiguous()
            motion_encoder_rep = motion_encoder_rep.reshape(-1, motion_encoder_rep.shape[-1]).contiguous()
            cat_rep = torch.cat([static_encoder_rep, motion_encoder_rep],-1)
            return cat_rep
        
        if 'C' in stage or 'E' in stage:
            cat_encoder = torch.cat( [ static_encoder, motion_encoder[0] ] ,1 )
            rep_dist, softassign = self.cluster(cat_encoder)
            loss_cluster = torch.mean( rep_dist*softassign ,[1,2,3])
            if 'E' not in stage:
                return loss_cluster, rep_dist, cat_encoder

        if 'F' in stage :
            loss_deblur = self.l2_criterion( static_decoder , static_in )
            grad_deblur = gradient_loss( static_decoder , static_in )

            if self.seq_tag:
                static_decoder = static_decoder.repeat(1,self.frame_nums-1,1,1)
            loss_predict = self.l2_criterion( motion_decoder + static_in, pred_target )
            grad_predict = gradient_loss( motion_decoder + static_in, pred_target )

            pred_recon = static_decoder + motion_decoder
            loss_recon = self.l2_criterion( pred_recon , pred_target )
            grad_recon = gradient_loss( pred_recon , pred_target )


            return loss_deblur,loss_predict,loss_recon,grad_deblur,grad_predict,grad_recon

        if 'E' in stage:
            loss_cluster_map = torch.mean( rep_dist*softassign ,[3], keepdim=True)
            loss_cluster_map = loss_cluster_map.permute([0,3,1,2]).contiguous()
            loss_cluster_map = self.upfunc(loss_cluster_map)
            return static_decoder, motion_decoder,loss_cluster_map