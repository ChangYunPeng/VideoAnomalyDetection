import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

from numpy import log10
from torch import log10 as tlog10
from sklearn import metrics

def batch_psnr(pred, gt):
    # pred = pred*255.0
    # gt = gt*255.0

    return 10 * log10(1 / torch.mean((pred - gt) ** 2).item())

def tpsnr(pred, gt):
    # pred = pred*255.0
    # gt = gt*255.0
    return 10 * tlog10(1 / torch.mean((pred - gt) ** 2,[1,2,3]))
    # .cpu().numpy())


def psnr(pred, gt):
    # pred = pred*255.0
    # gt = gt*255.0
    return 10 * log10(1 / torch.mean((pred - gt) ** 2,[1,2,3]).cpu().numpy())


def log_metric( batch_loss ):

    return 10*log10(1/ batch_loss )


def reciprocal_metric( batch_loss ):

    return 1/ batch_loss
    

def l1_metric(pred, gt):
    
    return torch.mean( torch.abs(pred - gt) ,[1,2,3]).cpu().numpy()

def l2_metric(pred, gt):

    return torch.mean((pred - gt) ** 2,[1,2,3]).cpu().numpy()


def loss_map(loss_recon_map, loss_cluster_map):
    return torch.mean( loss_recon_map*loss_cluster_map  , [1,2,3] ).cpu().numpy()

def pairwise_l2_metric(deblur, recon,gt_static, gt_motion):
    deblur_loss_map = torch.mean((deblur - gt_static) ** 2,[1])
    pred_loss_map = torch.mean( (recon - gt_motion) ** 2,[1] )
    return torch.mean( deblur_loss_map*pred_loss_map,[1,2]).cpu().numpy()

def pixel_wise_l2_metric(deblur, pred_diff ,gt_static, gt_motion):
    # import pdb;pdb.set_trace()
    deblur_loss_map = torch.mean((deblur - gt_static) ** 2,[1])
    pred_loss_map = torch.mean( ( (pred_diff+gt_static) - gt_motion) ** 2,[1] )
    recon_loss_map = torch.mean( ( (pred_diff+deblur) - gt_motion) ** 2,[1] )
    return torch.mean( deblur_loss_map*pred_loss_map*recon_loss_map,[1,2]).cpu().numpy()


def maxpatch_metric(deblur, recon,gt_static, gt_motion, patch_num=4):
    b,c,h,w = deblur.shape
    h_ps = int(h//patch_num)
    w_ps = int(w//patch_num)

    deblur_loss_map = torch.mean((deblur - gt_static) ** 2,[1]).cpu()
    pred_loss_map = torch.mean( (recon - gt_motion) ** 2,[1] ).cpu()

    loss_map = deblur_loss_map * pred_loss_map
    
    max_deblur_patch =torch.zeros((b,1))
    max_pred_patch = torch.zeros((b,1))
    max_loss_patch = torch.zeros((b,1))
    for h_idx in range(patch_num):
        for w_idx in range(patch_num):

            deblur_loss_patch = torch.mean( deblur_loss_map[:,(h_idx)*h_ps:(h_idx+1)*h_ps,(w_idx)*w_ps:(w_idx+1)*w_ps] , [1,2] ).unsqueeze(1)
            pred_loss_patch = torch.mean( pred_loss_map[:,(h_idx)*h_ps:(h_idx+1)*h_ps,(w_idx)*w_ps:(w_idx+1)*w_ps] , [1,2] ).unsqueeze(1)
            loss_patch = torch.mean( loss_map[:,(h_idx)*h_ps:(h_idx+1)*h_ps,(w_idx)*w_ps:(w_idx+1)*w_ps], [1,2] ).unsqueeze(1)
            max_deblur_patch,_ =torch.max( torch.cat( [max_deblur_patch,deblur_loss_patch] , 1) ,1,keepdim=True) 
            max_pred_patch,_   =torch.max( torch.cat( [max_pred_patch  ,pred_loss_patch] , 1),1,keepdim=True) 
            max_loss_patch,_   =torch.max( torch.cat( [max_loss_patch  ,loss_patch] , 1),1,keepdim=True) 
    max_deblur_patch = max_deblur_patch.squeeze(-1)
    max_pred_patch = max_pred_patch.squeeze(-1)
    max_loss_patch = max_loss_patch.squeeze(-1)
    # import pdb;pdb.set_trace()
    return max_deblur_patch.cpu().numpy(), max_pred_patch.cpu().numpy(), max_loss_patch.cpu().numpy()


def mean_metric(distance_maps):
    axis =list(range(  distance_maps.dim() ))
    return torch.mean( distance_maps , axis[1:] ).cpu().numpy()

def min_max():

    return (video_loss - video_loss.min())/(video_loss.max()- video_loss.min())


from scipy.ndimage.filters import gaussian_filter1d
def max_min(video_loss):
    # video_loss = gaussian_filter1d(video_loss,2)
    return (video_loss - video_loss.min())/(video_loss.max()- video_loss.min())

def min_max_np(video_loss):

    video_loss = np.asarray(video_loss)
    # video_loss = gaussian_filter1d(video_loss,2)
    video_losses_max = np.max(video_loss, axis=0)
    video_losses_min = np.min(video_loss, axis=0)
    video_losses = np.ones(video_loss.shape) - (video_loss - video_losses_min * np.ones( video_loss.shape)) / (video_losses_max * np.ones(video_loss.shape)  - video_losses_min * np.ones( video_loss.shape))

    return 1.0- (video_loss - video_loss.min())/(video_loss.max()- video_loss.min())


def calcu_auc( y_score,  y_true):
    auc = roc_auc_score(y_true=y_true, y_score=y_score)
    return auc 


def calcu_result(video_results, labels_list, converse=False, norm_tag=True, smooth_tag=False):
    """
    raw result and labels list for all videos
    """
    video_idx = 0
    norm_result = []
    if norm_tag:
        for label_iter in labels_list:
            video_length = len(label_iter)
            if smooth_tag:
                video_results[video_idx : video_idx+video_length ] = gaussian_filter1d(video_results[video_idx : video_idx+video_length ],1)
            if converse:
                norm_result.append( max_min( video_results[video_idx : video_idx+video_length ] )  )
            else:
                norm_result.append( min_max_np( video_results[video_idx : video_idx+video_length ] )  )
            video_idx += video_length
        norm_result = np.concatenate(norm_result)
    else:
        if converse:
            norm_result = video_results
        else:
            norm_result = 1.0 - video_results
            
    labels = np.concatenate(labels_list)
    auc = calcu_auc(norm_result, labels)
    print(auc)

    return auc


def auc_metrics(video_results, labels_list):
    video_idx = 0
    norm_result = []
    for label_iter in labels_list:
        video_length = len(label_iter)
        norm_result.append( max_min( video_results[video_idx : video_idx+video_length ] )  )
        video_idx += video_length
    norm_result = np.concatenate(norm_result)
            
    labels = np.concatenate(labels_list)

    fpr, tpr, thresholds = metrics.roc_curve(labels, norm_result, pos_label=0)
    auc = metrics.auc(fpr, tpr)
    print(auc)

    return

def plot_result(video_results, labels_list, converse=False, norm_tag=True):
    video_idx = 0
    norm_result = []
    if norm_tag:
        for label_iter in labels_list:
            video_length = len(label_iter)
            if converse:
                norm_result.append( max_min( video_results[video_idx : video_idx+video_length ] )  )
            else:
                norm_result.append( min_max_np( video_results[video_idx : video_idx+video_length ] )  )
        # norm_result = np.concatenate(norm_result)
            video_idx += video_length
    else:
        if converse:
            norm_result = video_results
        else:
            norm_result = 1.0 - video_results
            
    # labels = np.concatenate(labels_list)
    labels = labels_list

    return norm_result, labels


def save_outputframes(batch_list, labels_list):

    for v_idx in labels_list:
        video_length = len( labels_list[v_idx] )
        frames = batch_list[video_length]

    return