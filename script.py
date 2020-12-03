import sys
import os
import glob
import numpy as np
from sklearn.cluster import KMeans
import scipy.io as scio
import time


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader


from datasets.dataset import ImageFolder
from datasets.eval_dataset import sliding_whole_dataset

from arch.solver.Solver_PredRes import Solver
from arch.model.PredRes_Model import PredRes_AE_Cluster_Model


import config

def train(args):
    config.log_path = os.path.join( args.log_path, args.dataset_name )
    # os.path.join('/mnt/data/ctmp/TensorFlow_Saver/ANORMLY/V3-Analysis/', 'Split_Prediction_and_Deblur_Cluster_Alpha_AE' , 'pub' , dataset_name ) 

    config.dataset_path = args.dataset_path
    config.dataset_name = args.dataset_name

    solver = Solver(config, cluster_model=PredRes_AE_Cluster_Model,model_type=model_type)

    
    train_set = ImageFolder(config.dataset_path, config.clips_length , config.frame_interval , config.img_size, config.rgb_tags)
    training_loader = DataLoader(dataset=train_set, num_workers=config.batch_size, batch_size=config.batch_size, shuffle=True)
    training_iter = iter( training_loader )
    training_nums = len(training_loader)

    
    ts_idx = 0
    if args.pretrain_tag:
        for iter_idx in range(pretrain_batches):
            if (ts_idx+2) >= training_nums:
                training_iter = iter( training_loader )
                ts_idx = 0            
            train_batch = next(training_iter)
            ts_idx+=1
            solver.train_batch_AE(train_batch)
            
            if (iter_idx+1)%50 == 0 :
                solver.training_info( 'Epoches: idx - {} '.format( iter_idx+1 ) )
    
    if args.ini_cluster_tag:
        solver.init_Cluster( iter( training_loader ), emmbeding_length=config.ini_embbeding_length)
    
    if args.finetune_tag:
        for iter_idx in range(total_batches):
            if (ts_idx+2) >= training_nums:
                training_iter = iter( training_loader )
                ts_idx = 0
            
            train_batch = next(training_iter)
            ts_idx+=1

            solver.train_batch_AE(train_batch)
            solver.train_batch_Cluster(train_batch)

            if (iter_idx+1)%50 == 0 :
                solver.training_info( ' Epoches: idx - {} '.format( iter_idx+1) )
            if (iter_idx+1)%5000 == 0:
                solver.save_model(iter_idx+1)

    return


def eval_model(dataset_path,dataset_name, log_dir ,model_type):
    config.log_path = os.path.join('/mnt/data/ctmp/TensorFlow_Saver/ANORMLY/V3-Analysis/', 'Split_Prediction_and_Deblur_Cluster_Alpha_AE' , 'pub' , dataset_name ) 
    # os.path.join(log_dir,dataset_name)
    # os.path.join('/mnt/data/ctmp/TensorFlow_Saver/ANORMLY/V3-Analysis/', 'Split_Prediction_and_Deblur_Cluster_Alpha_AE' , 'pub' , dataset_name ) 

    config.dataset_path = dataset_path
    config.dataset_name = dataset_name

    solver = Solver(config, cluster_model=PredRes_AE_Cluster_Model,model_type=model_type)
    
    solver.load_model()
    eval_loader, eval_labels = sliding_whole_dataset(config).generate_video_sequence()
    solver.para_tag = True
    solver.eval_datasets(eval_loader, eval_labels, 111)

    return


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--gpu', type=int, default=0, help="selected gpu idx")
    parser.add_argument('--dataset_name', type=str, default='avenue', choices=['ped2', 'avenue', 'shanghai_tech'], help="selected datasets")
    parser.add_argument('--dataset_path', type=str, default='avenue', help="datasets root path ")
    parser.add_argument('--log_path', type=str, default='./log', help="log dir ")


    parser.add_argument('--pretrain_tag', type=int, default=1, help="pre train ae model")
    parser.add_argument('--ini_cluster_tag', type=int, default=1, help="initial cluster centers vis K-means")
    parser.add_argument('--finetune_tag', type=int, default=1, help="train ae model and cluster centers together")

    parser.add_argument('--eval', type=int, default=0, help="evaluation")

    parser.add_argument('--model_type', type=str, default='res', help="model type")
    
    args = parser.parse_args()
    if args.eval:
        eval_model(args.dataset_path,args.dataset_name,args.log_path, args.model_type)
    else:
        train(args)