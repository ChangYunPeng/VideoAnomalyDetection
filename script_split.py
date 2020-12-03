import sys
import os
import glob
# from datasets_sequence import multi_train_datasets, multi_test_datasets
import torch
import time
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from model.trainer import TwoStreamTrainer
from model.trainer_dg import DG_StreamTrainer
from model.trainer_together import Simu_StreamTrainer, Simu_TwoStreamTrainer
from datasets.dataset import ImageFolder
from datasets.eval_dataset import sliding_basic_dataset, sliding_whole_dataset


from trainer.trainer_single import OnePrediction_StreamTrainer, OneReconstruction_StreamTrainer

# from model.OneStream import OneStream_Prediction_Model, OneStream_UNet
from arch.trainer.trainer_cluster import Prediction_Deblur_with_Cluster
from arch.trainer.trainer_cluster_alter import Prediction_Deblur_with_ConsineCluster
from arch.model.Cluster_Pred import ClusterPred_Model
from arch.trainer.trainer_split import Split_Prediction_Deblur
from arch.trainer.trainer_split_cluster import Split_Prediction_Deblur_with_Cluster

from arch.module.cluster import cluster_alpha

def train(dataset=2,gpu_idx=0, seq_tag=False, epoches = 40):
    import config
    from arch.model.Cluster_Pred import Recon_from_Cluster_PredResModel, Recon_from_Cluster_PredRes_U_Model
    from arch.model.Split_Model import Split_PredRes_U_Model

    os.environ['CUDA_VISIBLE_DEVICES'] = '{}'.format(gpu_idx)
    
    config.log_path = '/mnt/data/ctmp/TensorFlow_Saver/ANORMLY/V3-Analysis/'

    config.selected_dataset = dataset
    config.dataset  = config.selected_dataset
    # epoches = 40
    # config.img_size = 256
    config.img_size = config.imgsize_list[dataset]

    train_set = ImageFolder(config.dataset_root_path, config.folder_path[config.dataset],config.clips_length , config.frame_interval , config.img_size, config.rgb_tags)
    training_loader = DataLoader(dataset=train_set, num_workers=4, batch_size=4, shuffle=True)
    training_iter = iter( training_loader )
    training_nums = len(training_loader)
    total_batches = 1000000

    eval_loader, eval_labels = sliding_whole_dataset(config).generate_video_sequence()


    # test_loader, _ = sliding_whole_dataset(config).generate_video_sequence()
    # test_iter = iter(test_loader)
    # test_nums = len(test_loader)


    # log_root = os.path.join(config.log_path, 'Prediction_and_Deblur_Cluster') # terminal last # former 
    # log_root = os.path.join(config.log_path, 'Prediction_and_Deblur_Cluster_Alpha') # add alpha list 0.85
    log_root = os.path.join(config.log_path, 'Split_Prediction_and_Deblur_Cluster_Alpha') # add alpha list   #ped2 0.9658
    # pairwise_l2_metric ped2 0.969

    if seq_tag:
        log_root = os.path.join( log_root, 'seq' )

    config.log_path = log_root
    # os.path.join(log_root,'only')

    # Recon_from_Cluster_PredResModel
    # Recon_from_Cluster_PredRes_U_Model
    # trainer_only = Split_Prediction_Deblur(config, epoches=epoches, seq_tag = seq_tag , blur_ratio=3 ,cluster_model=Split_PredRes_U_Model, gray_layer_struct = [2,3,4,5],motion_layer_struct= [2,3,4,5],gray_layer_nums=4,motion_layer_nums=4,cluster_num = 32)
    trainer_only = Split_Prediction_Deblur(config, epoches=epoches, seq_tag = seq_tag , blur_ratio=4 ,cluster_model=Split_PredRes_U_Model, gray_layer_struct = [3,4,5],motion_layer_struct= [3,4,5],gray_layer_nums=3,motion_layer_nums=3,cluster_num = 32)
    # without res connecttion terminal 1
    ts_idx = 0
    es_idx = 0
    pre_train = 1000


    alphas = cluster_alpha()
    alphas = alphas[:3]
    trainer_only.load_model()
    # trainer_only.eval_datasets(eval_loader, eval_labels, 0)
    # trainer_only.demo_datasets(eval_loader, eval_labels, 0)

    # trainer_only.load_pretrain_model()

    # # training  Static
    # for iter_idx in range(total_batches):
    #     if (ts_idx+2) >= training_nums:
    #         training_iter = iter( training_loader )
    #         ts_idx = 0
        
    #     train_batch = next(training_iter)
    #     ts_idx+=1

    #     trainer_only.train_batch_Static(train_batch )
    #     if (iter_idx+1)%50 == 0 :
    #         trainer_only.training_info( 'Model-Only \t Epoches: idx - {}/{} '.format( iter_idx+1,total_batches) )
        
    #     if (iter_idx+1)%1000 == 0 and (iter_idx+1)>=pre_train:
    #         trainer_only.eval_Static_datasets(eval_loader, eval_labels, iter_idx)

    
    # training Motion
    for iter_idx in range(total_batches):
        if (ts_idx+2) >= training_nums:
            training_iter = iter( training_loader )
            ts_idx = 0
        
        train_batch = next(training_iter)
        ts_idx+=1

        alpha_level = int(iter_idx//100)
        if alpha_level+1>=len(alphas):
            alpha_level = len(alphas)-1
        trainer_only.train_batch_Motion(train_batch)
        if (iter_idx+1)%50 == 0 :
            trainer_only.training_info( 'Model-Only \t Epoches: idx - {}/{} '.format( iter_idx+1,total_batches) )
        
        if (iter_idx+1)%1000 == 0 and (iter_idx+1)>=5000:
            trainer_only.eval_Motion_datasets(eval_loader, eval_labels, iter_idx)

    
    # training fine-tune
    for iter_idx in range(total_batches):
        if (ts_idx+2) >= training_nums:
            training_iter = iter( training_loader )
            ts_idx = 0
        
        train_batch = next(training_iter)
        ts_idx+=1

        alpha_level = int(iter_idx//100)
        if alpha_level+1>=len(alphas):
            alpha_level = len(alphas)-1
        trainer_only.train_batch_Finetune(train_batch)
        if (iter_idx+1)%50 == 0 :
            trainer_only.training_info( 'Model-Only \t Epoches: idx - {}/{} '.format( iter_idx+1,total_batches) )
        
        if (iter_idx+1)%1000 == 0 and (iter_idx+1)>=5000:
            trainer_only.eval_datasets(eval_loader, eval_labels, iter_idx)

    return



def train_ae(dataset=2,gpu_idx=0, seq_tag=False, epoches = 40, static_tag=0, motion_tag=0 , model_type='sh'):
    import config
    from arch.model.Cluster_Pred import Recon_from_Cluster_PredResModel, Recon_from_Cluster_PredRes_U_Model
    from arch.model.Split_Model import Split_PredRes_AE_Model

    # os.environ['CUDA_VISIBLE_DEVICES'] = '{}'.format(gpu_idx)
    
    config.log_path = '/mnt/data/ctmp/TensorFlow_Saver/ANORMLY/V3-Analysis/'

    config.selected_dataset = dataset
    config.dataset  = config.selected_dataset
    # epoches = 40
    # config.img_size = 256
    # config.rgb_tags = False
    config.img_size =[256,256]
    #  [192,192]
    #  [256,256]
    # config.imgsize_list[dataset]
    # [192,192]
    # config.imgsize_list[dataset]

    train_set = ImageFolder(config.dataset_root_path, config.folder_path[config.dataset],config.clips_length , config.frame_interval , config.img_size, config.rgb_tags)
    training_loader = DataLoader(dataset=train_set, num_workers=8, batch_size=8, shuffle=True)
    training_iter = iter( training_loader )
    training_nums = len(training_loader)
    total_batches = 1000000

    eval_loader, eval_labels = sliding_whole_dataset(config).generate_video_sequence()


    # test_loader, _ = sliding_whole_dataset(config).generate_video_sequence()
    # test_iter = iter(test_loader)
    # test_nums = len(test_loader)


    # log_root = os.path.join(config.log_path, 'Prediction_and_Deblur_Cluster') # terminal last # former 
    # log_root = os.path.join(config.log_path, 'Prediction_and_Deblur_Cluster_Alpha') # add alpha list 0.85
    log_root = os.path.join(config.log_path, 'Split_Prediction_and_Deblur_AE') # add alpha list   #ped2 0.9658
    # pairwise_l2_metric ped2 0.969

    if seq_tag:
        log_root = os.path.join( log_root, 'seq' )

    if not model_type == 'sh':
        log_root = os.path.join( log_root, model_type )

    if not config.rgb_tags:
        log_root =os.path.join( log_root, 'gray' )

    # log_root = os.path.join( log_root, 'light' )

    config.log_path = log_root
    # os.path.join( log_root, 'light' )
    # os.path.join(log_root,'only')

    # Recon_from_Cluster_PredResModel
    # Recon_from_Cluster_PredRes_U_Model
    # trainer_only = Split_Prediction_Deblur(config, epoches=epoches, seq_tag = seq_tag , blur_ratio=3 ,cluster_model=Split_PredRes_U_Model, gray_layer_struct = [2,3,4,5],motion_layer_struct= [2,3,4,5],gray_layer_nums=4,motion_layer_nums=4,cluster_num = 32)
    # trainer_only = Split_Prediction_Deblur(config, epoches=epoches, seq_tag = seq_tag , blur_ratio=2 ,cluster_model=Split_PredRes_AE_Model, gray_layer_struct = [3,4,5],motion_layer_struct= [3,4,5],gray_layer_nums=3,motion_layer_nums=3,cluster_num = 32,model_type=model_type, static_neck_planes  = 32, motion_neck_planes  = 32)
    trainer_only = Split_Prediction_Deblur(config, epoches=epoches, seq_tag = seq_tag , blur_ratio=1 ,cluster_model=Split_PredRes_AE_Model, gray_layer_struct = [1,1,1],motion_layer_struct= [1,1,1],gray_layer_nums=3,motion_layer_nums=3,cluster_num = 32,model_type=model_type, static_neck_planes  = 32, motion_neck_planes  = 32)
    trainer_only.para_tag = False


    # config.log_path =  os.path.join( log_root, 'larger' )
    # trainer_larger = Split_Prediction_Deblur(config, epoches=epoches, seq_tag = seq_tag , blur_ratio=2 ,cluster_model=Split_PredRes_AE_Model, gray_layer_struct = [3,4,5],motion_layer_struct= [3,4,5],gray_layer_nums=3,motion_layer_nums=3,cluster_num = 32,model_type=model_type, static_neck_planes  = 64, motion_neck_planes  = 64)
    # trainer_larger.para_tag = False
    
    # config.log_path =  os.path.join( log_root, 'light' )
    # trainer_light = Split_Prediction_Deblur(config, epoches=epoches, seq_tag = seq_tag , blur_ratio=2 ,cluster_model=Split_PredRes_AE_Model, gray_layer_struct = [3,4,5],motion_layer_struct= [3,4,5],gray_layer_nums=3,motion_layer_nums=3,cluster_num = 32,model_type=model_type, static_neck_planes  = 16, motion_neck_planes  = 16)
    # trainer_light.para_tag = False



    ts_idx = 0
    es_idx = 0
    pre_train = 1000


    alphas = cluster_alpha()
    alphas = alphas[:3]

    try:
        trainer_only.load_static_model()
        # trainer_light.load_static_model()
    except Exception as e:
        print(e)

    try:
        trainer_only.load_motion_model()
        # trainer_light.load_motion_model()
    except Exception as e:
        print(e)
    
    trainer_only.eval_datasets(eval_loader, eval_labels, 0)
    # trainer_light.eval_datasets(eval_loader, eval_labels, 0)
    # trainer_larger.eval_datasets(eval_loader, eval_labels, 0)
    import pdb;pdb.set_trace()

    # trainer_only.load_model()
    # trainer_only.eval_datasets(eval_loader, eval_labels, 0)
    # trainer_only.demo_datasets(eval_loader, eval_labels, 0)

    # trainer_only.load_pretrain_model()
    
    # if static_tag:
    #     # training  Static
    #     try:
    #         trainer_only.load_static_model()
    #     except Exception as e:
    #         print(e)

    # if motion_tag:
    #     try:
    #         trainer_only.load_motion_model()
    #     except Exception as e:
    #         print(e)
    if static_tag or motion_tag:

        for iter_idx in range(total_batches):
            if (ts_idx+2) >= training_nums:
                training_iter = iter( training_loader )
                ts_idx = 0
            
            train_batch = next(training_iter)
            ts_idx+=1
            
            if static_tag:
                trainer_only.train_batch_Static(train_batch )
                # trainer_larger.train_batch_Static(train_batch )
                # trainer_light.train_batch_Static(train_batch )

            if motion_tag:
                trainer_only.train_batch_Motion(train_batch)
                # trainer_larger.train_batch_Motion(train_batch)
                # trainer_light.train_batch_Motion(train_batch)

            if (iter_idx+1)%50 == 0 :
                trainer_only.training_info( 'Model-Only \t Epoches: idx - {}/{} '.format( iter_idx+1,total_batches) )
                # trainer_larger.training_info( 'Model-large \t Epoches: idx - {}/{} '.format( iter_idx+1,total_batches) )
                # trainer_light.training_info( 'Model-light \t Epoches: idx - {}/{} '.format( iter_idx+1,total_batches) )
                
            
            if (iter_idx+1)%500 == 0 and (iter_idx+1)>=500:
                if static_tag:
                    trainer_only.eval_Static_datasets(eval_loader, eval_labels, iter_idx)
                    # trainer_larger.eval_Static_datasets(eval_loader, eval_labels, iter_idx)
                    # trainer_light.eval_Static_datasets(eval_loader, eval_labels, iter_idx)
                if motion_tag:
                    trainer_only.eval_Motion_datasets(eval_loader, eval_labels, iter_idx)
                    # trainer_larger.eval_Motion_datasets(eval_loader, eval_labels, iter_idx)
                    # trainer_light.eval_Motion_datasets(eval_loader, eval_labels, iter_idx)
    else:
        # if not static_tag and not motion_tag:
        trainer_only.load_model()
        # trainer_only.eval_datasets(eval_loader, eval_labels, 0)
        # training fine-tune
        for iter_idx in range(total_batches):
            if (ts_idx+2) >= training_nums:
                training_iter = iter( training_loader )
                ts_idx = 0
            
            train_batch = next(training_iter)
            ts_idx+=1

            alpha_level = int(iter_idx//100)
            if alpha_level+1>=len(alphas):
                alpha_level = len(alphas)-1
            trainer_only.train_batch_Finetune(train_batch)
            if (iter_idx+1)%50 == 0 :
                trainer_only.training_info( 'Model-Only \t Epoches: idx - {}/{} '.format( iter_idx+1,total_batches) )
            
            if (iter_idx+1)%1000 == 0 and (iter_idx+1)>=1000:
                trainer_only.eval_datasets(eval_loader, eval_labels, iter_idx)

    return



def train_with_cluster(dataset=2,gpu_idx=0, seq_tag=False, epoches = 40, static_tag=0, motion_tag=0,ini_cluster_tag=0,finetune_tag=0, model_type='sh'):

    import config
    from arch.model.Cluster_Pred import Recon_from_Cluster_PredResModel, Recon_from_Cluster_PredRes_U_Model
    from arch.model.Split_Model import Split_PredRes_AE_Cluster_Model
    import numpy as np
    from sklearn.cluster import KMeans
    import scipy.io as scio

    # os.environ['CUDA_VISIBLE_DEVICES'] = '{}'.format(gpu_idx)
    
    config.log_path = '/mnt/data/ctmp/TensorFlow_Saver/ANORMLY/V3-Analysis/'

    config.selected_dataset = dataset
    config.dataset  = config.selected_dataset
    config.img_size = config.imgsize_list[dataset]
    # epoches = 40
    # config.img_size = 256

    train_set = ImageFolder(config.dataset_root_path, config.folder_path[config.dataset],config.clips_length , config.frame_interval , config.img_size, config.rgb_tags)
    training_loader = DataLoader(dataset=train_set, num_workers=4, batch_size=4, shuffle=True)
    training_iter = iter( training_loader )
    training_nums = len(training_loader)
    total_batches = 1000000

    eval_loader, eval_labels = sliding_whole_dataset(config).generate_video_sequence()



    # log_root = os.path.join(config.log_path, 'Prediction_and_Deblur_Cluster') # terminal last # former 
    # log_root = os.path.join(config.log_path, 'Prediction_and_Deblur_Cluster_Alpha') # add alpha list 0.85
    log_root = os.path.join(config.log_path, 'Split_Prediction_and_Deblur_Cluster_Alpha_AE') # 

    if seq_tag:
        log_root = os.path.join( log_root, 'seq' )
    
    # log_root = os.path.join( log_root, model_type )

    # log_root
    # os.path.join(log_root,'only')

    # Recon_from_Cluster_PredResModel
    # Recon_from_Cluster_PredRes_U_Model
    config.log_path = os.path.join( log_root, model_type )
    # log_root
    # 
    trainer_only = Split_Prediction_Deblur_with_Cluster(config, epoches=epoches, seq_tag = seq_tag , blur_ratio=1 ,cluster_model=Split_PredRes_AE_Cluster_Model, gray_layer_struct = [1,1,1],motion_layer_struct= [1,1,1],gray_layer_nums=3,motion_layer_nums=3,cluster_num = 32,model_type=model_type)
    trainer_only.para_tag = True

    
    # config.log_path = os.path.join( log_root, 'shp' )
    # trainer_shp = Split_Prediction_Deblur_with_Cluster(config, epoches=epoches, seq_tag = seq_tag , blur_ratio=1 ,cluster_model=Split_PredRes_AE_Cluster_Model, gray_layer_struct = [1,1,1],motion_layer_struct= [1,1,1],gray_layer_nums=3,motion_layer_nums=3,cluster_num = 32,model_type='shp')
    # trainer_shp.para_tag = False

    ts_idx = 0
    es_idx = 0
    pre_train = 1000


    alphas = cluster_alpha()
    alphas = alphas[:3]



    # trainer_only.load_model(3499)
    
    # trainer_only.optimizer_static = optim.Adam( trainer_only.model.static_par ,  lr= 5e-5 )
    # trainer_only.optimizer_motion = optim.Adam( trainer_only.model.motion_par ,  lr= 5e-5  )

    # trainer_only.eval_datasets(eval_loader, eval_labels, 0)
    # trainer_only.demo_datasets(eval_loader, eval_labels, 0)

    # trainer_only.load_pretrain_model()



    
    # try:
    #     trainer_only.load_static_model()
    #     # trainer_only.optimizer_static = optim.Adam( trainer_only.model.static_par ,  lr=1e-6  )
    #     # trainer_shp.load_static_model()
    #     # trainer_light.load_static_model()
    # except Exception as e:
    #     print(e)

    # try:
    #     trainer_only.load_motion_model()
    #     # trainer_light.load_motion_model()
    # except Exception as e:
    #     print(e)
    
    # trainer_only.eval_datasets(eval_loader, eval_labels, 0)
    # import pdb;pdb.set_trace()

    start_epoches = 0
    # ped2
    # shanghai 21999 psnr 728s
    start_epoches = trainer_only.load_model(1) 
    start_epoches = start_epoches +1
    # import pdb;pdb.set_trace()

    # # cluster path 39999
    # start_epoches = trainer_only.load_model(1) 
    # start_epoches = start_epoches +1


    # trainer_only.load_static_model()
    trainer_only.eval_datasets(eval_loader, eval_labels, 1)
    import pdb; pdb.set_trace()
    trainer_only.optimizer_static = optim.Adam( trainer_only.model.static_par ,  lr=1e-5  )
    trainer_only.optimizer_motion = optim.Adam( trainer_only.model.motion_par ,  lr=1e-5  )


    # trainer_only.optimizer_static.param_groups[0]['lr'] = 1e-5
    # trainer_only.optimizer_motion.param_groups[0]['lr'] = 1e-5

    # # 3999 shanghai7251 shanghai7294
    # start_epoches = trainer_only.load_model(3999)
    # trainer_only.load_motion_model()
    # # trainer_only.eval_datasets(eval_loader, eval_labels, 0)
    # # import pdb; pdb.set_trace()
    # trainer_only.optimizer_static = optim.Adam( trainer_only.model.static_par ,  lr=1e-5  )
    # trainer_only.optimizer_motion = optim.Adam( trainer_only.model.motion_par ,  lr=1e-5  )

    # # trainer_light.eval_datasets(eval_loader, eval_labels, 0)
    # # trainer_larger.eval_datasets(eval_loader, eval_labels, 0)
    # import pdb;pdb.set_trace()

    if static_tag or motion_tag or finetune_tag:
        # if static_tag:
        #     try:
        #         trainer_only.load_static_model()
        #     except Exception as e:
        #         print(e)
        # if motion_tag:
        #     try:
        #         trainer_only.load_motion_model()
        #     except Exception as e:
        #         print(e)

        # training  Static
        for iter_idx in range(start_epoches, 100000):
            if (ts_idx+2) >= training_nums:
                training_iter = iter( training_loader )
                ts_idx = 0
            
            train_batch = next(training_iter)
            ts_idx+=1

            if static_tag:
                trainer_only.train_batch_Static(train_batch )
            if motion_tag:
                trainer_only.train_batch_Motion(train_batch )
                # trainer_shp.train_batch_Motion(train_batch )
            if finetune_tag:
                trainer_only.train_batch_Finetune(train_batch)
                # trainer_shp.train_batch_Finetune(train_batch)

            if (iter_idx+1)%50 == 0 :
                trainer_only.training_info( 'Model-Only \t Epoches: idx - {}/{} '.format( iter_idx+1,total_batches) ,False)
                # trainer_shp.training_info( 'Model-SHP \t Epoches: idx - {}/{} '.format( iter_idx+1,total_batches) ,False)
            
            if (iter_idx+1)%500 == 0 and (iter_idx+1)>=1000:
                # if static_tag:
                #     trainer_only.eval_Static_datasets(eval_loader, eval_labels, iter_idx)
                # if motion_tag:
                #     trainer_only.eval_Motion_datasets(eval_loader, eval_labels, iter_idx)
                
                trainer_only.eval_datasets(eval_loader, eval_labels, iter_idx)
                # trainer_shp.eval_datasets(eval_loader, eval_labels, iter_idx)



    if static_tag:
        try:
            trainer_only.load_static_model()
        except Exception as e:
            print(e)
    if motion_tag:
        try:
            trainer_only.load_motion_model()
        except Exception as e:
            print(e)

    if ini_cluster_tag:
        trainer_only.init_Cluster( iter( training_loader ), emmbeding_length=500)

    if finetune_tag:
        

        # training fine-tune
        for iter_idx in range(total_batches):
            if (ts_idx+2) >= training_nums:
                training_iter = iter( training_loader )
                ts_idx = 0
            
            train_batch = next(training_iter)
            ts_idx+=1

            alpha_level = int(iter_idx//100)
            if alpha_level+1>=len(alphas):
                alpha_level = len(alphas)-1
            
            trainer_only.train_batch_Finetune(train_batch)
            trainer_only.train_batch_Cluster(train_batch)

            if (iter_idx+1)%50 == 0 :
                trainer_only.training_info( 'Model-Only \t Epoches: idx - {}/{} '.format( iter_idx+1,total_batches) )
            
            if (iter_idx+1)%1000 == 0 and (iter_idx+1)>=pre_train:
                trainer_only.eval_datasets(eval_loader, eval_labels, iter_idx)


    # # ini Static cluster    
    # print('start initial cluster centers.......')
    # embeddings_bank = []
    # for iter_idx in range(500):
    #     if (ts_idx+2) >= training_nums:
    #         training_iter = iter( training_loader )
    #         ts_idx = 0
        
    #     train_batch = next(training_iter)
    #     ts_idx+=1

    #     embeddings_bank.append(trainer_only.init_Static_Cluster(train_batch))    
    #     # import pdb; pdb.set_trace()
    # embeddings_bank = np.concatenate(embeddings_bank,0)
    # kmeans_model = KMeans(n_clusters=trainer_only.cluster_num, init="k-means++").fit(embeddings_bank)
    # print(kmeans_model.cluster_centers_)

    # kmeans_cluster_centers = {}
    # kmeans_cluster_centers['mat'] = kmeans_model.cluster_centers_
    # scio.savemat( os.path.join( trainer_only.log_dir, 'km.mat'), kmeans_cluster_centers)

    # trainer_only.model.cluster_static.cluster_center.data = torch.from_numpy(kmeans_model.cluster_centers_).cuda()
    # trainer_only.save_model(1000)

    # trainer_only.load_model()

    
    # training  Static
    for iter_idx in range(10000):
        if (ts_idx+2) >= training_nums:
            training_iter = iter( training_loader )
            ts_idx = 0
        
        train_batch = next(training_iter)
        ts_idx+=1

        trainer_only.train_batch_Static( train_batch )
        trainer_only.train_batch_Static_Cluster( train_batch )

        if (iter_idx+1)%50 == 0 :
            trainer_only.training_info( 'Model-Only \t Epoches: idx - {}/{} '.format( iter_idx+1,total_batches) , True)
        
        if (iter_idx+1)%1000 == 0 and (iter_idx+1)>=pre_train:
            trainer_only.eval_Static_datasets(eval_loader, eval_labels, iter_idx)

    
    # # training Motion
    # for iter_idx in range(total_batches):
    #     if (ts_idx+2) >= training_nums:
    #         training_iter = iter( training_loader )
    #         ts_idx = 0
        
    #     train_batch = next(training_iter)
    #     ts_idx+=1

    #     alpha_level = int(iter_idx//100)
    #     if alpha_level+1>=len(alphas):
    #         alpha_level = len(alphas)-1
    #     trainer_only.train_batch_Motion(train_batch)
    #     if (iter_idx+1)%50 == 0 :
    #         trainer_only.training_info( 'Model-Only \t Epoches: idx - {}/{} '.format( iter_idx+1,total_batches) )
        
    #     if (iter_idx+1)%1000 == 0 and (iter_idx+1)>=pre_train:
    #         trainer_only.eval_Motion_datasets(eval_loader, eval_labels, iter_idx)

    
    # # training fine-tune
    # for iter_idx in range(total_batches):
    #     if (ts_idx+2) >= training_nums:
    #         training_iter = iter( training_loader )
    #         ts_idx = 0
        
    #     train_batch = next(training_iter)
    #     ts_idx+=1

    #     alpha_level = int(iter_idx//100)
    #     if alpha_level+1>=len(alphas):
    #         alpha_level = len(alphas)-1
    #     trainer_only.train_batch_Motion(train_batch)
    #     if (iter_idx+1)%50 == 0 :
    #         trainer_only.training_info( 'Model-Only \t Epoches: idx - {}/{} '.format( iter_idx+1,total_batches) )
        
    #     if (iter_idx+1)%1000 == 0 and (iter_idx+1)>=pre_train:
    #         trainer_only.eval_Motion_datasets(eval_loader, eval_labels, iter_idx)

    return


def train_ablation(dataset=2,gpu_idx=0, seq_tag=False, epoches = 40, static_tag=0, motion_tag=0,ini_cluster_tag=0,finetune_tag=0, model_type='sh'):


    import config
    from arch.model.Cluster_Pred import Recon_from_Cluster_PredResModel, Recon_from_Cluster_PredRes_U_Model
    from arch.model.Split_Model import Split_PredRes_AE_Cluster_Model
    import numpy as np
    from sklearn.cluster import KMeans
    import scipy.io as scio

    # os.environ['CUDA_VISIBLE_DEVICES'] = '{}'.format(gpu_idx)
    
    config.log_path = '/mnt/data/ctmp/TensorFlow_Saver/ANORMLY/V3-Analysis/'

    config.selected_dataset = dataset
    config.dataset  = config.selected_dataset
    config.img_size = config.imgsize_list[dataset]
    # epoches = 40
    # config.img_size = 256

    train_set = ImageFolder(config.dataset_root_path, config.folder_path[config.dataset],config.clips_length , config.frame_interval , config.img_size, config.rgb_tags)
    training_loader = DataLoader(dataset=train_set, num_workers=4, batch_size=4, shuffle=True)
    training_iter = iter( training_loader )
    training_nums = len(training_loader)
    total_batches = 1000000

    eval_loader, eval_labels = sliding_whole_dataset(config).generate_video_sequence()


    log_root = os.path.join(config.log_path, 'Split_Prediction_and_Deblur_Cluster_Alpha_AE') # 

    if seq_tag:
        log_root = os.path.join( log_root, 'seq' )
    
    log_root = os.path.join( log_root, 'ablation' )
    log_root = os.path.join( log_root, 'without_cl' )

    config.log_path = os.path.join( log_root, model_type )
    
    trainer_only = Split_Prediction_Deblur_with_Cluster(config, epoches=epoches, seq_tag = seq_tag , blur_ratio=1 ,cluster_model=Split_PredRes_AE_Cluster_Model, gray_layer_struct = [1,1,1],motion_layer_struct= [1,1,1],gray_layer_nums=3,motion_layer_nums=3,cluster_num = 8,model_type=model_type)
    trainer_only.para_tag = True
    
    # ts_idx = 0
    # for iter_idx in range(10000):
    #     if (ts_idx+2) >= training_nums:
    #         training_iter = iter( training_loader )
    #         ts_idx = 0
        
    #     train_batch = next(training_iter)
    #     ts_idx+=1

    #     trainer_only.train_batch_Finetune(train_batch)
    
    # trainer_only.save_model(iter_idx+1)
    trainer_only.load_model(10000)
    trainer_only.init_Cluster( iter( training_loader ), emmbeding_length=200)

    # trainer_only.fetch_cluster_rep( iter( training_loader ), emmbeding_length=200)

    # if ini_cluster_tag:
    #     trainer_only.init_Cluster( iter( training_loader ), emmbeding_length=500)


    # if finetune_tag:
    #     # training fine-tune
    #     for iter_idx in range(10000):
    #         if (ts_idx+2) >= training_nums:
    #             training_iter = iter( training_loader )
    #             ts_idx = 0
            
    #         train_batch = next(training_iter)
    #         ts_idx+=1

            
    #         trainer_only.train_batch_Finetune(train_batch)
    #         trainer_only.train_batch_Cluster(train_batch)

    #         if (iter_idx+1)%50 == 0 :
    #             trainer_only.training_info( 'Model-Only \t Epoches: idx - {}/{} '.format( iter_idx+1,total_batches) )
            
    #         if (iter_idx+1)%1000 == 0 and (iter_idx+1)>=1000:
    #             trainer_only.eval_datasets(eval_loader, eval_labels, iter_idx)
    #         if (iter_idx+1)%5000 == 0:
    #             trainer_only.save_model(iter_idx+1)

    # if static_tag or motion_tag:

    #     for iter_idx in range(total_batches):
    #         if (ts_idx+2) >= training_nums:
    #             training_iter = iter( training_loader )
    #             ts_idx = 0
            
    #         train_batch = next(training_iter)
    #         ts_idx+=1
            
    #         if static_tag:
    #             trainer_only.train_batch_Static(train_batch )
    #             # trainer_larger.train_batch_Static(train_batch )
    #             # trainer_light.train_batch_Static(train_batch )

    #         if motion_tag:
    #             trainer_only.train_batch_Motion(train_batch)
    #             # trainer_larger.train_batch_Motion(train_batch)
    #             # trainer_light.train_batch_Motion(train_batch)

    #         if (iter_idx+1)%50 == 0 :
    #             trainer_only.training_info( 'Model-Only \t Epoches: idx - {}/{} '.format( iter_idx+1,total_batches) )
    #             # trainer_larger.training_info( 'Model-large \t Epoches: idx - {}/{} '.format( iter_idx+1,total_batches) )
    #             # trainer_light.training_info( 'Model-light \t Epoches: idx - {}/{} '.format( iter_idx+1,total_batches) )
                
            
    #         if (iter_idx+1)%500 == 0 and (iter_idx+1)>=500:
    #             trainer_only.eval_datasets(eval_loader, eval_labels, iter_idx)
    #             # if static_tag:
    #             #     trainer_only.eval_Static_datasets(eval_loader, eval_labels, iter_idx)
    #             #     # trainer_larger.eval_Static_datasets(eval_loader, eval_labels, iter_idx)
    #             #     # trainer_light.eval_Static_datasets(eval_loader, eval_labels, iter_idx)
    #             # if motion_tag:
    #             #     trainer_only.eval_Motion_datasets(eval_loader, eval_labels, iter_idx)
    #             #     # trainer_larger.eval_Motion_datasets(eval_loader, eval_labels, iter_idx)
    #             #     # trainer_light.eval_Motion_datasets(eval_loader, eval_labels, iter_idx)

    return

def train_cluster(dataset=2,gpu_idx=0, seq_tag=False, epoches = 40, static_tag=0, motion_tag=0,ini_cluster_tag=0,finetune_tag=0, model_type='sh'):

    import config
    from arch.module.cluster import EuclidDistance_Assign_Module
    from arch.model.Cluster_Pred import Recon_from_Cluster_PredResModel, Recon_from_Cluster_PredRes_U_Model
    from arch.model.Split_Model import Split_PredRes_AE_Cluster_Model
    import numpy as np
    from sklearn.cluster import KMeans
    import scipy.io as scio

    # os.environ['CUDA_VISIBLE_DEVICES'] = '{}'.format(gpu_idx)
    
    config.log_path = '/mnt/data/ctmp/TensorFlow_Saver/ANORMLY/V3-Analysis/'

    config.selected_dataset = dataset
    config.dataset  = config.selected_dataset
    config.img_size = config.imgsize_list[dataset]
    # epoches = 40
    # config.img_size = 256

    train_set = ImageFolder(config.dataset_root_path, config.folder_path[config.dataset],config.clips_length , config.frame_interval , config.img_size, config.rgb_tags)
    training_loader = DataLoader(dataset=train_set, num_workers=2, batch_size=2, shuffle=True)
    training_iter = iter( training_loader )
    training_nums = len(training_loader)
    total_batches = 1000000



    # log_root = os.path.join(config.log_path, 'Prediction_and_Deblur_Cluster') # terminal last # former 
    # log_root = os.path.join(config.log_path, 'Prediction_and_Deblur_Cluster_Alpha') # add alpha list 0.85
    log_root = os.path.join(config.log_path, 'Split_Prediction_and_Deblur_Cluster_Alpha_AE') # 

    # if seq_tag:
    log_root = os.path.join( log_root, 'cluster' )
    
    # log_root = os.path.join( log_root, model_type )

    # log_root
    # os.path.join(log_root,'only')

    # Recon_from_Cluster_PredResModel
    # Recon_from_Cluster_PredRes_U_Model
    config.log_path = os.path.join( log_root, model_type )
    # log_root
    # 
    trainer_only = Split_Prediction_Deblur_with_Cluster(config, epoches=epoches, seq_tag = seq_tag , blur_ratio=1 ,cluster_model=Split_PredRes_AE_Cluster_Model, gray_layer_struct = [1,1,1],motion_layer_struct= [1,1,1],gray_layer_nums=3,motion_layer_nums=3,cluster_num = 32,model_type=model_type)
    trainer_only.para_tag = True

    
    # config.log_path = os.path.join( log_root, 'shp' )
    # trainer_shp = Split_Prediction_Deblur_with_Cluster(config, epoches=epoches, seq_tag = seq_tag , blur_ratio=1 ,cluster_model=Split_PredRes_AE_Cluster_Model, gray_layer_struct = [1,1,1],motion_layer_struct= [1,1,1],gray_layer_nums=3,motion_layer_nums=3,cluster_num = 32,model_type='shp')
    # trainer_shp.para_tag = False

    ts_idx = 0
    es_idx = 0
    pre_train = 1000


    alphas = cluster_alpha()
    alphas = alphas[:3]



    # start_epoches = 0
    # # ped2
    # # shanghai 21999 psnr 728s
    # start_epoches = trainer_only.load_model(1) 
    # start_epoches = start_epoches +1
    # # import pdb;pdb.set_trace()
    
    # trainer_only.model.cluster = EuclidDistance_Assign_Module( trainer_only.model.inter_planes+trainer_only.model.inter_planes, cluster_num=trainer_only.model.cluster_num, soft_assign_alpha=25.0 )
    # trainer_only.save_model(1)
    

    # trainer_only.load_static_model()
    # trainer_only.eval_datasets(eval_loader, eval_labels, 2)
    # import pdb; pdb.set_trace()
    
    # start_epoches = trainer_only.load_model(1001) 
    # start_epoches = trainer_only.load_model(3) 

    
    # cluster path 39999
    # start_epoches = trainer_only.load_model(39999) 
    start_epoches = trainer_only.load_model(10) 
    start_epoches = start_epoches +1


    
    # kmeans_cluster_centers = scio.loadmat( os.path.join( trainer_only.log_dir, 'km.mat'))

    # # kmeans_cluster_centers = {}
    # # kmeans_cluster_centers['mat'] = kmeans_model.cluster_centers_
    # # scio.savemat( os.path.join( self.log_dir, 'km.mat'), kmeans_cluster_centers)
    
    # trainer_only.model.cluster.cluster_center.data = torch.from_numpy(kmeans_cluster_centers['mat']).cuda()
    # trainer_only.save_model(3)

    trainer_only.optimizer_static = optim.Adam( trainer_only.model.static_par ,  lr=1e-5  )
    trainer_only.optimizer_motion = optim.Adam( trainer_only.model.motion_par ,  lr=1e-5  )
    trainer_only.optimizer = optim.Adam( trainer_only.model.parameters() , lr=5e-5)
    # trainer_only.eval_datasets(eval_loader, eval_labels, 4)

    if ini_cluster_tag:
        trainer_only.init_Cluster( iter( training_loader ), emmbeding_length=500)



    eval_loader, eval_labels = sliding_whole_dataset(config).generate_video_sequence()
    # trainer_only.load_static_model()
    trainer_only.eval_datasets(eval_loader, eval_labels, 39)
    import pdb; pdb.set_trace()

    if finetune_tag:
        

        # training fine-tune
        for iter_idx in range(total_batches):
            if (ts_idx+2) >= training_nums:
                training_iter = iter( training_loader )
                ts_idx = 0
            
            train_batch = next(training_iter)
            ts_idx+=1

            alpha_level = int(iter_idx//100)
            if alpha_level+1>=len(alphas):
                alpha_level = len(alphas)-1
            
            # trainer_only.train_batch_Finetune(train_batch)
            trainer_only.train_batch_Cluster(train_batch)

            if (iter_idx+1)%50 == 0 :
                trainer_only.training_info( 'Model-Only \t Epoches: idx - {}/{} '.format( iter_idx+1,total_batches) )
            
            if (iter_idx+1)%1000 == 0 and (iter_idx+1)>=pre_train:
                trainer_only.eval_datasets(eval_loader, eval_labels, iter_idx)
            if (iter_idx+1)%5000 == 0:
                trainer_only.save_model(iter_idx+1)



    return

def train_with_concise(dataset=2,gpu_idx=0, seq_tag=False, epoches = 40, static_tag=0, motion_tag=0,ini_cluster_tag=0,finetune_tag=0, model_type='sh'):

    import config
    from arch.model.Cluster_Pred import Recon_from_Cluster_PredResModel, Recon_from_Cluster_PredRes_U_Model
    from arch.model.Split_Model import Split_PredRes_AE_Cluster_Model
    import numpy as np
    from sklearn.cluster import KMeans
    import scipy.io as scio
    
    config.log_path = '/mnt/data/ctmp/TensorFlow_Saver/ANORMLY/V3-Analysis/'

    config.selected_dataset = dataset
    config.dataset  = config.selected_dataset
    config.img_size = config.imgsize_list[dataset]
    # epoches = 40
    # config.img_size = 256

    train_set = ImageFolder(config.dataset_root_path, config.folder_path[config.dataset],config.clips_length , config.frame_interval , config.img_size, config.rgb_tags)
    training_loader = DataLoader(dataset=train_set, num_workers=16*3, batch_size=16*3, shuffle=True)
    training_iter = iter( training_loader )
    training_nums = len(training_loader)
    total_batches = 1000000

    # eval_loader, eval_labels = sliding_whole_dataset(config).generate_video_sequence()
    log_root = os.path.join(config.log_path, 'Split_Prediction_and_Deblur_Cluster_Alpha_AE') # 

    if seq_tag:
        log_root = os.path.join( log_root, 'seq' )
    
    config.log_path = os.path.join( log_root, model_type )
    trainer_only = Split_Prediction_Deblur_with_Cluster(config, epoches=epoches, seq_tag = seq_tag , blur_ratio=1 ,cluster_model=Split_PredRes_AE_Cluster_Model, gray_layer_struct = [1,1,1],motion_layer_struct= [1,1,1],gray_layer_nums=3,motion_layer_nums=3,cluster_num = 32,model_type=model_type)
    trainer_only.para_tag = True

    ts_idx = 0
    es_idx = 0
    pre_train = 1000


    alphas = cluster_alpha()
    alphas = alphas[:3]
    
    try:
        trainer_only.load_static_model()
        trainer_only.optimizer_static = optim.Adam( trainer_only.model.static_par ,  lr=1e-5  )
        # trainer_shp.load_static_model()
        # trainer_light.load_static_model()
    except Exception as e:
        print(e)

    # try:
    #     # trainer_only.load_motion_model()
    #     trainer_only.optimizer_motion = optim.Adam( trainer_only.model.motion_par ,  lr=1e-4  )
    #     # trainer_light.load_motion_model()
    # except Exception as e:
    #     print(e)
    
    # trainer_only.eval_datasets(eval_loader, eval_labels, 0)
    # # trainer_light.eval_datasets(eval_loader, eval_labels, 0)
    # # trainer_larger.eval_datasets(eval_loader, eval_labels, 0)
    # import pdb;pdb.set_trace()

    # trainer_only.save_motion_model(1)

    if static_tag or motion_tag or finetune_tag:

        # training  Static
        for iter_idx in range(200000,500000):
            if (ts_idx+2) >= training_nums:
                training_iter = iter( training_loader )
                ts_idx = 0
            
            train_batch = next(training_iter)
            ts_idx+=1

            if static_tag:
                trainer_only.train_batch_Static(train_batch )
            if motion_tag:
                trainer_only.train_batch_Motion(train_batch )
            if finetune_tag:
                trainer_only.train_batch_Finetune(train_batch)

            if (iter_idx+1)%50 == 0 :
                trainer_only.training_info( 'Model-Only \t Epoches: idx - {}/{} '.format( iter_idx+1,total_batches) ,False)
            
            if (iter_idx+1)%5000 == 0 and (iter_idx+1)>=1000:
                # trainer_only.eval_datasets(eval_loader, eval_labels, iter_idx)
                # trainer_only.save_model(iter_idx+1)
                # trainer_only.eval_Static_datasets(eval_loader, eval_labels, iter_idx)
                if static_tag:
                    trainer_only.save_static_model(iter_idx+1)
                if motion_tag:
                    trainer_only.save_motion_model(iter_idx+1)
    return

def eval_with_concise(dataset=2,gpu_idx=0, seq_tag=False, epoches = 40, static_tag=0, motion_tag=0,ini_cluster_tag=0,finetune_tag=0, model_type='sh'):

    import config
    from arch.model.Cluster_Pred import Recon_from_Cluster_PredResModel, Recon_from_Cluster_PredRes_U_Model
    from arch.model.Split_Model import Split_PredRes_AE_Cluster_Model
    import numpy as np
    from sklearn.cluster import KMeans
    import scipy.io as scio
    
    config.log_path = '/mnt/data/ctmp/TensorFlow_Saver/ANORMLY/V3-Analysis/'

    config.selected_dataset = dataset
    config.dataset  = config.selected_dataset
    config.img_size = config.imgsize_list[dataset]
    # epoches = 40
    # config.img_size = 256

    # train_set = ImageFolder(config.dataset_root_path, config.folder_path[config.dataset],config.clips_length , config.frame_interval , config.img_size, config.rgb_tags)
    # training_loader = DataLoader(dataset=train_set, num_workers=16*3, batch_size=16*3, shuffle=True)
    # training_iter = iter( training_loader )
    # training_nums = len(training_loader)
    # total_batches = 1000000

    eval_loader, eval_labels = sliding_whole_dataset(config).generate_video_sequence()
    log_root = os.path.join(config.log_path, 'Split_Prediction_and_Deblur_Cluster_Alpha_AE') # 

    if seq_tag:
        log_root = os.path.join( log_root, 'seq' )
    log_root = os.path.join( log_root, 'cluster' )
    
    config.log_path = os.path.join( log_root, model_type )
    trainer_only = Split_Prediction_Deblur_with_Cluster(config, epoches=epoches, seq_tag = seq_tag , blur_ratio=1 ,cluster_model=Split_PredRes_AE_Cluster_Model, gray_layer_struct = [1,1,1],motion_layer_struct= [1,1,1],gray_layer_nums=3,motion_layer_nums=3,cluster_num = 32,model_type=model_type)
    trainer_only.para_tag = True

    ts_idx = 0
    es_idx = 0
    pre_train = 1000


    model_list = glob.glob( os.path.join(trainer_only.log_dir, 'E'+'*'+'_model.pth') )
    print(model_list)
    # import pdb; pdb.set_trace()

    # trainer_only.load_model(1999)
    # trainer_only.load_motion_model('motion_model6811.pth')
    # trainer_only.load_static_model(117000)
    # trainer_only.load_static_model(110500)
    # motion_model6797_psnr37.pth
    for idx in range(len(model_list)):
        try:
            model_idx = int( os.path.basename(model_list[idx]).split('_model')[0].split('E')[1] )
            trainer_only.load_model(model_idx)
            trainer_only.eval_datasets(eval_loader, eval_labels, 111)
            # import pdb; pdb.set_trace()
        except Exception as e:
            print(e)

    # static_model_list = glob.glob( os.path.join(config.log_path, config.dataset_list[config.dataset] , '*'+'_static_model.pth') )
    # motion_model_list = glob.glob( os.path.join(config.log_path, config.dataset_list[config.dataset] ,'*'+'_motion_model.pth') )
    # print(config.log_path)
    # print(static_model_list)
    # print(motion_model_list)
    # idx = 10

    # static_model_idxs = []
    # motion_model_idxs = []
    
    # for static_model_path_iter in static_model_list:
    #     static_model_idx = int( os.path.basename(static_model_path_iter).split('_static_model')[0].split('E')[1] )
    #     static_model_idxs.append(static_model_idx)

    # for motion_model_path_iter in motion_model_list:
    #     motion_model_idx = int( os.path.basename(motion_model_path_iter).split('_motion_model')[0].split('E')[1] )
    #     motion_model_idxs.append(motion_model_idx)
    
    # static_model_idxs.sort()
    # motion_model_idxs.sort()
    
    # static_model_idxs = static_model_idxs[::-1]
    # motion_model_idxs = motion_model_idxs[::-1]

    # # print(static_model_idxs)
    # for static_model_idx in static_model_idxs:
    #     trainer_only.load_static_model(static_model_idx)
    #     for motion_model_idx in motion_model_idxs:
    #         trainer_only.load_motion_model(motion_model_idx)
    #         trainer_only.eval_datasets(eval_loader, eval_labels, idx)
    #         idx+=1
            

    # alphas = cluster_alpha()
    # alphas = alphas[:3]
    
    # # try:
    # #     trainer_only.load_static_model()
    # #     # trainer_only.optimizer_static = optim.Adam( trainer_only.model.static_par ,  lr=1e-6  )
    # #     # trainer_shp.load_static_model()
    # #     # trainer_light.load_static_model()
    # # except Exception as e:
    # #     print(e)

    # try:
    #     trainer_only.load_motion_model()
    #     # trainer_light.load_motion_model()
    # except Exception as e:
    #     print(e)
    
    # # trainer_only.eval_datasets(eval_loader, eval_labels, 0)
    # # # trainer_light.eval_datasets(eval_loader, eval_labels, 0)
    # # # trainer_larger.eval_datasets(eval_loader, eval_labels, 0)
    # # import pdb;pdb.set_trace()

    # trainer_only.save_motion_model(1)

    # if static_tag or motion_tag or finetune_tag:

    #     # training  Static
    #     for iter_idx in range(100000):
    #         if (ts_idx+2) >= training_nums:
    #             training_iter = iter( training_loader )
    #             ts_idx = 0
            
    #         train_batch = next(training_iter)
    #         ts_idx+=1

    #         if static_tag:
    #             trainer_only.train_batch_Static(train_batch )
    #         if motion_tag:
    #             trainer_only.train_batch_Motion(train_batch )
    #         if finetune_tag:
    #             trainer_only.train_batch_Finetune(train_batch)

    #         if (iter_idx+1)%50 == 0 :
    #             trainer_only.training_info( 'Model-Only \t Epoches: idx - {}/{} '.format( iter_idx+1,total_batches) ,False)
            
    #         if (iter_idx+1)%500 == 0 and (iter_idx+1)>=1000:
    #             # trainer_only.eval_datasets(eval_loader, eval_labels, iter_idx)
    #             # trainer_only.save_model(iter_idx+1)
    #             trainer_only.save_static_model(iter_idx+1)
    #             trainer_only.save_motion_model(iter_idx+1)
    return


def process_pth(dataset=2,gpu_idx=0, seq_tag=False, epoches = 40, static_tag=0, motion_tag=0,ini_cluster_tag=0,finetune_tag=0, model_type='sh'):
    from arch.module.cluster import EuclidDistance_Assign_Module
    from arch.model.Cluster_Pred import Recon_from_Cluster_PredResModel, Recon_from_Cluster_PredRes_U_Model
    from arch.model.Split_Model import Split_PredRes_AE_Cluster_Model
    import numpy as np
    from sklearn.cluster import KMeans
    import scipy.io as scio

    import config
    config.log_path = '/mnt/data/ctmp/TensorFlow_Saver/ANORMLY/V3-Analysis/'

    config.selected_dataset = dataset
    config.dataset  = config.selected_dataset
    config.img_size = config.imgsize_list[dataset]
    # epoches = 40
    # config.img_size = 256

    # train_set = ImageFolder(config.dataset_root_path, config.folder_path[config.dataset],config.clips_length , config.frame_interval , config.img_size, config.rgb_tags)
    # training_loader = DataLoader(dataset=train_set, num_workers=2, batch_size=2, shuffle=True)
    # training_iter = iter( training_loader )
    # training_nums = len(training_loader)
    # total_batches = 1000000

    # eval_loader, eval_labels = sliding_whole_dataset(config).generate_video_sequence()



    # log_root = os.path.join(config.log_path, 'Prediction_and_Deblur_Cluster') # terminal last # former 
    # log_root = os.path.join(config.log_path, 'Prediction_and_Deblur_Cluster_Alpha') # add alpha list 0.85
    log_root = os.path.join(config.log_path, 'Split_Prediction_and_Deblur_Cluster_Alpha_AE') # 

    # if seq_tag:
    log_root = os.path.join( log_root, 'cluster' )
    
    # log_root = os.path.join( log_root, model_type )

    # log_root
    # os.path.join(log_root,'only')

    # Recon_from_Cluster_PredResModel
    # Recon_from_Cluster_PredRes_U_Model

    load_path='/mnt/data/ctmp/TensorFlow_Saver/ANORMLY/V3-Analysis/Split_Prediction_and_Deblur_Cluster_Alpha_AE/cluster/res/avenue/E{}_model.pth'

    # state = torch.load( load_path.format(24999) )
    config.log_path = os.path.join( log_root, model_type )

    trainer_only = Split_Prediction_Deblur_with_Cluster(config, epoches=epoches, seq_tag = seq_tag , blur_ratio=1 ,cluster_model=Split_PredRes_AE_Cluster_Model, gray_layer_struct = [1,1,1],motion_layer_struct= [1,1,1],gray_layer_nums=3,motion_layer_nums=3,cluster_num = 32,model_type=model_type)
    # trainer_only.model.cluster_static = EuclidDistance_Assign_Module( trainer_only.model.inter_planes, cluster_num=trainer_only.model.cluster_num, soft_assign_alpha=25.0 )
    # trainer_only.model.cluster = EuclidDistance_Assign_Module( trainer_only.model.inter_planes, cluster_num=trainer_only.model.cluster_num, soft_assign_alpha=25.0 )
    # trainer_only.model.load_state_dict(state['state_dict'])
    
    trainer_only.load_model(24999)


    trainer_only.model.cluster = EuclidDistance_Assign_Module( trainer_only.model.inter_planes+trainer_only.model.inter_planes, cluster_num=trainer_only.model.cluster_num, soft_assign_alpha=25.0 )
    del trainer_only.model.cluster_static, trainer_only.model.cluster_motion
    trainer_only.save_model(10)
    # state = torch.load( load_path.format(10) )
    trainer_only.load_model(10)

    # for keys in state['state_dict'].keys():
    #     if 'cluster' in keys:
    #         print(keys)

    return



def demo_with_cluster(dataset=2,gpu_idx=0, seq_tag=False, epoches = 40, static_tag=0, motion_tag=0,ini_cluster_tag=0,finetune_tag=0, model_type='sh'):

    import config
    from arch.model.Cluster_Pred import Recon_from_Cluster_PredResModel, Recon_from_Cluster_PredRes_U_Model
    from arch.model.Split_Model import Split_PredRes_AE_Cluster_Model
    import numpy as np
    from sklearn.cluster import KMeans
    import scipy.io as scio

    # os.environ['CUDA_VISIBLE_DEVICES'] = '{}'.format(2)
    eval_batches = 32*3
    
    config.log_path = '/mnt/data/ctmp/TensorFlow_Saver/ANORMLY/V3-Analysis/'

    config.selected_dataset = dataset
    config.dataset  = config.selected_dataset
    config.img_size = config.imgsize_list[dataset]
    # epoches = 40
    # config.img_size = 256

    # train_set = ImageFolder(config.dataset_root_path, config.folder_path[config.dataset],config.clips_length , config.frame_interval , config.img_size, config.rgb_tags)
    # training_loader = DataLoader(dataset=train_set, num_workers=4*2, batch_size=4*2, shuffle=True)
    # training_iter = iter( training_loader )
    # training_nums = len(training_loader)
    # total_batches = 1000000




    # log_root = os.path.join(config.log_path, 'Prediction_and_Deblur_Cluster') # terminal last # former 
    # log_root = os.path.join(config.log_path, 'Prediction_and_Deblur_Cluster_Alpha') # add alpha list 0.85
    log_root = os.path.join(config.log_path, 'Split_Prediction_and_Deblur_Cluster_Alpha_AE') # 

    if seq_tag:
        log_root = os.path.join( log_root, 'seq' )
    
    # log_root = os.path.join( log_root, model_type )

    # log_root
    # os.path.join(log_root,'only')

    # Recon_from_Cluster_PredResModel
    # Recon_from_Cluster_PredRes_U_Model
    config.log_path = os.path.join( log_root ,'cluster', model_type)
    trainer_only = Split_Prediction_Deblur_with_Cluster(config, epoches=epoches, seq_tag = seq_tag , blur_ratio=1 ,cluster_model=Split_PredRes_AE_Cluster_Model, gray_layer_struct = [1,1,1],motion_layer_struct= [1,1,1],gray_layer_nums=3,motion_layer_nums=3,cluster_num = 32,model_type=model_type)
    trainer_only.para_tag = True

    ts_idx = 0
    es_idx = 0
    pre_train = 1000


    alphas = cluster_alpha()
    alphas = alphas[:3]


    trainer_only.load_model(10)
    
    # trainer_only.fetch_cluster_rep( iter( training_loader ), emmbeding_length=200)

    # import pdb; pdb.set_trace()
    eval_loader, eval_labels = sliding_whole_dataset(config).generate_video_sequence()
    trainer_only.demo_datasets(eval_loader, eval_labels, 0)


    return

if __name__ == "__main__":
    # train_UNet()
    # train()
    import argparse

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--gpu', type=int, default=0, help="selected gpu idx")
    parser.add_argument('--dataset', type=int, default=0, help="selected datasets")
    parser.add_argument('--epoches', type=int, default=40, help="training epoches")


    parser.add_argument('--seq', type=int, default=0, help="image size")
    parser.add_argument('--recon', type=int, default=0, help="image size")
    parser.add_argument('--ae', type=int, default=0, help="image size")
    parser.add_argument('--cluster', type=int, default=0, help="image size")
    parser.add_argument('--demo', type=int, default=0, help="image size")

    parser.add_argument('--st_tag', type=int, default=0, help="training static model")
    parser.add_argument('--mo_tag', type=int, default=0, help="training motion model")
    parser.add_argument('--ini_tag', type=int, default=0, help="training motion model")
    parser.add_argument('--cluster_tag', type=int, default=0, help="training motion model")
    parser.add_argument('--finetune_tag', type=int, default=0, help="training motion model")



    
    parser.add_argument('--model_type', type=str, default='sh', help="model type")

    args = parser.parse_args()
    
    if args.demo:
        demo_with_cluster(args.dataset,args.gpu, args.seq, args.epoches,args.st_tag, args.mo_tag, args.ini_tag, args.finetune_tag, model_type= args.model_type)
        import pdb; pdb.set_trace()

    if args.ae:
        train_ae(args.dataset,args.gpu, args.seq, args.epoches,args.st_tag, args.mo_tag, args.model_type)
    # elif args.recon:
    #     train(args.dataset,args.gpu, args.seq, args.epoches)
    # else:
    # process_pth(args.dataset,args.gpu, args.seq, args.epoches,args.st_tag, args.mo_tag, args.ini_tag, args.finetune_tag, model_type= args.model_type)
    if args.cluster:
        train_ablation(args.dataset,args.gpu, args.seq, args.epoches,args.st_tag, args.mo_tag, args.ini_tag, args.finetune_tag, model_type= args.model_type)
        # train_cluster(args.dataset,args.gpu, args.seq, args.epoches,args.st_tag, args.mo_tag, args.ini_tag, args.finetune_tag, model_type= args.model_type)
        # train_with_cluster(args.dataset,args.gpu, args.seq, args.epoches,args.st_tag, args.mo_tag, args.ini_tag, args.finetune_tag, model_type= args.model_type)


        # train_with_concise(args.dataset,args.gpu, args.seq, args.epoches,args.st_tag, args.mo_tag, args.ini_tag, args.finetune_tag, model_type= args.model_type)
        # eval_with_concise(args.dataset,args.gpu, args.seq, args.epoches,args.st_tag, args.mo_tag, args.ini_tag, args.finetune_tag, model_type= args.model_type)
        # dataset=2,gpu_idx=0, seq_tag=False, epoches = 40, static_tag=0, motion_tag=0,ini_cluster_tag=0,finetune_tag=0, model_type='sh'