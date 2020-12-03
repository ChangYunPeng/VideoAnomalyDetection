import numpy as np
import os
from scipy.io import loadmat
# import scipy.io.loadmat as loadmat

def ShanghaiTechCampus_frames_labels(root_path = 'DATASET/' ):
    label_list_path = os.path.join(root_path, 'testing/test_frame_mask' )
    label_path_list = os.listdir(label_list_path)
    label_path_list.sort()
    # for label_path_iter in label_path_list:
    #     print( label_path_iter )
    label_list = []
    videos_label_path = []
    for label_path in label_path_list:
        videos_label_path.append(os.path.join(label_list_path, label_path))
        label_iter = np.load(os.path.join(label_list_path, label_path))
        # print(label_iter.shape)
        label_list.append(label_iter)
    return label_list

def ped1_label(root_path = 'DATASET/'):
    import scipy.io as scio
    mat_file_path = os.path.join(root_path, 'ped1.mat')
    video_file_path = os.path.join(root_path, 'testing/frames')
    testing_frames_list = os.listdir( video_file_path )
    testing_frames_list.sort()
    label_raw = scio.loadmat(mat_file_path, squeeze_me=True)['gt']
    if not len(testing_frames_list) == label_raw.shape[0]:
        print('error')
        return
    video_num = len(testing_frames_list)
    video_gt = []
    for idx in range(video_num):
        tmp_video_path = os.path.join(video_file_path, testing_frames_list[idx] )
        video_length = len( os.listdir( tmp_video_path ) )
        sub_video_gt = np.zeros((video_length,), dtype=np.int8)
        abnormal_np = label_raw[idx]
        if len(abnormal_np.shape) == 1:
            abnormal_np = abnormal_np[:,np.newaxis]
        
        for ab_seg in range(abnormal_np.shape[1]):
            sub_video_gt[ abnormal_np[0,ab_seg] -1 : abnormal_np[1,ab_seg] ] = 1
        video_gt.append(sub_video_gt)    
    return video_gt

def ped2_label(root_path = 'DATASET/'):
    import scipy.io as scio
    mat_file_path = os.path.join(root_path, 'ped2.mat')
    video_file_path = os.path.join(root_path, 'testing/frames')
    testing_frames_list = os.listdir( video_file_path )
    testing_frames_list.sort()


    label_raw = scio.loadmat(mat_file_path, squeeze_me=True)['gt']
    if not len(testing_frames_list) == label_raw.shape[0]:
        print('error')
        return
    video_num = len(testing_frames_list)
    video_gt = []
    for idx in range(video_num):
        tmp_video_path = os.path.join(video_file_path, testing_frames_list[idx] )
        video_length = len( os.listdir( tmp_video_path ) )
        sub_video_gt = np.zeros((video_length,), dtype=np.int8)
        abnormal_np = label_raw[idx]
        if len(abnormal_np.shape) == 1:
            abnormal_np = abnormal_np[:,np.newaxis]
        
        for ab_seg in range(abnormal_np.shape[1]):
            sub_video_gt[ abnormal_np[0,ab_seg] -1 : abnormal_np[1,ab_seg] ] = 1
        video_gt.append(sub_video_gt)    
    return video_gt

def avenue_label(root_path = 'DATASET/'):
    import scipy.io as scio
    mat_file_path = os.path.join(root_path, 'avenue.mat')
    video_file_path = os.path.join(root_path, 'testing/frames')
    testing_frames_list = os.listdir( video_file_path )
    testing_frames_list.sort()
    label_raw = scio.loadmat(mat_file_path, squeeze_me=True)['gt']
    if not len(testing_frames_list) == label_raw.shape[0]:
        print('error')
        return
    video_num = len(testing_frames_list)
    video_gt = []
    for idx in range(video_num):
        tmp_video_path = os.path.join(video_file_path, testing_frames_list[idx] )
        video_length = len( os.listdir( tmp_video_path ) )
        sub_video_gt = np.zeros((video_length,), dtype=np.int8)
        abnormal_np = label_raw[idx]
        if len(abnormal_np.shape) == 1:
            abnormal_np = abnormal_np[:,np.newaxis]
        
        for ab_seg in range(abnormal_np.shape[1]):
            sub_video_gt[ abnormal_np[0,ab_seg] -1 : abnormal_np[1,ab_seg] ] = 1            
        video_gt.append(sub_video_gt)    
    return video_gt

def gather_datasets_labels(root_path='/mnt/data/DataSet/datasets/', dataset='shanghai_tech'):
    datasets_labels = {}
    datasets_labels['shanghai_tech'] = ShanghaiTechCampus_frames_labels
    datasets_labels['avenue'] = avenue_label
    datasets_labels['ped2'] = ped2_label

    return datasets_labels[dataset](root_path)

if __name__ == '__main__':
    # label = avenue_label('/mnt/data/DataSet/datasets/' )
    # print(label)
    # import pdb; pdb.set_trace()
    
    # print('tesr')
    datasets_labels = gather_datasets_labels()
    # print(datasets_labels[3])