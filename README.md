
#  Video Anomaly Detection with Spatio-Temporal Dissociation


## Prerequisites

The code is built with following libraries:
opencv-python
Pillow
scikit-learn
scipy
sklearn
pytorch
tqdm

## Data Preparation
We have trained on [Ped2](http://visal.cs.cityu.edu.hk/downloads/ucsdpeds-vids/), [Avenue](http://www.cse.cuhk.edu.hk/leojia/projects/detectabnormal/dataset.html) and [ShanghaiTech](https://svip-lab.github.io/dataset/campus_dataset.html)

## Training

```bash
python script.py 
    --pretrain_tag 1
    --ini_cluster_tag 1
    --finetune_tag 1
    --log_path log_dir
    --dataset_path datasets_dir/ShanghaiTechCampus  
    --dataset_name shanghai_tech 
    --eval 0
```

## Evaluation

```bash
python script.py 
    --log_path log_dir
    --dataset_path datasets_dir/ShanghaiTechCampus  
    --dataset_name shanghai_tech 
    --eval 1
```

## Pretrained Model Weights
We also provide the checkpoints of [Ped2](https://whueducn-my.sharepoint.com/:f:/g/personal/changyunpeng_whu_edu_cn/Ekj22tBWW5FFnBIHmkrKulABy3NZAGn1P5rOyeuZWe-Lsg?e=a5pT4d), [Avenue](https://whueducn-my.sharepoint.com/:f:/g/personal/changyunpeng_whu_edu_cn/Ek56w93uJuFBlqr6R1UYARsB3Ax04085N5_xU_W1CM-prg?e=W60cRd) and [ShanghaiTech](https://whueducn-my.sharepoint.com/:f:/g/personal/changyunpeng_whu_edu_cn/EsvhOLRPo3dPtW92jHZ9SF4BlwxLUqtub8vPlP-FHieXRA?e=EoS5Db).

