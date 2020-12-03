import os

### Architecture
clips_length = 5
rgb_tags = True   # rgb_tags = False
img_channel = 3 if rgb_tags else 1
img_size = [256, 256]

static_layer_struct = [1,1,1]
motion_layer_struct = [1,1,1]
static_layer_nums = 3
motion_layer_nums = 3
cluster_num = 32


### training parameters
frame_interval = 1
batch_size = 4
pretrain_batches = 5000
ini_embbeding_length = 500
total_batches = 100000

### eval setting
eval_batches = 52*3
eval_workers = 1



