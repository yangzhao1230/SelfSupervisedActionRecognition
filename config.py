import os
from sacred import Experiment

ex = Experiment("MS2L")

@ex.config
def my_config():
    epoch_num = 2
    train_clip = 1.0
    label_clip = 1.0
    clip_gradient = 10
    hidden_size = 600
    lamb = 0.9
    weight_path = './output/model/pretrain.pt'
    log_path = './output/log/exp.log'
    train_mode = 'pretrain'
    gpus = "0"
    os.environ['CUDA_VISIBLE_DEVICES'] = gpus
    dataset = "NTU"
    batch_size = 128
    input_size = 150
    channel_num = 3
    person_num = 2
    joint_num = 25
    label_num = 60
    max_frame = 60
    train_list = './data/ntu/xsub/train_data_joint.npy'
    test_list = './data/ntu/xsub/val_data_joint.npy'
    train_label = './data/ntu/xsub/train_label.pkl'
    test_label = './data/ntu/xsub/val_label.pkl'
    train_frame = None
    test_frame = None
    max_frame = 300
    bn_size = 256
    lambd = 0.0051
    self = None
# %%
