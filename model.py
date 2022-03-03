from config import *
import torch.nn as nn
import torch
from itertools import permutations
import random

# def mask_empty_frame(X, frame_num):
#     batch = X.size(0)
#     time_step = X.size(1)
#     num_classes = X.size(2)
#
#     idx = torch.arange(0, time_step, 1).cuda().long().expand(batch, time_step)
#     frame_num_expand = frame_num.view(batch,1).repeat(1,time_step)
#     mask = (idx < frame_num_expand).float().view(batch, time_step, 1).repeat(1,1,num_classes)
#     X = X * mask
#     return X
#
# def mask_mean(X, frame_num):
#     X = mask_empty_frame(X, frame_num)
#     X = torch.sum(X, dim = 1)
#     eps = 0.01
#     frame_num = frame_num.view(-1,1).float() + eps
#     X = X / frame_num
#     return X

def init_weights(m):
    class_name=m.__class__.__name__

    if "Conv2d" in class_name or "Linear" in class_name:
        for name, param in m.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param.data)
            elif 'bias' in name:
                nn.init.constant_(param.data, 0.0)

    if "GRU" in class_name:
        for name, param in m.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                nn.init.constant_(param.data, 0.0)

# class TemporalMask:
#     @ex.capture
#     def __init__(self, mask_ratio, person_num, joint_num, channel_num):
#         self.mask_ratio = mask_ratio
#         self.person_num = person_num
#         self.joint_num = joint_num
#         self.channel_num = channel_num
#
#     def __call__(self, data, frame):
#         size, max_frame, feature_dim = data.shape
#         data = data.view(size, max_frame, self.person_num, self.joint_num, self.channel_num)
#         mask_idx = (frame * (1 - self.mask_ratio)).reshape((size, 1, 1, 1, 1)).repeat(1, max_frame, self.person_num, self.joint_num, self.channel_num)
#         frame_idx = torch.arange(max_frame).reshape((1, max_frame, 1, 1, 1)).repeat(size, 1, self.person_num, self.joint_num, self.channel_num)
#         mask = (frame_idx < mask_idx).float()
#         trans_data = data * mask
#         trans_data = trans_data.view(size, max_frame, feature_dim)
#         return trans_data
#
# class TemporalJigsaw:
#     @ex.capture
#     def __init__(self, seg_num, person_num, joint_num, channel_num):
#         self.seg_num = seg_num
#         self.person_num = person_num
#         self.joint_num = joint_num
#         self.channel_num = channel_num
#         self.permutations = list(permutations(list(range(self.seg_num))))
#         self.jig_num = len(self.permutations)
#
#     def __call__(self, data, frame):
#         size, max_frame, feature_dim = data.shape
#         data = data.view(size, max_frame, self.person_num, self.joint_num, self.channel_num)
#         trans_data = torch.zeros_like(data)
#         labels = []
#         for i in range(size):
#             length = frame[i] // self.seg_num
#             if length == 0:
#                 frame[i] = max_frame
#                 length = frame[i] // self.seg_num
#             f = lambda a: map(lambda b: a[b: b+length], range(0, len(a), length))
#             segments = list(f(list(range(0, frame[i]))))
#             label = random.randint(0, self.jig_num - 1)
#             shuffle_segments = []
#             for perm in self.permutations[label]:
#                 shuffle_segments.extend(segments[perm])
#             shuffle_frames = list(range(0, max_frame))
#             shuffle_frames[:len(shuffle_segments)] = shuffle_segments
#             trans_data[i] = data[i, shuffle_frames]
#             labels.append(label)
#         trans_data = trans_data.view(size, max_frame, feature_dim)
#         labels = torch.LongTensor(labels)
#         return trans_data, labels

# input_size MCV
# 2层GRU 600个units
class Encoder(nn.Module):
    @ex.capture
    def __init__(self, input_size, hidden_size):
        super(Encoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.gru = nn.GRU(
            input_size=self.input_size,
            hidden_size=self.hidden_size//2,
            num_layers=2,
            batch_first=True,
            #双向GRU
            bidirectional=True
        )
        self.apply(init_weights)

    def forward(self, X):
        self.gru.flatten_parameters()
        X, _ = self.gru(X)
        return X

class Decoder(nn.Module):
    @ex.capture
    def __init__(self, input_size, hidden_size):
        super(Decoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.decoder = nn.GRU(
            input_size=self.hidden_size,
            hidden_size=self.hidden_size,
            num_layers=2,
            batch_first=True
        )
        self.reconstruction = nn.Linear(self.hidden_size, self.input_size)

    def forward(self, X):
        self.decoder.flatten_parameters()
        X, _ = self.decoder(X)
        X = self.reconstruction(X)
        return X

class Linear(nn.Module):
    @ex.capture
    def __init__(self, hidden_size, label_num): 
        super(Linear, self).__init__()
        self.classifier = nn.Linear(hidden_size, label_num)
        self.apply(init_weights)

    def forward(self, X):
        X = self.classifier(X)
        return X

class Projector(nn.Module):
    @ex.capture
    def __init__(self, hidden_size, feature_num = 1024): 
        super(Projector, self).__init__()
        self.classifier = nn.Sequential(nn.Linear(hidden_size, 2048), nn.BatchNorm1d(2048),
                                        nn.ReLU(inplace=True), nn.Linear(2048, feature_num))
        self.apply(init_weights)

    def forward(self, X):
        X = self.classifier(X)
        return X

def off_diagonal(x):
    # return a flattened view of the off-diagonal elements of a square matrix
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


class BTwins(nn.Module):
    @ex.capture
    def __init__(self, max_frame, hidden_size, bn_size, lambd):
        super(BTwins, self).__init__()
        self.line = nn.Linear(max_frame * hidden_size, bn_size)
        self.bn = nn.BatchNorm1d(bn_size, affine=False)
        self.lambd = lambd
        self.apply(init_weights)

    def forward(self, feat1, feat2):
        N, T, CVM = feat1.shape
        feat1 = feat1.contiguous().view(N, T * CVM)
        feat2 = feat2.contiguous().view(N, T * CVM)
        feat1 = self.line(feat1)
        feat2 = self.line(feat2)
        c = self.bn(feat1).T @ self.bn(feat2)
        c.div_(N)
        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
        off_diag = off_diagonal(c).pow_(2).sum()

        # 超参数lanbd，需要调试
        BTloss = on_diag + self.lambd * off_diag

        return BTloss