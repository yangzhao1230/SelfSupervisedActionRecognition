from config import *
import torch.nn as nn
import torch
from itertools import permutations
import random

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
        N, C, T, V, M = X.shape
        X = X.permute(0, 2, 1, 3, 4).contiguous().view(N, T, C * V * M)
        X, _ = self.gru(X)
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


def off_diagonal(x):
    # return a flattened view of the off-diagonal elements of a square matrix
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


class BTwins(nn.Module):
    @ex.capture
    def __init__(self, max_frame, hidden_size, bn_size, lambd):
        super(BTwins, self).__init__()
        self.linear = nn.Linear(max_frame * hidden_size, bn_size)
        self.bn = nn.BatchNorm1d(bn_size, affine=False)
        self.lambd = lambd
        self.apply(init_weights)

    def forward(self, feat1, feat2):
        N, T, CVM = feat1.shape
        feat1 = feat1.contiguous().view(N, T * CVM)
        feat2 = feat2.contiguous().view(N, T * CVM)
        feat1 = self.linear(feat1)
        feat2 = self.linear(feat2)
        c = self.bn(feat1).T @ self.bn(feat2)
        c.div_(N)
        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
        off_diag = off_diagonal(c).pow_(2).sum()

        # 超参数lanbd，需要调试
        BTloss = on_diag + self.lambd * off_diag

        return BTloss
