from config import *
from model import *
from dataset import DataSet
from logger import Log

import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm

from torch.utils.tensorboard import SummaryWriter


class BaseProcessor:

    @ex.capture
    def load_data(self, train_list, train_label, train_frame, test_list, test_label, test_frame, batch_size, train_clip,
                  label_clip):
        self.dataset = dict()
        self.data_loader = dict()
        self.auto_data_loader = dict()

        self.dataset['train'] = DataSet(train_list, train_label, train_frame)

        full_len = len(self.dataset['train'])
        train_len = int(train_clip * full_len)
        val_len = full_len - train_len
        self.dataset['train'], self.dataset['val'] = torch.utils.data.random_split(self.dataset['train'],
                                                                                   [train_len, val_len])

        self.data_loader['train'] = torch.utils.data.DataLoader(
            dataset=self.dataset['train'],
            batch_size=batch_size,
            shuffle=False)

        self.data_loader['val'] = torch.utils.data.DataLoader(
            dataset=self.dataset['val'],
            batch_size=batch_size,
            shuffle=False)

        if label_clip != 1.0:
            label_len = int(label_clip * train_len)
            unlabel_len = train_len - label_len
            self.dataset['label'], self.dataset['unlabel'] = torch.utils.data.random_split(self.dataset['train'],
                                                                                           [label_len, unlabel_len])

            self.data_loader['label'] = torch.utils.data.DataLoader(
                dataset=self.dataset['label'],
                batch_size=batch_size,
                shuffle=False)

            self.data_loader['unlabel'] = torch.utils.data.DataLoader(
                dataset=self.dataset['unlabel'],
                batch_size=batch_size,
                shuffle=False)
        else:
            self.data_loader['label'] = torch.utils.data.DataLoader(
                dataset=self.dataset['train'],
                batch_size=batch_size,
                shuffle=False)

        self.dataset['test'] = DataSet(test_list, test_label, test_frame)

        self.data_loader['test'] = torch.utils.data.DataLoader(
            dataset=self.dataset['test'],
            batch_size=batch_size,
            shuffle=False)

    def load_weights(self, model=None, weight_path=None):
        if weight_path:
            pretrained_dict = torch.load(weight_path)
            model.load_state_dict(pretrained_dict)

    def initialize(self):
        self.load_data()
        self.load_model()
        self.load_optim()
        self.log = Log()

    @ex.capture
    def optimize(self, epoch_num):
        for epoch in range(epoch_num):
            self.epoch = epoch
            self.train_epoch()
            self.val_epoch()
            self.test_epoch()
            self.log.update_epoch(self.epoch)

    @ex.capture
    def save_model(self, train_mode):
        torch.save(self.encoder.state_dict(), f"output/model/{train_mode}.pt")

    def start(self):
        self.initialize()
        self.optimize()
        self.save_model()


# %%
class RecognitionProcessor(BaseProcessor):

    @ex.capture
    def load_model(self, train_mode, weight_path):
        self.encoder = Encoder()
        self.encoder = torch.nn.DataParallel(self.encoder).cuda()
        self.classifier = Linear()
        self.classifier = torch.nn.DataParallel(self.classifier).cuda()
        if 'loadweight' in train_mode:
            self.load_weights(self.encoder, weight_path)

    @ex.capture
    def load_optim(self):
        self.optimizer = torch.optim.Adam([
            {'params': self.encoder.parameters()},
            {'params': self.classifier.parameters(), 'lr': 1e-3}], lr=1e-3)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=30, gamma=0.1)
        self.CrossEntropyLoss = torch.nn.CrossEntropyLoss().cuda()

    @ex.capture
    def train_epoch(self, clip_gradient):
        self.encoder.train()
        self.classifier.train()
        loader = self.data_loader['label']
        for data, label, frame in tqdm(loader):
            data = data.type(torch.FloatTensor).cuda()
            label = label.type(torch.LongTensor).cuda()
            frame = frame.type(torch.LongTensor).cuda()
            loss = self.train_batch(data, label, frame)
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), clip_gradient)
            torch.nn.utils.clip_grad_norm_(self.classifier.parameters(), clip_gradient)
            self.optimizer.step()
        self.scheduler.step()

    @ex.capture
    def train_batch(self, data, label, frame, train_mode):
        Z = self.encoder(data)
        if "linear" in train_mode:
            Z = Z.detach()
        # Z = mask_mean(Z, frame)
        predict = self.classifier(Z).mean(1)
        _, pred = torch.max(predict, 1)
        acc = pred.eq(label.view_as(pred)).float().mean()

        cls_loss = self.CrossEntropyLoss(predict, label)
        loss = cls_loss

        self.log.update_batch("log/train/cls_acc", acc.item())
        self.log.update_batch("log/train/cls_loss", loss.item())

        return loss

    def test_epoch(self):
        self.encoder.eval()
        self.classifier.eval()

        loader = self.data_loader['test']
        for data, label, frame in tqdm(loader):
            data = data.type(torch.FloatTensor).cuda()
            label = label.type(torch.LongTensor).cuda()
            frame = frame.type(torch.LongTensor).cuda()
            # inference
            with torch.no_grad():
                Z = self.encoder(data)
                # Z = mask_mean(Z, frame)
                predict = self.classifier(Z).mean(1)
            _, pred = torch.max(predict, 1)
            acc = pred.eq(label.view_as(pred)).float().mean()
            cls_loss = self.CrossEntropyLoss(predict, label)
            loss = cls_loss
            self.log.update_batch("log/test/cls_acc", acc.item())
            self.log.update_batch("log/test/cls_loss", loss.item())

    def val_epoch(self):
        self.encoder.eval()
        self.classifier.eval()

        loader = self.data_loader['val']
        for data, label, frame in tqdm(loader):
            data = data.type(torch.FloatTensor).cuda()
            label = label.type(torch.LongTensor).cuda()
            frame = frame.type(torch.LongTensor).cuda()
            # inference
            with torch.no_grad():
                Z = self.encoder(data)
                # Z = mask_mean(Z, frame)
                predict = self.classifier(Z)
            _, pred = torch.max(predict, 1)
            acc = pred.eq(label.view_as(pred)).float().mean()
            cls_loss = self.CrossEntropyLoss(predict, label)
            loss = cls_loss
            self.log.update_batch("log/val/cls_acc", acc.item())
            self.log.update_batch("log/val/cls_loss", loss.item())


class MS2LProcessor(BaseProcessor):

    def load_model(self):
        self.encoder = Encoder()
        self.encoder = torch.nn.DataParallel(self.encoder).cuda()
        
        self.btwins_head = BTwins()
        self.btwins_head = torch.nn.DataParallel(self.btwins_head).cuda()

    def load_optim(self):
        self.optimizer = torch.optim.Adam([
            {'params': self.encoder.parameters()},
            {'params': self.btwins_head.parameters(), 'lr': 1e-3}], lr=1e-3)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=30, gamma=0.1)

    def btwins_batch(self, feat1, feat2):
        BTloss = self.btwins_head(feat1, feat2)
        self.log.update_batch("log/train/bt_loss", BTloss.item())
        return BTloss

    @ex.capture
    def train_epoch(self, clip_gradient, train_mode):
        self.encoder.train()
        loader = self.data_loader['train']

        for data, label, frame in tqdm(loader):
            data = data.type(torch.FloatTensor)
            label = label.type(torch.LongTensor)
            frame = frame.type(torch.LongTensor)
            input1 = data
            noise = 0.1 * torch.randn(input1.shape)
            input2 = input1 + noise

            data = data.cuda()
            label = label.cuda()
            frame = frame.cuda()
            input1 = input1.cuda()
            input2 = input2.cuda()
        
            feat1 = self.encoder(input1)
            feat2 = self.encoder(input2)

            loss = self.btwins_batch(feat1, feat2)

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), clip_gradient)
            torch.nn.utils.clip_grad_norm_(self.btwins_head.parameters(), clip_gradient)
            self.optimizer.step()

        self.scheduler.step()

    @ex.capture
    def optimize(self, epoch_num):
        for epoch in range(epoch_num):
            self.epoch = epoch
            self.train_epoch()
            self.log.update_epoch(self.epoch)


# %%
@ex.automain
def main(train_mode):
    # 固定随机种子
    torch.cuda.manual_seed_all(1)
    torch.manual_seed(1)
    np.random.seed(1)
    random.seed(1)
    torch.backends.cudnn.deterministic = True
    #torch.backends.cudnn.benchmark = False
    #
    if "pretrain" in train_mode:
        p = MS2LProcessor()
        p.start()

    if "loadweight" in train_mode:
        p = RecognitionProcessor()
        p.start()
