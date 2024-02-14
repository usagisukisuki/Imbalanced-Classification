#coding: utf-8
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class IBLoss(nn.Module):
    def __init__(self, cls_num_list=None, alpha=10000., start_ib_epoch=100):
        super(IBLoss, self).__init__()
        assert alpha > 0
        self.alpha = alpha
        self.epsilon = 0.001
        self.start_ib_epoch = start_ib_epoch

        self.weight = (1.0 / cls_num_list) / torch.sum(cls_num_list) * len(cls_num_list)

    def ib_loss(self, input_values, ib):
        loss = input_values * ib
        return loss.mean()

    def forward(self, input, target, features, epoch):
        if epoch >= self.start_ib_epoch:
            features = torch.sum(torch.abs(features), 1).reshape(-1, 1)
            grads = torch.sum(torch.abs(F.softmax(input, dim=1) - F.one_hot(target, input.shape[1])),1) # N * 1
            ib = grads*features.reshape(-1)
            ib = self.alpha / (ib + self.epsilon)
            loss = self.ib_loss(F.cross_entropy(input, target, reduction='none', weight=self.weight), ib)
        
        else:
            loss = F.cross_entropy(input, target)

        return loss



class IB_FocalLoss(nn.Module):
    def __init__(self, cls_num_list=None, alpha=10000., gamma=0., start_ib_epoch=100):
        super(IB_FocalLoss, self).__init__()
        assert alpha > 0
        self.alpha = alpha
        self.epsilon = 0.001
        self.gamma = gamma
        self.start_ib_epoch = start_ib_epoch

        self.weight = (1.0 / cls_num_list) / torch.sum(cls_num_list) * len(cls_num_list)

    def ib_focal_loss(self, input_values, ib, gamma):
        """Computes the ib focal loss"""
        p = torch.exp(-input_values)
        loss = (1 - p) ** gamma * input_values * ib
        return loss.mean()

    def forward(self, input, target, features, epoch):
        if epoch >= self.start_ib_epoch:
            grads = torch.sum(torch.abs(F.softmax(input, dim=1) - F.one_hot(target, input.shape[1])),1) # N * 1
            ib = grads*(features.reshape(-1))
            ib = self.alpha / (ib + self.epsilon)
            loss = self.ib_focal_loss(F.cross_entropy(input, target, reduction='none', weight=self.weight), ib, self.gamma)

        else:
            loss = F.cross_entropy(input, target)

        return loss




