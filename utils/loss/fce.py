#coding: utf-8
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class FCELoss(nn.Module):
    def __init__(self, cls_num_list=None, temp=4.0, weight=None, device=None):
        super().__init__()
        self.temp = temp
        self.weight = weight

        ### make per_class onehot label ###
        per_cls_label = []
        cls_num_list = cls_num_list.cpu().numpy()
        for i in range(len(cls_num_list)):
            w = np.where(cls_num_list[i] < cls_num_list, 1, 0)
            per_cls_label.append(w)
        self.per_cls_label = torch.FloatTensor(per_cls_label)



    def forward(self, pred, teacher):
        ### prediction ###
        pred = F.softmax(pred/self.temp, dim=1)

        ### make true label ###
        onehot_label = torch.eye(pred.shape[1])[teacher]

        ### make false label ###
        mask1 = torch.ones((onehot_label.shape[0], onehot_label.shape[1]))
        mask2 = torch.zeros((onehot_label.shape[0], onehot_label.shape[1]))
        mask1[onehot_label==1] = 0

        for j in range(pred.shape[0]):
            mask2[j] = mask1[j] * self.per_cls_label[teacher[j]]

        ### label ###
        mask2 = mask2.cuda()
        onehot_label = onehot_label.cuda()

        ### loss ###
        loss = -1 * (onehot_label * torch.log(pred + 1e-7) + mask2 * torch.log(1 - pred + 1e-7))
        loss = loss.sum(dim=1)

        return loss.mean()



class LAFCELoss(nn.Module):
    def __init__(self, cls_num_list=None, temp=6.0, weight=None, device=None):
        super().__init__()
        self.temp = temp
        self.weight = weight

        ### logit adjust ###
        cls_num_ratio = cls_num_list / torch.sum(cls_num_list)
        log_cls_num = torch.log(cls_num_ratio)
        self.log_cls_num = log_cls_num

        ### make per_class onehot label ###
        per_cls_label = []
        cls_num_list = cls_num_list.cpu().numpy()
        for i in range(len(cls_num_list)):
            w = np.where(cls_num_list[i] < cls_num_list, 1, 0)
            per_cls_label.append(w)
        self.per_cls_label = torch.FloatTensor(per_cls_label)



    def forward(self, pred, teacher):
        ### prediction ###
        pred = F.softmax(pred/self.temp + self.log_cls_num.unsqueeze(0), dim=1)

        ### make true label ###
        onehot_label = torch.eye(pred.shape[1])[teacher]

        ### make false label ###
        mask1 = torch.ones((onehot_label.shape[0], onehot_label.shape[1]))
        mask2 = torch.zeros((onehot_label.shape[0], onehot_label.shape[1]))
        mask1[onehot_label==1] = 0

        for j in range(pred.shape[0]):
            mask2[j] = mask1[j] * self.per_cls_label[teacher[j]]

        ### label ###
        mask2 = mask2.cuda()
        onehot_label = onehot_label.cuda()

        ### loss ###
        loss = -1 * (onehot_label * torch.log(pred + 1e-7) + mask2 * torch.log(1 - pred + 1e-7))
        loss = loss.sum(dim=1)

        return loss.mean()







