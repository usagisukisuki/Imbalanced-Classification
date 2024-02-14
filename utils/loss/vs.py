#coding: utf-8
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class VSLoss(nn.Module):

    def __init__(self, cls_num_list, gamma=0.3, tau=1.0, weight=None):
        super(VSLoss, self).__init__()

        cls_probs = cls_num_list / torch.sum(cls_num_list)
        temp = (1.0 / cls_num_list) ** gamma
        temp = temp / torch.min(temp)

        iota_list = tau * torch.log(cls_probs)
        Delta_list = temp

        self.iota_list = iota_list
        self.Delta_list = Delta_list
        self.weight = weight

    def forward(self, x, target):
        output = x / self.Delta_list + self.iota_list

        return F.cross_entropy(output, target, weight=self.weight)




