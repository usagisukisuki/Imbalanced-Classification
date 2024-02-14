#coding: utf-8
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class LogitAdjustedLoss(nn.Module):
    def __init__(self, cls_num_list, tau=1.0, weight=None):
        super().__init__()
        cls_num_ratio = cls_num_list / torch.sum(cls_num_list)
        log_cls_num = torch.log(cls_num_ratio)
        self.log_cls_num = log_cls_num
        self.tau = tau
        self.weight = weight

    def forward(self, logit, target):
        logit_adjusted = logit + self.tau * self.log_cls_num.unsqueeze(0)
        return F.cross_entropy(logit_adjusted, target, weight=self.weight)




