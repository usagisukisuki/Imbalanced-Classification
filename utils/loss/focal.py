#coding: utf-8
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=2.0):
        super().__init__()
        assert gamma >= 0
        self.gamma = gamma
        self.weight = weight

    def focal_loss(self, input_values, gamma):
        """Computes the focal loss"""
        p = torch.exp(-input_values)
        loss = (1 - p) ** gamma * input_values
        return loss.mean()

    def forward(self, logit, target):
        return self.focal_loss(F.cross_entropy(logit, target, reduction='none', weight=self.weight), self.gamma)



