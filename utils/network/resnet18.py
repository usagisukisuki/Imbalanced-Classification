#coding: utf-8
import numpy as np
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import torchvision.models as models
from torchvision import datasets, transforms
import torch.nn.init as init
from torch.nn import Parameter


class ResNet18(nn.Module):
    def __init__(self, num_classes=10, pretrained=False):
        super(ResNet18, self).__init__()
        model = models.resnet18(pretrained=pretrained)
        layers = list(model.children())[:-1]
        self.backborn = nn.Sequential(*layers)
        self.fc = nn.Linear(512, num_classes)


    def __call__(self, x):
        feature = self.backborn(x)
        feature = feature.view(feature.shape[0], -1)
        out = self.fc(feature)
        return out, feature
