import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import torchvision.models as models
from torchvision import datasets, transforms

from .aug import *


##### mix augmentatio #####
def mix_data(x=None, y=None, aug_name=None):
    if aug_name=='Mixup':
        inputs, targets_a, targets_b, lam = mixup_data(x, y, alpha=1.0, use_cuda=True)
    elif aug_name=='CutMix':
        inputs, targets_a, targets_b, lam = cutmix_data(x, y, beta=1.0)

    return inputs, targets_a, targets_b, lam


##### loss #####
def mix_criterion(criterion, pred, y_a, y_b, lam, features=None, epoch=None):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

