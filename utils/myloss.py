import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import torchvision.models as models
from torchvision import datasets, transforms

from .loss import *
from .dataset import *

##### class balancing weight #####
def class_weight(args, cls_num_list, epoch):
    if args.weight_rule == 'CBReweight':
        if args.weight_scheduler == 'DRW': # https://arxiv.org/abs/1906.07413
            idx = epoch // 160
            betas = [0, 0.9999]
        else:
            idx = 0
            betas = [0.9999, 0.9999]

        effective_num = 1.0 - torch.pow(betas[idx], cls_num_list)
        per_cls_weights = (1.0 - betas[idx]) / effective_num
        per_cls_weights = per_cls_weights / torch.sum(per_cls_weights) * len(cls_num_list)

    elif args.weight_rule == 'IBReweight':
        if args.weight_scheduler == 'DRW': # https://arxiv.org/abs/1906.07413
            idx = epoch // 160
            betas = [0, 1]
        else:
            idx = 0
            betas = [1, 1]
        per_cls_weights = (1.0 / cls_num_list) * betas[idx]
        per_cls_weights = per_cls_weights / torch.sum(per_cls_weights) * len(cls_num_list)

    else:
        per_cls_weights = None

    return per_cls_weights


##### loss #####
def choice_loss(args, cls_num_list, weight, device):
    if args.loss=='CE':
        criterion = nn.CrossEntropyLoss(weight=weight).cuda(device)
    elif args.loss=='Focal':
        # https://arxiv.org/abs/1708.02002
        criterion = FocalLoss(weight=weight).cuda(device)
    elif args.loss=='CBW':
        # https://arxiv.org/abs/1901.05555
        criterion = ClassBalancedLoss(cls_num_list=cls_num_list).cuda(device)
    elif args.loss=='GR':
        # https://arxiv.org/abs/2103.16370
        criterion = GeneralizedReweightLoss(cls_num_list=cls_num_list).cuda(device)
    elif args.loss=='BS':
       # https://arxiv.org/abs/2007.10740
        criterion = BalancedSoftmaxLoss(cls_num_list=cls_num_list).cuda(device)
    elif args.loss=='LADE':
        # https://arxiv.org/abs/2012.00321
        criterion = LADELoss(cls_num_list=cls_num_list).cuda(device)
    elif args.loss=='LDAM': 
        # https://arxiv.org/abs/1906.07413
        criterion = LDAMLoss(cls_num_list=cls_num_list, weight=weight).cuda(device)
    elif args.loss=='LA': 
        # https://arxiv.org/abs/2007.07314
        criterion = LogitAdjustedLoss(cls_num_list=cls_num_list, weight=weight).cuda(device)
    elif args.loss=='VS': 
        # https://arxiv.org/abs/2103.01550
        if args.dataset == 'CIFAR10':
            criterion = VSLoss(cls_num_list=cls_num_list, tau=0.15, gamma=1.25, weight=weight).cuda(device)
        elif args.dataset == 'CIFAR100':
            criterion = VSLoss(cls_num_list=cls_num_list, tau=0.05, gamma=0.75, weight=weight).cuda(device)
    elif args.loss=='IB': 
        # https://arxiv.org/abs/2110.02444
        criterion = IBLoss(cls_num_list=cls_num_list, alpha=1000).cuda(device)
    elif args.loss=='IBFL': 
        # https://arxiv.org/abs/2110.02444
        criterion = IB_FocalLoss(cls_num_list=cls_num_list, alpha=1000).cuda(device)
    elif args.loss=='ELM': 
        # https://arxiv.org/abs/2306.09132
        criterion = ELMLoss(cls_num_list=cls_num_list, weight=weight).cuda(device)
    elif args.loss=='FCE':
        criterion = FCELoss(cls_num_list=cls_num_list).cuda(device)
    elif args.loss=='LAFCE':
        criterion = LAFCELoss(cls_num_list=cls_num_list).cuda(device)

    return criterion

