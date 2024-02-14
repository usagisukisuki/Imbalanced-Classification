import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import torchvision.models as models
from torchvision import datasets, transforms

from utils.dataset import IMBALANCECIFAR10, IMBALANCECIFAR100


##### data loader #####
def data_loader(args):
    transform_train = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(),
                                          transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

    transform_val = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])


    if args.dataset=='CIFAR10':
        n_classes = 10
        train_dataset = IMBALANCECIFAR10(root='./data', imb_type='exp', imb_factor=args.ratio, rand_number=args.seed, train=True, download=True, transform=transform_train)
        val_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_val)
    elif args.dataset=='CIFAR100':
        n_classes = 100
        train_dataset = IMBALANCECIFAR100(root='./data', imb_type='exp', imb_factor=args.ratio, rand_number=args.seed, train=True, download=True, transform=transform_train)
        val_dataset = datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_val)


    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batchsize, shuffle=True, drop_last=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=50, shuffle=False, drop_last=True)


    cls_num_lists = train_dataset.get_cls_num_list()


    return train_loader, val_loader, cls_num_lists, n_classes

