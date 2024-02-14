#coding: utf-8
import random
import numpy as np
import os
import argparse
from tqdm import tqdm

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import torchvision.models as models
from torchvision import datasets, transforms
import torch.backends.cudnn as cudnn

from utils.network import *
from utils.mydataset import data_loader
from utils.myloss import choice_loss, class_weight
from utils.myscheduler import adjust_learning_rate
from utils.myaug import mix_data, mix_criterion


##### train ######
def train(epoch, criterion):
    model.train()
    sum_loss = 0
    correct = 0
    total = 0

    for batch_idx, (inputs, targets) in enumerate(tqdm(train_loader, leave=False)):
        inputs = inputs.cuda(device)
        targets = targets.cuda(device)
        targets = targets.long()

        ### hard augmentation ###
        if 'Mix' in args.augmentation:
            inputs, targets_a, targets_b, lam = mix_data(inputs, targets, args.augmentation)
            inputs, targets_a, targets_b = map(Variable, (inputs, targets_a, targets_b))


        output, features = model(inputs)

        ### loss ###
        if 'Mix' in args.augmentation:
            if 'IB' in args.loss:
                loss = mix_criterion(criterion, output, targets_a, targets_b, lam, features, epoch)
            else:
                loss = mix_criterion(criterion, output, targets_a, targets_b, lam, None, None)

        else:
            if 'IB' in args.loss:
                loss = criterion(output, targets, features, epoch)
            else:
                loss = criterion(output, targets)


        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        output = F.softmax(output, dim=1)
        _, predicted = output.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        sum_loss += loss.item()
 
    return sum_loss/(batch_idx+1), float(correct)/float(total)


###### test #######
def test(epoch, criterion):
    model.eval()
    sum_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(tqdm(val_loader, leave=False)):
            inputs = inputs.cuda(device)
            targets = targets.cuda(device)
            targets = targets.long()

            output, features = model(inputs)

            ### loss ###
            if 'IB' in args.loss:
                loss = criterion(output, targets, features, epoch)
            else:
                loss = criterion(output, targets)
  

            output = F.softmax(output, dim=1)
            _, predicted = output.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            sum_loss += loss.item()

    return sum_loss/(batch_idx+1), float(correct)/float(total)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='IMLoss')
    parser.add_argument('--dataset', type=str, default='CIFAR10')
    parser.add_argument('--batchsize', type=int, default=128)
    parser.add_argument('--num_epochs', type=int, default=200)
    parser.add_argument('--out', type=str, default='result')
    parser.add_argument('--gpu', type=int, default=-1)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--ratio', type=float, default=0.01)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--loss', type=str, default='CE', choices=['CE','Focal','CBW','GR','BS','LADE','LDAM','LA',
                                                                   'VS','IB','IBFL','ELM','FCE','LAFCE'], help='loss name')
    parser.add_argument('--norm', action='store_true', help='use norm?')
    parser.add_argument('--weight_rule', type=str, default='None', help='CBReweight or IBReweight')
    parser.add_argument('--weight_scheduler', type=str, default='None', help='DRW')
    parser.add_argument('--augmentation', type=str, default='None', help='Mixup or CutMix')
    args = parser.parse_args()
    gpu_flag = args.gpu


    ### plot ###
    print("[Experimental conditions]")
    print(" GPU ID         : {}".format(args.gpu))
    print(" Dataset        : Im-{}".format(args.dataset))
    print(" Imbalance rate : {}".format(int(1/args.ratio)))
    print(" Loss function  : {}".format(args.loss))
    print(" Class weight   : {}".format(args.weight_rule))
    print(" CW scheduler   : {}".format(args.weight_scheduler))
    print(" Augmentation   : {}".format(args.augmentation))
    print("")


    ### save dir ###
    if not os.path.exists("{}".format(args.out)):
      	os.mkdir("{}".format(args.out))
    if not os.path.exists(os.path.join("{}".format(args.out), "{}".format(args.ratio))):
      	os.mkdir(os.path.join("{}".format(args.out), "{}".format(args.ratio)))
    if not os.path.exists(os.path.join("{}".format(args.out), "{}".format(args.ratio), "model")):
      	os.mkdir(os.path.join("{}".format(args.out), "{}".format(args.ratio), "model"))

    PATH_1 = "{}/{}/trainloss.txt".format(args.out, args.ratio)
    PATH_2 = "{}/{}/testloss.txt".format(args.out, args.ratio)
    PATH_3 = "{}/{}/trainaccuracy.txt".format(args.out, args.ratio)
    PATH_4 = "{}/{}/testaccuracy.txt".format(args.out, args.ratio)

    with open(PATH_1, mode = 'w') as f:
        pass
    with open(PATH_2, mode = 'w') as f:
        pass
    with open(PATH_3, mode = 'w') as f:
        pass
    with open(PATH_4, mode = 'w') as f:
        pass


    ### seed ###
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


    ### device ###
    device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() else 'cpu')


    ### dataset ###
    train_loader, val_loader, cls_num_list, n_classes = data_loader(args)

    cls_num_list = torch.Tensor(cls_num_list).cuda(device)


    ### model ###
    model = resnet32(num_classes=n_classes, use_norm=args.norm).cuda(device)


    ### optimizer ###
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=2e-4)


    sample_acc = 0
    sample_loss = 1000000
    ##### training & test #####
    for epoch in range(args.num_epochs):
        ### learning rate scheduler ###
        adjust_learning_rate(optimizer, epoch, args.lr)

        ### class balancing weight ###
        per_class_weights = class_weight(args, cls_num_list, epoch)

        ### loss function ###
        criterion = choice_loss(args, cls_num_list, per_class_weights, device)

        ### training ###
        loss_train, acc_train = train(epoch, criterion)
        
        ### test ###
        loss_test, acc_test = test(epoch, criterion)

        print("Epoch{:3d}/{:3d}  TrainLoss={:.4f}  TestAccuracy={:.2f}%".format(epoch+1, args.num_epochs, loss_train, acc_test*100))

        with open(PATH_1, mode = 'a') as f:
            f.write("\t%d\t%f\n" % (epoch+1, loss_train))
        with open(PATH_2, mode = 'a') as f:
            f.write("\t%d\t%f\n" % (epoch+1, loss_test))
        with open(PATH_3, mode = 'a') as f:
            f.write("\t%d\t%f\n" % (epoch+1, acc_train))
        with open(PATH_4, mode = 'a') as f:
            f.write("\t%d\t%f\n" % (epoch+1, acc_test))

        if acc_test >= sample_acc:
           sample_acc = acc_test
           PATH_best ="{}/{}/model/model_bestacc.pth".format(args.out, args.ratio)
           torch.save(model.state_dict(), PATH_best)

        if loss_train <= sample_loss:
           sample_loss = loss_train
           PATH_best ="{}/{}/model/model_bestloss.pth".format(args.out, args.ratio)
           torch.save(model.state_dict(), PATH_best)


