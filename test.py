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
from utils.myevaluate import topk_accuracy, confusionmatrix, shot_acc
from utils.mydataset import data_loader


# test関数
def test():
    model.eval()
    sum_acc1 = 0
    sum_acc3 = 0
    sum_acc5 = 0
    predict = []
    answer = []
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(tqdm(test_loader, leave=False)):
            inputs = inputs.cuda(device)
            targets = targets.cuda(device)
            targets = targets.long()


            output, _ = model(inputs)


            acc1, acc3, acc5 = topk_accuracy(output, targets, topk=(1, 3, 5))
            sum_acc1 += acc1[0]
            sum_acc3 += acc3[0]
            sum_acc5 += acc5[0]


            output = F.softmax(output, dim=1)
            _, predicted = output.max(1)

            for j in range(predicted.shape[0]):
                predict.append(predicted[j].cpu())
                answer.append(targets[j].cpu())
        

    # many, medium, few = shot_acc(predict, answer, np.array(cls_num_list), many_shot_thr=100, low_shot_thr=20)

    return sum_acc1/(batch_idx+1), sum_acc3/(batch_idx+1), sum_acc5/(batch_idx+1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='IMLoss')
    parser.add_argument('--dataset', type=str, default='CIFAR10')
    parser.add_argument('--datatype', type=str, default='exp', help='exp or step')
    parser.add_argument('--batchsize', type=int, default=128)
    parser.add_argument('--out', type=str, default='result')
    parser.add_argument('--gpu', type=int, default=-1)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--ratio', type=float, default=0.01)
    parser.add_argument('--norm', action='store_true', help='use norm?')
    args = parser.parse_args()
    gpu_flag = args.gpu


    ### save dir ###
    PATH = "{}/{}/prediction.txt".format(args.out, args.ratio)

    with open(PATH, mode = 'w') as f:
        pass


    ### device ###
    device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() else 'cpu')


    ### dataset ###
    _, test_loader, _, n_classes = data_loader(args)


    ### model ###
    model = resnet32(num_classes=n_classes, use_norm=args.norm).cuda(device)


    ### model load ###
    model_path = "{}/{}/model/model_bestacc.pth".format(args.out, args.ratio)
    model.load_state_dict(torch.load(model_path))


    acc_top1, acc_top3, acc_top5 = test()


    print("Top1 accuracy = {:.2f}%".format(acc_top1))
    print("Top3 accuracy = {:.2f}%".format(acc_top3))
    print("Top5 accuracy = {:.2f}%".format(acc_top5))


    with open(PATH, mode = 'a') as f:
        f.write("%.2f\t%.2f\t%.2f\n" % (acc_top1, acc_top3, acc_top5))


