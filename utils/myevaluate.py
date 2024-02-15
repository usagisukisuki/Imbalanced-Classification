import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn  as sns
from matplotlib import pyplot as plt


def confusionmatrix(answer, predict, n_classes, args):
    label = list(range(n_classes))

    mat = confusion_matrix(answer, predict, labels=label)

    sns.heatmap(mat, annot=True, fmt='.0f', cmap='jet')
    plt.xticks(fontsize=10) 
    plt.yticks(fontsize=10) 
    plt.ylabel("Ground truth", fontsize=14)
    plt.xlabel("Predicted label", fontsize=14)
    plt.title("Imbalanced CIFAR10 (Ratio={:2d})".format(int(1/args.ratio)), fontsize=14)
    plt.savefig("{}/{}/ConfusionMatrix.png".format(args.out, args.ratio))
    plt.close()



def topk_accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        output = F.softmax(output, dim=1)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))

        return res


def shot_acc(preds, labels, train_data, many_shot_thr=100, low_shot_thr=20, acc_per_cls=False):
    if isinstance(train_data, np.ndarray):
        training_labels = np.array(train_data).astype(int)
    else:
        training_labels = np.array(train_data.dataset.labels).astype(int)

    if isinstance(preds, torch.Tensor):
        preds = preds.detach().cpu().numpy()
        labels = labels.detach().cpu().numpy()
    elif isinstance(preds, np.ndarray):
        pass
    else:
        raise TypeError('Type ({}) of preds not supported'.format(type(preds)))
    train_class_count = []
    test_class_count = []
    class_correct = []
    for l in np.unique(labels):
        train_class_count.append(len(training_labels[training_labels == l]))
        test_class_count.append(len(labels[labels == l]))
        class_correct.append((preds[labels == l] == labels[labels == l]).sum())
    print(class_correct)
    many_shot = []
    median_shot = []
    low_shot = []
    for i in range(len(train_data)):
        if train_data[i] > many_shot_thr:
            many_shot.append((class_correct[i] / test_class_count[i]))
        elif train_data[i] < low_shot_thr:
            low_shot.append((class_correct[i] / test_class_count[i]))
        else:
            median_shot.append((class_correct[i] / test_class_count[i]))    
 
    if len(many_shot) == 0:
        many_shot.append(0)
    if len(median_shot) == 0:
        median_shot.append(0)
    if len(low_shot) == 0:
        low_shot.append(0)

    if acc_per_cls:
        class_accs = [c / cnt for c, cnt in zip(class_correct, test_class_count)] 
        return np.mean(many_shot), np.mean(median_shot), np.mean(low_shot), class_accs
    else:
        return np.mean(many_shot), np.mean(median_shot), np.mean(low_shot)
