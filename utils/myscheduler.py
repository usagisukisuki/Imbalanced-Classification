import torch

def adjust_learning_rate(optimizer, epoch, lr):
    epoch = epoch + 1
    if epoch <= 5:
        lr = lr * epoch / 5
    elif epoch > 180:
        lr = lr * 0.0001
    elif epoch > 160:
        lr = lr * 0.01
    else:
        lr = lr
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
