# Imbalanced-Classification

## Learning on long-tailed CIFAR10 or CIFAR100 [[paper]](https://arxiv.org/abs/1901.05555)
### Imbalanced ratio=10
```
python3 train.py --dataset CIFAR10 (or CIFAR100) --gpu 0 --ratio 0.1
```
### Imbalanced ratio=100
```
python3 train.py --dataset CIFAR10 (or CIFAR100) --gpu 0 --ratio 0.01
```
### Imbalanced ratio=200
```
python3 train.py --dataset CIFAR10 (or CIFAR100) --gpu 0 --ratio 0.005
```
### Imbalanced ratio=500
```
python3 train.py --dataset CIFAR10 (or CIFAR100) --gpu 0 --ratio 0.002
```
## Learning with effective loss function
### Focal loss [[paper]](https://arxiv.org/abs/1708.02002)
```
python3 train.py --dataset CIFAR10 --gpu 0 --ratio 0.1 --loss Focal
```
### Class balanced loss [[paper]](https://arxiv.org/abs/1901.05555)
```
python3 train.py --dataset CIFAR10 --gpu 0 --ratio 0.1 --loss CBW
```
### Generalized reweight loss [[paper]](https://arxiv.org/abs/2103.16370)
```
ppython3 train.py --dataset CIFAR10 --gpu 0 --ratio 0.1 --loss GR
```
### Balanced softmax loss [[paper]](https://arxiv.org/abs/2007.10740)
```
python3 train.py --dataset CIFAR10 --gpu 0 --ratio 0.1 --loss BS
```
### LADE loss [[paper]](https://arxiv.org/abs/2012.00321)
```
python3 train.py --dataset CIFAR10 --gpu 0 --ratio 0.1 --loss LADE
```
### LDAM loss [[paper]](https://arxiv.org/abs/1906.07413)
```
python3 train.py --dataset CIFAR10 --gpu 0 --ratio 0.1 --loss LDAM --norm
```
### Logit adjusted loss [[paper]](https://arxiv.org/abs/2007.07314)
```
python3 train.py --dataset CIFAR10 --gpu 0 --ratio 0.1 --loss LA
```
### Vector scaling loss [[paper]](https://arxiv.org/abs/2103.01550)
ppython3 train.py --dataset CIFAR10 --gpu 0 --ratio 0.1 --loss VS
```
### Influence-Balanced loss [[paper]](https://arxiv.org/abs/2110.02444)
```
python3 train.py --dataset CIFAR10 --gpu 0 --ratio 0.1 --loss IB
python3 train.py --dataset CIFAR10 --gpu 0 --ratio 0.1 --loss IBFL
```
### ELM loss [[paper]](https://arxiv.org/abs/2306.09132)
```
python3 train.py --dataset CIFAR10 --gpu 0 --ratio 0.1 --loss ELM --norm
```
### False cross-entropy loss [[paper]]
```
python3 train.py --dataset CIFAR10 --gpu 0 --ratio 0.1 --loss FCE
python3 train.py --dataset CIFAR10 --gpu 0 --ratio 0.1 --loss LAFCE
```

## Learning with class balancing weight [[paper]](https://arxiv.org/abs/1901.05555)
```
python3 train.py --dataset CIFAR10 --gpu 0 --ratio 0.1 --weight_rule CBReweight
python3 train.py --dataset CIFAR10 --gpu 0 --ratio 0.1 --weight_rule IBReweight
```

## Learning with class balancing weighting scheduler [[paper]](https://arxiv.org/abs/1906.07413)
```
python3 train.py --dataset CIFAR10 --gpu 0 --ratio 0.1 --weight_rule CBReweight --weight_scheduler DRW
```

## Learning with hard augmentation
### Mixup [[paper]](https://arxiv.org/abs/1710.09412)
```
python3 train.py --dataset CIFAR10 --gpu 0 --ratio 0.1 --augmentation Mixup
```
### CutMix [[paper]](https://arxiv.org/abs/1905.04899)
```
python3 train.py --dataset CIFAR10 --gpu 0 --ratio 0.1 --augmentation CutMix
```



