# Imbalanced-Classification

## Learning on long-tailed CIFAR10
### Imbalanced ratio=10
```
python3 train.py --dataset CIFAR10 --gpu 0 --ratio 0.1
```
### Imbalanced ratio=100
```
python3 train.py --dataset CIFAR10 --gpu 0 --ratio 0.01
```
### Imbalanced ratio=200
```
python3 train.py --dataset CIFAR10 --gpu 0 --ratio 0.005
```
### Imbalanced ratio=500
```
python3 train.py --dataset CIFAR10 --gpu 0 --ratio 0.002
```
## Learning with effective loss function
### Focal loss
```
python3 train.py --dataset CIFAR10 --gpu 0 --ratio 0.1 --loss Focal
```
### Class balanced loss
```
python3 train.py --dataset CIFAR10 --gpu 0 --ratio 0.1 --loss CBW
```
### Generalized reweight loss
```
ppython3 train.py --dataset CIFAR10 --gpu 0 --ratio 0.1 --loss GR
```
### Balanced softmax loss
```
python3 train.py --dataset CIFAR10 --gpu 0 --ratio 0.1 --loss BS
```
### LADE loss
```
python3 train.py --dataset CIFAR10 --gpu 0 --ratio 0.1 --loss LADE
```
### LDAM loss
```
python3 train.py --dataset CIFAR10 --gpu 0 --ratio 0.1 --loss LDAM --norm
```
### Logit adjusted loss
```
python3 train.py --dataset CIFAR10 --gpu 0 --ratio 0.1 --loss LA
```
### Vector scaling loss
```
ppython3 train.py --dataset CIFAR10 --gpu 0 --ratio 0.1 --loss VS
```
### Influence-Balanced loss
```
python3 train.py --dataset CIFAR10 --gpu 0 --ratio 0.1 --loss IB
python3 train.py --dataset CIFAR10 --gpu 0 --ratio 0.1 --loss IBFL
```
### ELM loss
```
python3 train.py --dataset CIFAR10 --gpu 0 --ratio 0.1 --loss ELM --norm
```
### False cross-entropy loss
```
python3 train.py --dataset CIFAR10 --gpu 0 --ratio 0.1 --loss FCE
python3 train.py --dataset CIFAR10 --gpu 0 --ratio 0.1 --loss LAFCE
```

## Learning with class balancing weight
```
python3 train.py --dataset CIFAR10 --gpu 0 --ratio 0.1 --weight_rule CBReweight
python3 train.py --dataset CIFAR10 --gpu 0 --ratio 0.1 --weight_rule IBReweight
```

## Learning with class balancing weighting scheduler
```
python3 train.py --dataset CIFAR10 --gpu 0 --ratio 0.1 --weight_rule CBReweight --weight_scheduler DRW
```

## Learning with hard augmentation
### Mixup
```
python3 train.py --dataset CIFAR10 --gpu 0 --ratio 0.1 --augmentation Mixup
```
### CutMix
```
python3 train.py --dataset CIFAR10 --gpu 0 --ratio 0.1 --augmentation CutMix
```



