### learning on long-tailed CIFAR10 ###
#ratio=10
python3 train.py --dataset CIFAR10 --gpu 0 --ratio 0.1
#ratio=100
python3 train.py --dataset CIFAR10 --gpu 0 --ratio 0.01
#ratio=200
python3 train.py --dataset CIFAR10 --gpu 0 --ratio 0.005
#ratio=500
python3 train.py --dataset CIFAR10 --gpu 0 --ratio 0.002


### learning on stepped CIFAR10 ###
#ratio=10
python3 train.py --dataset CIFAR10 --gpu 0 --ratio 0.1 --datatype step


### learning on long-tailed CIFAR100 ###
#ratio=10
python3 train.py --dataset CIFAR100 --gpu 0 --ratio 0.1
#ratio=100
python3 train.py --dataset CIFAR100 --gpu 0 --ratio 0.01
#ratio=200
python3 train.py --dataset CIFAR100 --gpu 0 --ratio 0.005
#ratio=500
python3 train.py --dataset CIFAR100 --gpu 0 --ratio 0.002


### learning on stepped CIFAR100 ###
#ratio=10
python3 train.py --dataset CIFAR100 --gpu 0 --ratio 0.1 --datatype step


### learning with Focal loss ###
python3 train.py --dataset CIFAR10 --gpu 0 --ratio 0.1 --loss Focal

### learning with class balanced loss ###
python3 train.py --dataset CIFAR10 --gpu 0 --ratio 0.1 --loss CBW

### learning with generalized reweight loss ###
python3 train.py --dataset CIFAR10 --gpu 0 --ratio 0.1 --loss GR

### learning with balanced softmax loss ###
python3 train.py --dataset CIFAR10 --gpu 0 --ratio 0.1 --loss BS

### learning with LADE loss ###
python3 train.py --dataset CIFAR10 --gpu 0 --ratio 0.1 --loss LADE

### learning with LDAM loss ###
python3 train.py --dataset CIFAR10 --gpu 0 --ratio 0.1 --loss LDAM --norm

### learning with logit adjusted loss ###
python3 train.py --dataset CIFAR10 --gpu 0 --ratio 0.1 --loss LA

### learning with vector scaling loss ###
python3 train.py --dataset CIFAR10 --gpu 0 --ratio 0.1 --loss VS

### learning with Influence-Balanced loss ###
python3 train.py --dataset CIFAR10 --gpu 0 --ratio 0.1 --loss IB
python3 train.py --dataset CIFAR10 --gpu 0 --ratio 0.1 --loss IBFL

### learning with ELM loss ###
python3 train.py --dataset CIFAR10 --gpu 0 --ratio 0.1 --loss ELM --norm

### learning with false crossentreopy loss ###
python3 train.py --dataset CIFAR10 --gpu 0 --ratio 0.1 --loss FCE
python3 train.py --dataset CIFAR10 --gpu 0 --ratio 0.1 --loss LAFCE


### learning with class balancing weight ###
python3 train.py --dataset CIFAR10 --gpu 0 --ratio 0.1 --weight_rule CBReweight
python3 train.py --dataset CIFAR10 --gpu 0 --ratio 0.1 --weight_rule IBReweight


### learning with class balancing weighting scheduler ###
python3 train.py --dataset CIFAR10 --gpu 0 --ratio 0.1 --weight_rule CBReweight --weight_scheduler DRW


### learning with Mixup ###
python3 train.py --dataset CIFAR10 --gpu 0 --ratio 0.1 --augmentation Mixup


### learning with CutMix ###
python3 train.py --dataset CIFAR10 --gpu 0 --ratio 0.1 --augmentation CutMix


