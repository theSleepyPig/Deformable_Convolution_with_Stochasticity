# Adversarial Robustness via Deformable Convolution with Stochasticity
A random adversarial defense method based on stochastic deformable convolution

## Environment
- python: 3.9.12
- pyTorch: 2.2.0
- torchVision: 0.17.0
- cuda: 12.1
- torchattacks: 3.4.0
- tqdm: 4.65.0
- wandb: 0.17.4
- numpy: 1.24.3
- scipy: 1.9.1


## Main Training

### View all available options:

```
python train.py -h
```
```
usage: train.py [-h] [--batch_size BATCH_SIZE] [--data_dir DATA_DIR] [--dataset {cifar10,cifar100,tiny-imagenet}] [--epochs EPOCHS] [--network {ResNet18,WideResNet34,ResNet50,Vit}]
                [--worker WORKER] [--lr_schedule {cyclic,multistep,cosine}] [--lr_min LR_MIN] [--lr_max LR_MAX] [--weight_decay WEIGHT_DECAY] [--momentum MOMENTUM]
                [--none_random_training] [--rand_deform_training] [--randpos_deform_training] [--randpos_multi_deform_training] [--is_n_repeat] [--reNum RENUM]
                [--only_adv_randpos_training] [--rand_path_training] [--epsilon EPSILON] [--alpha ALPHA] [--seed SEED] [--attack_iters ATTACK_ITERS] [--restarts RESTARTS]
                [--none_adv_training] [--save_dir SAVE_DIR] [--pretrain PRETRAIN] [--continue_training] [--lb LB] [--pos POS] [--eot] [--hang] [--device DEVICE]
```



### Example: Train a ResNet18 on CIFAR-10 with PGD Gradient-Selective Adversarial Training and DCS:

```
python train.py --network ResNet18 --dataset cifar10 --batch_size 128  --device 0 --randpos_deform_training --is_n_repeat
```
<!-- ### To run by `nohup`, please add `--hang` to avoid long log by `tqdm`:

```
nohup python train.py [other hyperparameters] --hang > [name of log file] 2>&1 &
``` -->

## Main Evaluation

### Evaluate a trained model under multiple types of attacks (ResNet18 on CIFAR-10):

```
python evaluate_nonrepeated.py --network ResNet18 --dataset cifar10 --batch_size 128  --device 0 --randpos_deform_training --pretrain [path_to_model_ckpt]
```

<!-- ### Evaluate with bpda:
```
python evaluate_nonrepeated_blacktransfer.py --network WideResNet34 --dataset cifar10 --batch_size 128  --device 1 --pretrain /home/yxma/hzx/hzx/hzx/rand_defence/ckpt/cifar10/WideResNet34/ckpt/model_20241107185544.pth --pretraina /home/yxma/hzx/hzx/hzx/rand_defence/ckpt/cifar10/WideResNet34/ckpt/model_20240803034941.pth --randpos_deform_training
``` -->

## DCS on Transformer (Vit) (Organizing)
### Training:
Stage one:
```
python train_vit_pretrain.py --device 0 --randpos_deform_training --optimizer sgd --network earlyVit --dataset cifar10 --lr_max 0.1 --weight_decay 5e-4 --lr_schedule multistep --batch_size 128 --epochs 200 --fix
```
Stage two:
```
python train_vit_pretrain_2.py --device 0 --optimizer sgd --network earlyVit --dataset cifar10 --lr_max 0.01 --weight_decay 5e-4 --lr_schedule cosine --batch_size 128 --epochs 200 --vit_pretrained_path [path_to_model_stage_one] --randpos_deform_training 
```
### Evaluation:
BASE(ViT-t+Conv):
```
python evaluate_vit_repeat_2.py --device 0 --network earlyVit --dataset cifar10 --batch_size 128 --pretrain [path_to_model_base]
```
DCS(ViT-t+DCS):
```
python evaluate_vit_repeat_2.py --device 0 --network earlyVit --dataset cifar10 --batch_size 128 --pretrain [path_to_model_dcs] --randpos_deform_training
```

## Notable Arguments

| Argument                         | Description                                      |
|----------------------------------|--------------------------------------------------|
| --randpos_deform_training        | Enable DCS                                       |
| --is_n_repeat                    | Enable Gradient-Selective Adversarial Training   |
| --save_dir / --pretrain          | Checkpoint save/load                             |


## Pretrained Models

Pretrained models are available on [Hugging Face 🤗](https://huggingface.co/xuanzhu07/Deformable_Convolution_with_Stochasticity_ModelWeight).



## Citation

Coming soon. Please consider citing us if you find this work helpful.


## Note
More codes and weights will be provided after further organization.


<!-- ## Contact

For questions or feedback, please open an issue or contact  
[theSleepyPig](https://github.com/theSleepyPig)

## Pretrained Models
Pretrained models are provided in google drive. The url is

```
https://drive.google.com/drive/folders/1dUY2PoS3HHGrlSEA0M20ToRJpzW2v067?usp=sharing
``` -->