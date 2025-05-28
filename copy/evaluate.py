import argparse
import copy
import logging
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import evaluate_standard, evaluate_standard_random_weights, get_loaders, load_model, evaluate_standard_random_mask

import torchattacks
from tqdm import tqdm

import wandb  

logger = logging.getLogger(__name__)
#torch.manual_seed(0)

def get_args():
    parser = argparse.ArgumentParser()
    # training settings
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--data_dir', default='~/datasets/CIFAR10/', type=str)
    parser.add_argument('--dataset', default='cifar10', choices=['cifar10', 'cifar100'])
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--network', default='ResNet18', choices=['ResNet18', 'WideResNet34'], type=str)
    parser.add_argument('--worker', default=4, type=int)
    parser.add_argument('--lr_schedule', default='multistep', choices=['cyclic', 'multistep', 'cosine'], type=str)
    parser.add_argument('--lr_min', default=0., type=float)
    parser.add_argument('--lr_max', default=0.1, type=float)
    parser.add_argument('--weight_decay', default=5e-4, type=float)
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--none_random_training', action='store_true', default=True,
                        help='Disable random weight training')

    parser.add_argument('--rand_deform_training', action='store_true', default=False,
                        help='able random weight training')
    parser.add_argument('--randpos_deform_training', action='store_true', default=True,
                        help='able random mask training')
    parser.add_argument('--randpos_multi_deform_training', action='store_true', default=False,
                        help='able multi random mask training')
    
    parser.add_argument('--only_adv_randpos_training', action='store_true', default=False,
                        help='able random weight training')
    parser.add_argument('--rand_path_training', action='store_true', default=False,
                        help='able random path training')
    #none_rand_deform_training False True

    # adversarial settings
    parser.add_argument('--epsilon', default=8, type=int)
    parser.add_argument('--alpha', default=2, type=float, help='Step size')
    parser.add_argument('--c', default=1e-4, type=float, help='c in torchattacks')
    parser.add_argument('--steps', default=1000, type=int, help='steps in torchattacks')
    parser.add_argument('--seed', default=0, type=int, help='Random seed')
    parser.add_argument('--attack_iters', default=20, type=int, help='Attack iterations')
    parser.add_argument('--restarts', default=1, type=int)
    
    # checkpoint settings
    parser.add_argument('--save_dir', default='ResNet18_DeformableConvolution', type=str, help='Output directory')
    parser.add_argument('--pretrain', default=os.path.abspath('hzx/rand_defence/ckpt/cifar10/ResNet18/ckpt/model_20240928164626.pth'), type=str, help='Path to load the pretrained model')
    #这里hzx/rand_defence/ckpt/cifar10/ResNet18/ckpt/model_20240928164626.pth
    parser.add_argument('--continue_training', action='store_true', help='Continue training at the checkpoint if exists')

    # CTRW settings
    parser.add_argument('--lb', default=2048, help='The Lower bound of sum of sigma.')
    parser.add_argument('--pos', default=0, help='The position of CTRW over the whole network.')
    parser.add_argument('--eot', action='store_true', help='Whether set random weight each step.')

    # running settings
    parser.add_argument('--hang', action='store_true', help='Whether hang up. If yes, please add it. This will block "tqdm" mode to reduce the size of log file.')
    parser.add_argument('--device', default=3, type=int, help='CUDA device')

    return parser.parse_args()



def evaluate_attack(device, model, test_loader, args, atk, atk_name, logger):
    test_loss = 0
    test_acc = 0
    n = 0
    model.eval()
    state_dict = model.state_dict()

    test_loader = iter(test_loader)

    bar_format = '{desc}[{elapsed}<{remaining},{rate_fmt}]'
    if args.hang:
        pbar = range(len(test_loader))
    else:
        pbar = tqdm(range(len(test_loader)), file=sys.stdout, bar_format=bar_format, ncols=80, disable=False) #True False
    for i in pbar:
        X, y = test_loader.__next__()
        X, y = X.to(device), y.to(device)

        # random select a path to attack
        if not args.none_random_training:
            model.set_rands()

        if args.randpos_deform_training:
            model.set_rand_mask()
            
        # # 调试信息：检查模型输出的类别数量
        # with torch.no_grad():
        #     output = model(X)
        #     print(f"Model output shape: {output.shape}")
        # ([128, 10]),([16, 10])

        X_adv = atk(X, y)  # advtorch
        model.load_state_dict(state_dict)

        # random select a path to infer
        if not args.none_random_training:
            model.set_rands()

        if args.randpos_deform_training:
            model.set_rand_mask()

        with torch.no_grad():
            output = model(X_adv)
        loss = F.cross_entropy(output, y)
        test_loss += loss.item() * y.size(0)
        test_acc += (output.max(1)[1] == y).sum().item()
        n += y.size(0)

    pgd_acc = test_acc / n

    logger.info(atk_name)
    logger.info('adv: %.4f \t', pgd_acc)
    wandb.log({f"{atk_name}_adv": pgd_acc})  # 将对抗准确率记录到 wandb
    return pgd_acc

def main():
    args = get_args()
    device = torch.device(args.device)

     # 初始化 wandb
    wandb.init(project='-Test-pos', entity='xuanzhu_07-university-of-sydney', reinit=True, config=args)
    config = wandb.config   
    
    args.save_dir = os.path.join('logs', args.save_dir)
    print(f"Process ID: {os.getpid()}")

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    logfile = os.path.join(args.save_dir, 'output.log')
    if os.path.exists(logfile):
        os.remove(logfile)


    log_path = os.path.join(args.save_dir, 'output_test.log')

    handlers = [logging.FileHandler(log_path, mode='a+'),
                logging.StreamHandler()]

    logging.basicConfig(
        format='[%(asctime)s] - %(message)s',
        datefmt='%Y/%m/%d %H:%M:%S',
        level=logging.INFO,
        handlers=handlers)

    logger.info(args)
    print(f"Pretrain model path: {args.pretrain}")
    print(f"Does pretrain model path exist? {os.path.exists(args.pretrain)}")

    assert type(args.pretrain) == str and os.path.exists(args.pretrain)

    if args.dataset == 'cifar10':
        args.num_classes = 10
    elif args.dataset == 'cifar100':
        args.num_classes = 100
    else:
        print('Wrong dataset:', args.dataset)
        exit()

    logger.info('Dataset: %s', args.dataset)

    train_loader, test_loader, dataset_normalization = get_loaders(args.data_dir, args.batch_size, dataset=args.dataset,
                                                                   worker=args.worker, norm=False)

    # setup network
    net = load_model(args = args)
    model = net(num_classes=args.num_classes, normalize = dataset_normalization, device = device, pos = args.pos, eot = args.eot, lb = args.lb).to(device)


    #model = torch.nn.DataParallel(model)
    print(model)

    # load pretrained model
    pretrained_model = torch.load(args.pretrain, map_location=device)#['state_dict']
    partial = pretrained_model['state_dict']
    #partial['SD_'] = partial['std']
    #partial['Mu_'] = partial['mean']

    state = model.state_dict()
    

    pretrained_dict = {k: v for k, v in partial.items() if k in list(state.keys()) and state[k].size() == partial[k].size()}
    state.update(pretrained_dict)
    print("Different keys:")
    for i in model.state_dict().keys():
        if i not in pretrained_dict.keys():
            print(i)

    model.load_state_dict(state)
    
    # normalize_mean = torch.tensor([0.4914, 0.4822, 0.4465], device=device)
    # normalize_std = torch.tensor([0.2471, 0.2435, 0.2616], device=device)
    # if 'normalize.mean' not in state:
    #     model.normalize.mean = normalize_mean
    # if 'normalize.std' not in state:
    #     model.normalize.std = normalize_std
            
    model.eval()
    

    # Evaluation
    if not args.none_random_training:
        logger.info('Evaluating with standard images with random weight...')
        _, nature_acc = evaluate_standard_random_weights(device, test_loader, model, args)
        logger.info('Nature Acc: %.4f \t', nature_acc)
        wandb.log({'Nature Acc': nature_acc})
    if args.randpos_deform_training:
        logger.info('Evaluating with standard images with random mask...')
        _, nature_acc = evaluate_standard_random_mask(device, test_loader, model, args)
        logger.info('Nature Acc: %.4f \t', nature_acc)       
        wandb.log({'Nature Acc': nature_acc})  # 记录使用随机权重的标准准确率
    else:
        logger.info('Evaluating with standard images...')
        _, nature_acc = evaluate_standard(device, test_loader, model)
        logger.info('Nature Acc: %.4f \t', nature_acc)
        wandb.log({'Nature Acc': nature_acc})
    
    print('PGD attacking')
    pgd_acc_list = []
    for i in range(10):
        atk = torchattacks.PGD(model, eps=8 / 255, alpha=2 / 255, steps=20, random_start=True)
        pgd_acc = evaluate_attack(device, model, test_loader, args, atk, 'pgd', logger)
        pgd_acc_list.append(pgd_acc)
        logger.info(f'PGD Acc Run {i+1}: %.4f \t', pgd_acc)

    pgd_mean_acc = np.mean(pgd_acc_list)
    pgd_std_acc = np.std(pgd_acc_list)
    logger.info('PGD Acc Mean: %.4f \t PGD Acc Std: %.4f \t', pgd_mean_acc, pgd_std_acc)
    wandb.log({'PGD Acc Mean': pgd_mean_acc, 'PGD Acc Std': pgd_std_acc})

    logger.info('Testing done.')
    # wandb.finish()


if __name__ == "__main__":
    main()