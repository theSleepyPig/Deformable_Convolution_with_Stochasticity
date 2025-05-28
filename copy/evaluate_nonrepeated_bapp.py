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

from QEBA.foolboxbased.foolbox.attacks.bapp import BoundaryAttackPlusPlus
from QEBA.foolboxbased.foolbox.attacks.bapp_custom import BAPP_custom
import QEBA.foolboxbased.foolbox as fb
import foolbox

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
    parser.add_argument('--randpos_deform_training', action='store_true', default=False,
                        help='able random mask training')
    parser.add_argument('--randpos_multi_deform_training', action='store_true', default=False,
                        help='able multi random mask training')
    parser.add_argument('--is_n_repeat', action='store_true', default=False,
                        help='Add to disable the repeat path in randomized kernel')
    parser.add_argument('--reNum', default=5, type=int)    
    
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
    parser.add_argument('--pretrain', default=os.path.abspath('ckpt/cifar10/ResNet18/ckpt/model_20240928164626.pth'), type=str, help='Path to load the pretrained model')
    parser.add_argument('--continue_training', action='store_true', help='Continue training at the checkpoint if exists')

    # CTRW settings
    parser.add_argument('--lb', default=2048, help='The Lower bound of sum of sigma.')
    parser.add_argument('--pos', default=0, help='The position of CTRW over the whole network.')
    parser.add_argument('--eot', action='store_true', help='Whether set random weight each step.')
    
    # BAPP++
    parser.add_argument('--bapp_iterations', default=1000, type=int, help='Iterations for Boundary Attack++')
    parser.add_argument('--bapp_stepsize', default='geometric_progression', type=str, choices=['adaptive', 'geometric_progression'], help='Stepsize search strategy for Boundary Attack++')
    parser.add_argument('--bapp_max_evals', default=100, type=int, help='Maximum number of evaluations per iteration')
    parser.add_argument('--bapp_initial_evals', default=100, type=int, help='Initial number of evaluations')

    # running settings
    parser.add_argument('--hang', action='store_true', help='Whether hang up. If yes, please add it. This will block "tqdm" mode to reduce the size of log file.')
    parser.add_argument('--device', default=0, type=int, help='CUDA device')

    return parser.parse_args()



def evaluate_attack(device, model, test_loader, args, atk, atk_name, logger, repeat=5, is_decision_based=False):
    """
    Repeat the attack `repeat` times, return the mean and variance of accuracy.
    """
    test_accs = []
    model.eval()
    state_dict = model.state_dict()
    
    for _ in range(repeat):
        start_time = time.time()
        test_loss = 0
        test_acc = 0
        n = 0
        model.load_state_dict(state_dict)
        test_loader_iter = iter(test_loader)

        bar_format = '{desc}[{elapsed}<{remaining},{rate_fmt}]'
        if args.hang:
            pbar = range(len(test_loader))
        else:
            pbar = tqdm(range(len(test_loader)), file=sys.stdout, bar_format=bar_format, ncols=80, disable=False)

        for i in pbar:
            X, y = test_loader_iter.__next__()
            X, y = X.to(device), y.to(device)

            if not args.none_random_training:
                model.set_rands()

            if args.randpos_deform_training:
                if isinstance(model, nn.DataParallel):
                    model.module.set_rand_mask()
                else:
                    model.set_rand_mask()

            if is_decision_based:
                # 使用 Foolbox 攻击接口
                X_adv = X_adv  # 使用第一个样本作为起点
            else:
                X_adv = atk(X, y)

            model.load_state_dict(state_dict)

            if not args.none_random_training:
                model.set_rands()

            if args.randpos_deform_training:
                if isinstance(model, nn.DataParallel):
                    model.module.set_rand_mask()
                else:
                    model.set_rand_mask()

            with torch.no_grad():
                output = model(X_adv)
            loss = F.cross_entropy(output, y)
            test_loss += loss.item() * y.size(0)
            test_acc += (output.max(1)[1] == y).sum().item()
            n += y.size(0)

        acc = test_acc / n
        test_accs.append(acc)
        
        end_time = time.time()
        total_time = end_time - start_time
        
        logger.info(f'{atk_name} - Run {i+1}: Accuracy: {acc:.4f}, Time Taken: {total_time:.2f} seconds')
    
    mean_acc = np.mean(test_accs)
    std_acc = np.std(test_accs)
    
    logger.info(f'{atk_name} Mean: {mean_acc:.4f}, Std: {std_acc:.4f}')
    wandb.log({f"{atk_name}_mean_adv": mean_acc, f"{atk_name}_std_adv": std_acc})
    
    return mean_acc, std_acc


def evaluate_nature_acc(device, model, test_loader, args, logger, repeat=5):
    """
    Repeat the evaluation of nature accuracy `repeat` times, return the mean and variance.
    """
    nature_accs = []
    
    for _ in range(repeat):
        if not args.none_random_training:
            logger.info('Evaluating with standard images with random weight...')
            _, nature_acc = evaluate_standard_random_weights(device, test_loader, model, args)
        elif args.randpos_deform_training:
            logger.info('Evaluating with standard images with random mask...')
            _, nature_acc = evaluate_standard_random_mask(device, test_loader, model, args)
        else:
            logger.info('Evaluating with standard images...')
            _, nature_acc = evaluate_standard(device, test_loader, model)

        nature_accs.append(nature_acc)

    mean_nature_acc = np.mean(nature_accs)
    std_nature_acc = np.std(nature_accs)
    
    logger.info(f'Nature Acc Mean: {mean_nature_acc:.4f}, Std: {std_nature_acc:.4f}')
    wandb.log({'Nature Acc Mean': mean_nature_acc, 'Nature Acc Std': std_nature_acc})
    
    return mean_nature_acc, std_nature_acc

def main():
    
    args = get_args()
    device = torch.device(args.device)

     # 初始化 wandb
    # wandb.init(project='-Test-R18-C100-4-1-nonrepeated', entity='xuanzhu_07-university-of-sydney', reinit=True, config=args)
    wandb.init(project='-Test-more', entity='xuanzhu_07-university-of-sydney', reinit=True, config=args)
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
    model = net(num_classes=args.num_classes, is_n_repeat=args.is_n_repeat, normalize=dataset_normalization, device = device).to(device)
# normalize=dataset_normalization,

    #model = torch.nn.DataParallel(model)
    print(model)

    # load pretrained model
    pretrained_model = torch.load(args.pretrain, map_location=device)#['state_dict']
    
    if 'state_dict' in pretrained_model:
        partial = pretrained_model['state_dict']
    else:# 如果没有 'state_dict'，直接使用预训练模型
        partial = pretrained_model

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
    

    # Evaluate nature accuracy 5 times
    print('Nature:')
    evaluate_nature_acc(device, model, test_loader, args, logger, repeat=1)
    
    print('BAPP attacking')
        
    # 获取一批数据
    X_sample, _ = next(iter(test_loader))
    # 打印数据的最小值和最大值
    print(f"Data range: min={X_sample.min()}, max={X_sample.max()}")
    # 初始化 Foolbox 模型
    # preprocessing = (np.array([104, 116, 123]), 1)  # 如果需要可以设置预处理
    fmodel = fb.models.pytorch.PyTorchModel(model, bounds=(0, 1), num_classes=args.num_classes)  # 注意 bounds 的范围
    
    atk = BoundaryAttackPlusPlus(  # 调用 Foolbox 内置的 BAPP 攻击
        fmodel
    )

    # 从测试集提取样本
    X_sample, y_sample = next(iter(test_loader))
    X_sample, y_sample = X_sample.to(device), y_sample.to(device)

    # 设置起点和目标
    starting_point = X_sample[0].squeeze(0)  # 使用第一张图片作为起点
    target_image = X_sample[1].squeeze(0)  # 使用第二张图片作为目标（可调整）


    # 使用起点执行攻击
    X_adv = atk(
        input_or_adv=target_image,
        label=y_sample[1],  # 目标的标签
        starting_point=starting_point
    )
    

    # 执行攻击
    evaluate_attack(device, model, test_loader, args, X_adv, 'bapp', logger, repeat=3, is_decision_based=True)
    
    # print('Square attacking')
    # atk = torchattacks.Square(model, eps=8/255, n_queries=5000, p_init=0.8, loss='ce')
    # evaluate_attack(device, model, test_loader, args, atk, 'square', logger, repeat=3)

    

    

        
    # print('Pixel attacking')
    # # atk = torchattacks.Pixle(model, pixels=1, steps=10, popsize=10, inf_batch=128)
    # atk = torchattacks.OnePixel(model, pixels=1, steps=10, popsize=10, inf_batch=128)
    # evaluate_attack(device, model, test_loader, args, atk, 'pixel', logger, repeat=3)

    # print('Square attacking')
    # atk = torchattacks.Square(model, eps=8/255, n_queries=5000, p_init=0.8, loss='ce')
    # evaluate_attack(device, model, test_loader, args, atk, 'square', logger, repeat=3)

    logger.info('Testing done.')
    # wandb.finish()


if __name__ == "__main__":
    main()