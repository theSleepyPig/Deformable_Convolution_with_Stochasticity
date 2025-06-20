import logging
import os
import datetime
import torchvision.models as models
import math
import torch
import yaml
from easydict import EasyDict
import shutil
import numpy as np


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(initial_lr, optimizer, epoch, n_repeats):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = initial_lr * (0.1 ** (epoch // int(math.ceil(30. / n_repeats))))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def fgsm(gradz, step_size):
    return step_size * torch.sign(gradz)


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def initiate_logger(output_path):
    if not os.path.isdir(output_path):
        os.makedirs(output_path)
    logging.basicConfig(level=logging.INFO, format='[%(asctime)s] - %(message)s', datefmt='%Y/%m/%d %H:%M:%S', )
    logger = logging.getLogger()
    logger.addHandler(logging.StreamHandler())
    logger.addHandler(logging.FileHandler(os.path.join(output_path, 'log.txt'), 'a+'))
    logger.info(pad_str(' LOGISTICS '))
    logger.info('Experiment Date: {}'.format(datetime.datetime.now().strftime('%Y-%m-%d %H:%M')))
    logger.info('Output Name: {}'.format(output_path))
    logger.info('User: {}'.format(os.getenv('USER')))
    return logger


def get_model_names():
    return sorted(name for name in models.__dict__
                  if name.islower() and not name.startswith("__")
                  and callable(models.__dict__[name]))


def pad_str(msg, total_len=70):
    rem_len = total_len - len(msg)
    return '*' * int(rem_len / 2) + msg + '*' * int(rem_len / 2)\

def parse_config_file(args):
    with open(args.config) as f:
        config = EasyDict(yaml.full_load(f))

    # Add args parameters to the dict
    for k, v in vars(args).items():
        config[k] = v

    return config


def save_checkpoint(state, is_best, filepath, timestamp):
    # Save the latest model with a timestamp
    filename = os.path.join(filepath, f'model_latest_{timestamp}.pth')
    torch.save(state, filename)
    
    # Save the best model with a timestamp
    if is_best:
        best_filename = os.path.join(filepath, f'model_best_{timestamp}.pth')
        shutil.copyfile(filename, best_filename)

