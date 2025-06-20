from utils_first_imagenet import *
import torch
import sys
import numpy as np
import time
from torch.autograd import Variable

import wandb

def validate_pgd(val_loader, model, criterion, K, step, config, logger):
    # Mean/Std for normalization
    mean = torch.Tensor(np.array(config.TRAIN.mean)[:, np.newaxis, np.newaxis])
    mean = mean.expand(3, config.DATA.crop_size, config.DATA.crop_size).cuda()
    std = torch.Tensor(np.array(config.TRAIN.std)[:, np.newaxis, np.newaxis])
    std = std.expand(3, config.DATA.crop_size, config.DATA.crop_size).cuda()
    # Initiate the meters
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    eps = config.ADV.clip_eps
    model.eval()
    end = time.time()
    logger.info(pad_str(' PGD eps: {}, K: {}, step: {} '.format(eps, K, step)))
    for i, (input, target) in enumerate(val_loader):

        input = input.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        orig_input = input.clone()
        randn = torch.FloatTensor(input.size()).uniform_(-eps, eps).cuda()
        input += randn
        input.clamp_(0, 1.0)
        for _ in range(K):
            invar = Variable(input, requires_grad=True)
            in1 = invar - mean
            in1.div_(std)
            output = model(in1)
            ascend_loss = criterion(output, target)
            ascend_grad = torch.autograd.grad(ascend_loss, invar)[0]
            pert = fgsm(ascend_grad, step)
            # Apply purturbation
            input += pert.data
            input = torch.max(orig_input - eps, input)
            input = torch.min(orig_input + eps, input)
            input.clamp_(0, 1.0)

        input.sub_(mean).div_(std)
        with torch.no_grad():
            # compute output
            output = model(input)
            loss = criterion(output, target)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(prec1[0], input.size(0))
            top5.update(prec5[0], input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % config.TRAIN.print_freq == 0:
                logger.info('PGD Test: [{0}/{1}]\t'
                            'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                            'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                            'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                            'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                    i, len(val_loader), batch_time=batch_time, loss=losses,
                    top1=top1, top5=top5))
                sys.stdout.flush()
                
                
                wandb.log({
                    "PGD Test Loss": losses.avg,
                    "PGD Test Prec@1": top1.avg,
                    "PGD Test Prec@5": top5.avg,
                    "PGD Steps": K,
                    "PGD Eps": eps
                })

    logger.info(' PGD Final Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
                .format(top1=top1, top5=top5))

    return top1.avg


def validate_pgd_random(val_loader, model, criterion, K, step, config, logger):
    # Mean/Std for normalization
    mean = torch.Tensor(np.array(config.TRAIN.mean)[:, np.newaxis, np.newaxis])
    mean = mean.expand(3, config.DATA.crop_size, config.DATA.crop_size).cuda()
    std = torch.Tensor(np.array(config.TRAIN.std)[:, np.newaxis, np.newaxis])
    std = std.expand(3, config.DATA.crop_size, config.DATA.crop_size).cuda()
    # Initiate the meters
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    eps = config.ADV.clip_eps
    model.eval()
    end = time.time()
    logger.info(pad_str(' PGD eps: {}, K: {}, step: {} '.format(eps, K, step)))
    for i, (input, target) in enumerate(val_loader):

        input = input.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        # random select a path to attack
        if config.rp:
            model.module.set_rands()

        orig_input = input.clone()
        randn = torch.FloatTensor(input.size()).uniform_(-eps, eps).cuda()
        input += randn
        input.clamp_(0, 1.0)
        for _ in range(K):
            invar = Variable(input, requires_grad=True)
            in1 = invar - mean
            in1.div_(std)
            output = model(in1)
            ascend_loss = criterion(output, target)
            ascend_grad = torch.autograd.grad(ascend_loss, invar)[0]
            pert = fgsm(ascend_grad, step)
            # Apply purturbation
            input += pert.data
            input = torch.max(orig_input - eps, input)
            input = torch.min(orig_input + eps, input)
            input.clamp_(0, 1.0)

        # random select a path to attack
        if config.rp:
            model.module.set_rands()

        input.sub_(mean).div_(std)
        with torch.no_grad():
            # compute output
            output = model(input)
            loss = criterion(output, target)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(prec1[0], input.size(0))
            top5.update(prec5[0], input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % config.TRAIN.print_freq == 0:
                logger.info('PGD Test: [{0}/{1}]\t'
                            'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                            'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                            'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                            'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                    i, len(val_loader), batch_time=batch_time, loss=losses,
                    top1=top1, top5=top5))
                sys.stdout.flush()

    logger.info(' PGD Final Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
                .format(top1=top1, top5=top5))

    return top1.avg


def validate(val_loader, model, criterion, config, logger):
    # Mean/Std for normalization
    mean = torch.Tensor(np.array(config.TRAIN.mean)[:, np.newaxis, np.newaxis])
    mean = mean.expand(3, config.DATA.crop_size, config.DATA.crop_size).cuda()
    std = torch.Tensor(np.array(config.TRAIN.std)[:, np.newaxis, np.newaxis])
    std = std.expand(3, config.DATA.crop_size, config.DATA.crop_size).cuda()

    # Initiate the meters
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    # switch to evaluate mode
    model.eval()
    end = time.time()

    for i, (input, target) in enumerate(val_loader):
        with torch.no_grad():
            input = input.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)

            # compute output
            input = input - mean
            input.div_(std)
            output = model(input)
            loss = criterion(output, target)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(prec1[0], input.size(0))
            top5.update(prec5[0], input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % config.TRAIN.print_freq == 0:
                logger.info('Test: [{0}/{1}]\t'
                            'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                            'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                            'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                            'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                    i, len(val_loader), batch_time=batch_time, loss=losses,
                    top1=top1, top5=top5))
                sys.stdout.flush()
                
                # 在WandB中记录验证的指标 
                wandb.log({
                    "Validation Loss": losses.avg,
                    "Validation Prec@1": top1.avg,
                    "Validation Prec@5": top5.avg
                })

    logger.info(' Final Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
                .format(top1=top1, top5=top5))
    return top1.avg


def validate_random(val_loader, model, criterion, config, logger):
    # Mean/Std for normalization
    mean = torch.Tensor(np.array(config.TRAIN.mean)[:, np.newaxis, np.newaxis])
    mean = mean.expand(3, config.DATA.crop_size, config.DATA.crop_size).cuda()
    std = torch.Tensor(np.array(config.TRAIN.std)[:, np.newaxis, np.newaxis])
    std = std.expand(3, config.DATA.crop_size, config.DATA.crop_size).cuda()

    # Initiate the meters
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    # switch to evaluate mode
    model.eval()
    end = time.time()

    for i, (input, target) in enumerate(val_loader):
        with torch.no_grad():
            input = input.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)

            # compute output
            input = input - mean
            input.div_(std)

            # random select a path to attack
            if config.rp:
                model.module.set_rands()

            output = model(input)
            loss = criterion(output, target)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(prec1[0], input.size(0))
            top5.update(prec5[0], input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % config.TRAIN.print_freq == 0:
                logger.info('Test: [{0}/{1}]\t'
                            'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                            'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                            'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                            'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                    i, len(val_loader), batch_time=batch_time, loss=losses,
                    top1=top1, top5=top5))
                sys.stdout.flush()

    logger.info(' Final Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
                .format(top1=top1, top5=top5))
    return top1.avg

