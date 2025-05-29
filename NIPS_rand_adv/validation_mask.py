from torchattacks import AutoAttack
from utils_mask_imagenet import *
import torch
import sys
import numpy as np
import time
from torch.autograd import Variable

import torchvision.utils as vutils

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

def validate_pgd_mask(val_loader, model, criterion, K, step, config, logger):
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
        if config.rd:
            model.module.set_rand_mask()

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
        if config.rd:
            model.module.set_rand_mask()

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
                    "PGD Test Prec@5": top5.avg
                })

    logger.info(' PGD Final Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
                .format(top1=top1, top5=top5))

    return top1.avg

def validate_pgd_mask_r(val_loader, model, criterion, K, step, config, logger):
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
        if config.rd:
            model.module.set_rand_mask()
            model.module.set_rand_mask()

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
        if config.rd:
            model.module.set_rand_mask()
            model.module.set_rand_mask()

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
                    "PGD Test Prec@5": top5.avg
                })

    logger.info(' PGD Final Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
                .format(top1=top1, top5=top5))

    return top1.avg

import torchattacks

def validate_aa_mask_r(val_loader, model, criterion, config, logger):
    """
    使用 AutoAttack 进行对抗鲁棒性评测，并支持随机 mask 选择。
    """
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

    eps = 8 / 255  # 设定 AutoAttack 的 epsilon
    model.eval()

    # 初始化 AutoAttack
    # attacker = torchattacks.AutoAttack(model, norm='Linf', eps=eps, version='standard', n_classes=1000)
    attacker = torchattacks.AutoAttack(
    model, norm='Linf', eps=eps, version='standard', n_classes=1000
)

    end = time.time()
    logger.info(pad_str(f' AutoAttack eps: {eps}, norm: Linf, version: standard '))

    for i, (input, target) in enumerate(val_loader):
        input = input.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        # **随机 mask 选择**
        if config.rd:
            model.module.set_rand_mask()
            # model.module.set_rand_mask()

        with torch.no_grad():
            # **AutoAttack 生成对抗样本**
            adv_images = attacker(input, target)
            
            # **计算噪声**
            noise = adv_images - input
            noise = torch.clamp(noise, -eps, eps)  # 限制噪声范围
            
            adv_images.sub_(mean).div_(std)  

            # **计算对抗样本上的输出**
            output = model(adv_images)
            loss = criterion(output, target)

            # **计算准确率**
            prec1, prec5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(prec1[0], input.size(0))
            top5.update(prec5[0], input.size(0))
            
            # **保存前几个样本进行可视化**
            if i == 0:
                num_samples = min(5, input.size(0))  # 只保存前5张图片
                img_grid = vutils.make_grid(input[:num_samples], nrow=5, normalize=True)
                adv_grid = vutils.make_grid(adv_images[:num_samples], nrow=5, normalize=True)
                noise_grid = vutils.make_grid(noise[:num_samples], nrow=5, normalize=True)

                # 记录到 WandB
                wandb.log({
                    "AA Original Images": [wandb.Image(img_grid, caption="Original Images")],
                    "AA Adversarial Images": [wandb.Image(adv_grid, caption="Adversarial Images")],
                    "AA Noise": [wandb.Image(noise_grid, caption="Noise")],
                })

            # **计算时间**
            batch_time.update(time.time() - end)
            end = time.time()

            # 计算时间消耗
            batch_time.update(time.time() - end)
            end = time.time()

            if i % 1 == 0:
                logger.info(
                    'AA Test: [{0}/{1}]\t'
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                    'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                    'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                        i, len(val_loader), batch_time=batch_time, loss=losses,
                        top1=top1, top5=top5
                    )
                )
                sys.stdout.flush()

                # 记录到 WandB
                wandb.log({
                    "AA Mask Test Loss": losses.avg,
                    "AA Mask Test Prec@1": top1.avg,
                    "AA Mask Test Prec@5": top5.avg
                })

    logger.info(' AA Mask Final Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
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


def validate_mask(val_loader, model, criterion, config, logger):
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
            if config.rd:
                model.module.set_rand_mask()

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

def validate_mask_r(val_loader, model, criterion, config, logger):
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
            if config.rd:
                model.module.set_rand_mask()
                model.module.set_rand_mask()

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

def validate_pgd_mask50(val_loader, model, criterion, K, step, config, logger):
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
        if config.rd:
            model.module.set_rand_mask()

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
        if config.rd:
            model.module.set_rand_mask()

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
                    "PGD50 Test Loss": losses.avg,
                    "PGD50 Test Prec@1": top1.avg,
                    "PGD50 Test Prec@5": top5.avg
                })

    logger.info(' PGD Final Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
                .format(top1=top1, top5=top5))

    return top1.avg

def validate_pgd_mask50_r(val_loader, model, criterion, K, step, config, logger):
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
        if config.rd:
            model.module.set_rand_mask()
            model.module.set_rand_mask()

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
        if config.rd:
            model.module.set_rand_mask()
            model.module.set_rand_mask()

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
                    "PGD50 Test Loss": losses.avg,
                    "PGD50 Test Prec@1": top1.avg,
                    "PGD50 Test Prec@5": top5.avg
                })

    logger.info(' PGD Final Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
                .format(top1=top1, top5=top5))

    return top1.avg



import time
import torchattacks

import time
import torchattacks
from tqdm import tqdm

def validate_aa(val_loader, model, criterion, config, logger, repeat=3):
    """
    使用 AutoAttack 进行对抗鲁棒性评测，并重复 `repeat` 次获取均值和标准差。
    额外增加 `tqdm` 进度条，并显示每个 run 的剩余时间估算。
    """
    logger.info(f"Starting AutoAttack evaluation: norm=Linf, eps={8/255}, version=standard")

    # 初始化 AutoAttack
    atk = torchattacks.AutoAttack(model, norm='Linf', eps=8/255, version='standard', n_classes=1000)

    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    adv_accs = []  # 存储多个 run 的对抗精度
    start_time = time.time()  # 记录整体开始时间

    for run in range(repeat):
        logger.info(f'AutoAttack - Run {run + 1}/{repeat}')
        run_start = time.time()  # 记录当前 run 的开始时间
        model.eval()
        end = time.time()

        correct = 0
        total = 0

        # 使用 `tqdm` 进度条
        with tqdm(total=len(val_loader), desc=f'AA Run {run + 1}/{repeat}', ncols=80, leave=True) as pbar:
            for i, (input, target) in enumerate(val_loader):
                input = input.cuda(non_blocking=True)
                target = target.cuda(non_blocking=True)

                with torch.no_grad():
                    # 生成对抗样本
                    adv_images = atk(input, target)  # torchattacks 里直接调用

                    # 计算对抗样本上的输出
                    output = model(adv_images)
                    loss = criterion(output, target)

                    # 计算准确率
                    prec1, prec5 = accuracy(output, target, topk=(1, 5))
                    losses.update(loss.item(), input.size(0))
                    top1.update(prec1[0], input.size(0))
                    top5.update(prec5[0], input.size(0))

                    # 记录正确预测的数量
                    correct += (output.argmax(dim=1) == target).sum().item()
                    total += target.size(0)

                    # 计算时间消耗
                    batch_time.update(time.time() - end)
                    end = time.time()

                    # 更新进度条
                    pbar.set_postfix(loss=f"{losses.avg:.4f}", acc=f"{top1.avg:.2f}%", time=f"{batch_time.val:.2f}s")
                    pbar.update(1)

        # 计算当前 run 的对抗精度
        adv_acc = correct / total
        adv_accs.append(adv_acc)

        run_time = time.time() - run_start  # 计算当前 run 运行时间
        remaining_time = run_time * (repeat - (run + 1))  # 估算剩余时间
        logger.info(f'AutoAttack - Run {run + 1}: Accuracy: {adv_acc:.4f}, Time: {run_time:.2f}s, Remaining: {remaining_time:.2f}s')

    # 计算均值和标准差
    mean_adv_acc = np.mean(adv_accs)
    std_adv_acc = np.std(adv_accs)

    total_time = time.time() - start_time  # 计算总运行时间
    logger.info(f'AutoAttack Final Mean Accuracy: {mean_adv_acc:.4f}, Std: {std_adv_acc:.4f}, Total Time: {total_time:.2f}s')

    # 记录到 WandB
    wandb.log({
        "AA Mean Accuracy": mean_adv_acc,
        "AA Std Accuracy": std_adv_acc,
        "AA Test Loss": losses.avg,
        "AA Test Prec@1": top1.avg,
        "AA Test Prec@5": top5.avg,
        "AA Total Time": total_time
    })

    return mean_adv_acc, std_adv_acc
