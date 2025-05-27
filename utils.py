# import apex.amp as amp
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import numpy as np
#from tqdm import tqdm

import torchvision.models as models


cifar10_mean = (0.4914, 0.4822, 0.4465)
cifar10_std = (0.2471, 0.2435, 0.2616)

mu = torch.tensor(cifar10_mean).view(3, 1, 1)
std = torch.tensor(cifar10_std).view(3, 1, 1)

upper_limit = ((1 - mu) / std)
lower_limit = ((0 - mu) / std)

def load_baseline(name):
    if name == 'ResNet18':
        from model.resnet import ResNet18
        return ResNet18
    elif name == 'WideResNet34':
        from model.wide_resnet import WideResNet34
        return WideResNet34
    elif name == 'ResNet50':
        from model.resnet50 import ResNet50
        return ResNet50
    else:
        raise Exception("Not a good name.")

def load_rand(name):
    if name == 'ResNet18':
        from model.first_rand_resnet_lower_bound import ResNet18
        return ResNet18
    elif name == 'WideResNet34':
        from model.first_wide_resnet import WideResNet34
        return WideResNet34

def load_rand_deform(name):
    if name == 'ResNet18':
        from model.rand_resnet import ResNet18
        return ResNet18

def load_randpos_deform(args):
    if args.network == 'ResNet18':
        from model.randpos_multi_resnet import ResNet18_1layer
        return ResNet18_1layer
    elif args.network == 'WideResNet34':
        from model.widepos_resnet import WideResNet34_1layer
        return WideResNet34_1layer
    elif args.network == 'Vit':
        if args.randType == 'repeated':
            from model.vit_mask_repeated_7 import vit_b_16_mask
            return vit_b_16_mask
        elif args.randType == 'nonrepeated':
            from model.vit_mask_nonrepeated_7 import vit_b_16_mask
            return vit_b_16_mask
        elif args.randType == '2304':
            from model.vit_mask_2304 import vit_b_16_mask
            return vit_b_16_mask  
        elif args.randType == '2592':
            from model.vit_mask_2592 import vit_b_16_mask
            return vit_b_16_mask      
        elif args.randType == 'repeated5':
            from model.vit_mask_repeated import vit_b_16_mask
            return vit_b_16_mask
        elif args.randType == 'nonrepeated5':
            from model.vit_mask_nonrepeated import vit_b_16_mask
            return vit_b_16_mask  
def load_randpath(name):
    if name == 'ResNet18':
        from model.randpos_resnet import ResNet18_3
        return ResNet18_3
    
def load_multi_rand_deform(name):
    if name == 'ResNet18':
        from model.randpos_multi_resnet import ResNet18
        return ResNet18
    elif name == 'WideResNet34':
        from model.widepos_resnet import WideResNet34
        return WideResNet34
    elif name == 'ResNet50':
        from model.resnet50 import ResNet50_1
        return ResNet50_1
    
    
def load_model(args):
    if args.none_random_training:
        if args.rand_deform_training:
            return load_rand_deform(args.network)
        elif args.randpos_deform_training:
            if args.randpos_multi_deform_training:
                return load_multi_rand_deform(args.network)
            else:
                return load_randpos_deform(args)
        elif args.rand_path_training:
            return load_randpath(args.network)
        else:
            return load_baseline(args.network)

    else:
        return load_rand(args.network)

def clamp(X, lower_limit, upper_limit):
    return torch.max(torch.min(X, upper_limit.to(X.device)), lower_limit.to(X.device))


def normalize_fn(tensor, mean, std):
    """Differentiable version of torchvision.functional.normalize"""
    # here we assume the color channel is in at dim=1
    mean = mean[None, :, None, None]
    std = std[None, :, None, None]
    return (tensor.sub(mean)).div(std)


class NormalizeByChannelMeanStd(nn.Module):
    def __init__(self, mean, std):
        super(NormalizeByChannelMeanStd, self).__init__()
        if not isinstance(mean, torch.Tensor):
            mean = torch.tensor(mean)
        if not isinstance(std, torch.Tensor):
            std = torch.tensor(std)
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)

    def forward(self, tensor):
        return normalize_fn(tensor, self.mean, self.std)

    def extra_repr(self):
        return 'mean={}, std={}'.format(self.mean, self.std)


def get_loaders(dir_, batch_size, dataset='cifar10', worker=4, norm=True):
    
    if norm:
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(cifar10_mean, cifar10_std),
        ])
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(cifar10_mean, cifar10_std),
        ])
        dataset_normalization = None

    else:
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])
        test_transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        dataset_normalization = NormalizeByChannelMeanStd(
            mean=cifar10_mean, std=cifar10_std)

    if dataset == 'cifar10':
        train_dataset = datasets.CIFAR10(
            dir_, train=True, transform=train_transform, download=True)
        test_dataset = datasets.CIFAR10(
            dir_, train=False, transform=test_transform, download=True)
    elif dataset == 'cifar100':
        train_dataset = datasets.CIFAR100(
            dir_, train=True, transform=train_transform, download=True)
        test_dataset = datasets.CIFAR100(
            dir_, train=False, transform=test_transform, download=True)
    elif dataset == 'tiny-imagenet':
        train_dataset = datasets.ImageFolder(
            root="/mnt/ssd_2/yxma/tiny-imagenet-200/train", transform=train_transform)
        test_dataset = datasets.ImageFolder(
            root="/mnt/ssd_2/yxma/tiny-imagenet-200/val", transform=test_transform)
    else:
        print('Wrong dataset:', dataset)
        exit()

    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=worker,
    )
    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=worker,
    )
    return train_loader, test_loader, dataset_normalization

def get_224loaders(dir_, batch_size, dataset='cifar10', worker=4, norm=True):
    
    if norm:
        train_transform = transforms.Compose([
            transforms.Resize(224),  
            transforms.RandomCrop(224, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(cifar10_mean, cifar10_std),
        ])
        test_transform = transforms.Compose([
            transforms.Resize(224),  
            transforms.ToTensor(),
            transforms.Normalize(cifar10_mean, cifar10_std),
        ])
        dataset_normalization = None

    else:
        train_transform = transforms.Compose([
            transforms.Resize(224),  
            transforms.RandomCrop(224, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])
        test_transform = transforms.Compose([
            transforms.Resize(224),  
            transforms.ToTensor(),
        ])
        dataset_normalization = NormalizeByChannelMeanStd(
            mean=cifar10_mean, std=cifar10_std)
        
    if dataset == 'cifar10':
        train_dataset = datasets.CIFAR10(
            dir_, train=True, transform=train_transform, download=True)
        test_dataset = datasets.CIFAR10(
            dir_, train=False, transform=test_transform, download=True)
    elif dataset == 'cifar100':
        train_dataset = datasets.CIFAR100(
            dir_, train=True, transform=train_transform, download=True)
        test_dataset = datasets.CIFAR100(
            dir_, train=False, transform=test_transform, download=True)
    elif dataset == 'tiny-imagenet':
        train_dataset = datasets.ImageFolder(
            root="/mnt/ssd_2/yxma/tiny-imagenet-200/train", transform=train_transform)
        test_dataset = datasets.ImageFolder(
            root="/mnt/ssd_2/yxma/tiny-imagenet-200/val", transform=test_transform)
    else:
        print('Wrong dataset:', dataset)
        exit()

    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=worker,
    )
    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=worker,
    )
    return train_loader, test_loader, dataset_normalization

# pgd attack
def attack_pgd(model, X, y, epsilon, alpha, attack_iters, restarts, device):
    max_loss = torch.zeros(y.shape[0]).to(device)
    max_delta = torch.zeros_like(X).to(device)
    for zz in range(restarts):
        delta = torch.zeros_like(X).to(device)
        for i in range(len(epsilon)):
            delta[:, i, :, :].uniform_(-epsilon[i][0][0].item(), epsilon[i][0][0].item())
        delta.data = clamp(delta, lower_limit.to(device) - X, upper_limit.to(device) - X)
        delta.requires_grad = True
        for _ in range(attack_iters):
            output = model(X + delta)
            loss = F.cross_entropy(output, y)
            loss.backward()
            grad = delta.grad.detach()

            # attack all cases
            d = delta[:, :, :, :]
            g = grad[:, :, :, :]
            d = clamp(d + alpha.to(device) * torch.sign(g), -epsilon, epsilon)
            d = clamp(d, lower_limit.to(device) - X[:, :, :, :], upper_limit.to(device) - X[:, :, :, :])
            delta.data = d

            delta.grad.zero_()
        all_loss = F.cross_entropy(model(X+delta), y, reduction='none').detach()
        max_delta[all_loss >= max_loss] = delta.detach()[all_loss >= max_loss]
        max_loss = torch.max(max_loss, all_loss)
    return max_delta


# evaluate on clean images
def evaluate_standard(device, test_loader, model):
    test_loss = 0
    test_acc = 0
    n = 0
    model.eval()

    with torch.no_grad():
        for i, (X, y) in enumerate(test_loader):
            X, y = X.to(device), y.to(device)
            output = model(X)
            loss = F.cross_entropy(output, y)
            test_loss += loss.item() * y.size(0)
            test_acc += (output.max(1)[1] == y).sum().item()
            n += y.size(0)
    return test_loss/n, test_acc/n


# evaluate on clean images with random weights
def evaluate_standard_random_weights(device, test_loader, model, args):
    test_loss = 0
    test_acc = 0
    n = 0
    model.eval()
    #model.set_rands()
    #model.set_rands()
    #model.set_rands()

    with torch.no_grad():
        for i, (X, y) in enumerate(test_loader):
            model.set_rands()
            X, y = X.to(device), y.to(device)
            output = model(X)
            loss = F.cross_entropy(output, y)
            test_loss += loss.item() * y.size(0)
            test_acc += (output.max(1)[1] == y).sum().item()
            n += y.size(0)
            #print(test_acc, n)
    return test_loss / n, test_acc / n


# evaluate on clean images with random mask
def evaluate_standard_random_mask(device, test_loader, model, args):
    test_loss = 0
    test_acc = 0
    n = 0
    model.eval()
    #model.set_rands()
    #model.set_rands()
    #model.set_rands()

    with torch.no_grad():
        for i, (X, y) in enumerate(test_loader):
            if isinstance(model, nn.DataParallel):
                model.module.set_rand_mask()
            else:
                model.set_rand_mask()
            X, y = X.to(device), y.to(device)
            output = model(X)
            loss = F.cross_entropy(output, y)
            test_loss += loss.item() * y.size(0)
            test_acc += (output.max(1)[1] == y).sum().item()
            n += y.size(0)
            #print(test_acc, n)
    return test_loss / n, test_acc / n

# evaluate on adv images
def evaluate_pgd(device, test_loader, model, attack_iters, restarts, args):
    epsilon = (args.epsilon / 255.) / std
    alpha = (args.alpha / 255.) / std
    pgd_loss = 0
    pgd_acc = 0
    n = 0
    model.eval()
    for i, (X, y) in enumerate(test_loader):
        X, y = X.to(device), y.to(device)

        pgd_delta = attack_pgd(model, X, y, epsilon, alpha, attack_iters, restarts, device)

        with torch.no_grad():
            output = model(X + pgd_delta)
            loss = F.cross_entropy(output, y)
            pgd_loss += loss.item() * y.size(0)
            pgd_acc += (output.max(1)[1] == y).sum().item()
            n += y.size(0)

    return pgd_loss/n, pgd_acc/n


# evaluate on adv images with random weights
def evaluate_pgd_random_weights(device, test_loader, model, attack_iters, restarts, args, num_round=3):
    epsilon = (args.epsilon / 255.) / std
    alpha = (args.alpha / 255.) / std
    pgd_loss = 0
    pgd_acc = 0
    n = 0
    model.eval()

    for r in range(num_round):
        for i, (X, y) in enumerate(test_loader):
            X, y = X.to(device), y.to(device)

            # random select a path to attack
            model.set_rands()
            #set_random_weight(args, model)

            pgd_delta = attack_pgd(model, X, y, epsilon, alpha, attack_iters, restarts, device)

            # random select a path to infer
            model.set_rands()
            #set_random_weight(args, model)

            with torch.no_grad():
                model.set_rands()
                output = model(X + pgd_delta)
                loss = F.cross_entropy(output, y)
                pgd_loss += loss.item() * y.size(0)
                pgd_acc += (output.max(1)[1] == y).sum().item()
                n += y.size(0)
                #print(pgd_acc, n)

    return pgd_loss/n, pgd_acc/n


# evaluate on adv images with random mask
def evaluate_pgd_random_mask(device, test_loader, model, attack_iters, restarts, args, num_round=3):
    epsilon = (args.epsilon / 255.) / std
    alpha = (args.alpha / 255.) / std
    pgd_loss = 0
    pgd_acc = 0
    n = 0
    model.eval()
    
    if isinstance(model, nn.DataParallel):
        modelto = model.module
    else:
        modelto = model

    for r in range(num_round):
        for i, (X, y) in enumerate(test_loader):
            X, y = X.to(device), y.to(device)

            # random select a path to attack
            
            if isinstance(model, nn.DataParallel):
                model.module.set_rand_mask()
            else:
                model.set_rand_mask()
            #set_random_weight(args, model)

            pgd_delta = attack_pgd(model, X, y, epsilon, alpha, attack_iters, restarts, device)

            # random select a path to infer
            
            if isinstance(model, nn.DataParallel):
                model.module.set_rand_mask()
            else:
                model.set_rand_mask()
            #set_random_weight(args, model)

            with torch.no_grad():
                output = model(X + pgd_delta)
                loss = F.cross_entropy(output, y)
                pgd_loss += loss.item() * y.size(0)
                pgd_acc += (output.max(1)[1] == y).sum().item()
                n += y.size(0)
                #print(pgd_acc, n)

    return pgd_loss/n, pgd_acc/n



# random weight for entire network
def set_random_weight(args, model):
    distribution_list = []#set_distribution_list(args.num_group_schedule[0], args.num_group_schedule[1], args.random_type, args.gn_type)
    distribution = []#np.random.choice(distribution_list)
    model.set_rands()
    return distribution


def set_distribution_list(min_group, max_group, random_type, dtb_type):
    num_group_list = []
    for i in range(min_group, max_group + 1):
        num_group_list.append(2 ** i)
    distribution_list = []
    if 'bn' in random_type:
        distribution_list.append('bn')
    if 'in' in random_type:
        distribution_list.append('in')
    if '_' not in dtb_type:
        for item in num_group_list:
            distribution_list.append(dtb_type + '_' + str(item))
    else:
        dtb_str = dtb_type[:dtb_type.index('_')]
        gbn_str = dtb_type[dtb_type.index('_')+1:]
        for item in num_group_list:
            distribution_list.append(dtb_str + '_' + str(item))
            distribution_list.append(gbn_str + '_' + str(item))

    return distribution_list

