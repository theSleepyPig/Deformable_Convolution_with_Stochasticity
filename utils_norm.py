# import apex.amp as amp
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import numpy as np
#from tqdm import tqdm

cifar10_mean = (0.4914, 0.4822, 0.4465)
cifar10_std = (0.2471, 0.2435, 0.2616)

mu = torch.tensor(cifar10_mean).view(3, 1, 1).to('cuda:1')
std = torch.tensor(cifar10_std).view(3, 1, 1).to('cuda:1')

upper_limit = torch.ones_like(mu)#((1 - mu) / std)
lower_limit = torch.zeros_like(mu)#((0 - mu) / std)


def clamp(X, lower_limit, upper_limit):
    return torch.max(torch.min(X, upper_limit), lower_limit)


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
def attack_pgd(model, X, y, epsilon, alpha, attack_iters, restarts):
    max_loss = torch.zeros(y.shape[0]).to('cuda:1')
    max_delta = torch.zeros_like(X).to('cuda:1')
    for zz in range(restarts):
        delta = torch.zeros_like(X).to('cuda:1')
        for i in range(len(epsilon)):
            delta[:, i, :, :].uniform_(-epsilon[i][0][0].item(), epsilon[i][0][0].item())
        delta.data = clamp(delta, lower_limit - X, upper_limit - X)
        delta.requires_grad = True
        for _ in range(attack_iters):
            output = model(X + delta)
            loss = F.cross_entropy(output, y)
            loss.backward()
            grad = delta.grad.detach()

            # attack all cases
            d = delta[:, :, :, :]
            g = grad[:, :, :, :]
            d = clamp(d + alpha * torch.sign(g), -epsilon, epsilon)
            d = clamp(d, lower_limit - X[:, :, :, :], upper_limit - X[:, :, :, :])
            delta.data = d

            delta.grad.zero_()
        all_loss = F.cross_entropy(model(X+delta), y, reduction='none').detach()
        max_delta[all_loss >= max_loss] = delta.detach()[all_loss >= max_loss]
        max_loss = torch.max(max_loss, all_loss)
    return max_delta


# evaluate on clean images with single norm
def evaluate_standard(test_loader, model):
    test_loss = 0
    test_acc = 0
    n = 0
    model.eval()

    with torch.no_grad():
        for i, (X, y) in enumerate(test_loader):
            X, y = X.to('cuda:1'), y.to('cuda:1')
            output = model(X)
            loss = F.cross_entropy(output, y)
            test_loss += loss.item() * y.size(0)
            test_acc += (output.max(1)[1] == y).sum().item()
            n += y.size(0)
    return test_loss/n, test_acc/n


# evaluate on clean images with random norms
def evaluate_standard_random_norms(test_loader, model, args):
    test_loss = 0
    test_acc = 0
    n = 0
    model.eval()

    with torch.no_grad():
        for i, (X, y) in enumerate(test_loader):
            if args.mixed:
                set_random_norm_mixed(args, model)
            else:
                set_random_norm(args, model)
            X, y = X.to('cuda:1'), y.to('cuda:1')
            output = model(X)
            loss = F.cross_entropy(output, y)
            test_loss += loss.item() * y.size(0)
            test_acc += (output.max(1)[1] == y).sum().item()
            n += y.size(0)
    return test_loss / n, test_acc / n


# evaluate on clean images with random weights
def evaluate_standard_random_weights(test_loader, model, args):
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
            X, y = X.to('cuda:1'), y.to('cuda:1')
            output = model(X)
            loss = F.cross_entropy(output, y)
            test_loss += loss.item() * y.size(0)
            test_acc += (output.max(1)[1] == y).sum().item()
            n += y.size(0)
            print(test_acc, n)
    return test_loss / n, test_acc / n


# evaluate on adv images with single norm
def evaluate_pgd(test_loader, model, attack_iters, restarts, args):
    epsilon = torch.ones_like(std) * (args.epsilon / 255.)# / std
    alpha = torch.ones_like(std) * (args.alpha / 255.)# / std
    pgd_loss = 0
    pgd_acc = 0
    n = 0
    model.eval()
    for i, (X, y) in enumerate(test_loader):
        X, y = X.to('cuda:1'), y.to('cuda:1')

        pgd_delta = attack_pgd(model, X, y, epsilon, alpha, attack_iters, restarts)

        with torch.no_grad():
            output = model(X + pgd_delta)
            loss = F.cross_entropy(output, y)
            pgd_loss += loss.item() * y.size(0)
            pgd_acc += (output.max(1)[1] == y).sum().item()
            n += y.size(0)

    return pgd_loss/n, pgd_acc/n


# evaluate on adv images with random norms
def evaluate_pgd_random_norms(test_loader, model, attack_iters, restarts, args, num_round=3):
    epsilon = torch.ones_like(std) * (args.epsilon / 255.)# / std
    alpha = torch.ones_like(std) * (args.alpha / 255.)# / std
    pgd_loss = 0
    pgd_acc = 0
    n = 0
    model.eval()

    for r in range(num_round):
        for i, (X, y) in enumerate(test_loader):
            X, y = X.to('cuda:1'), y.to('cuda:1')

            # random select a path to attack
            if args.mixed:
                set_random_norm_mixed(args, model)
            else:
                set_random_norm(args, model)

            pgd_delta = attack_pgd(model, X, y, epsilon, alpha, attack_iters, restarts)

            # random select a path to infer
            if args.mixed:
                set_random_norm_mixed(args, model)
            else:
                set_random_norm(args, model)

            with torch.no_grad():
                output = model(X + pgd_delta)
                loss = F.cross_entropy(output, y)
                pgd_loss += loss.item() * y.size(0)
                pgd_acc += (output.max(1)[1] == y).sum().item()
                n += y.size(0)

    return pgd_loss/n, pgd_acc/n


# evaluate on adv images with random weights
def evaluate_pgd_random_weights(test_loader, model, attack_iters, restarts, args, num_round=3):
    epsilon = torch.ones_like(std) * (args.epsilon / 255.)# / std
    alpha = torch.ones_like(std) * (args.alpha / 255.)# / std
    pgd_loss = 0
    pgd_acc = 0
    n = 0
    model.eval()

    for r in range(num_round):
        for i, (X, y) in enumerate(test_loader):
            X, y = X.to('cuda:1'), y.to('cuda:1')

            # random select a path to attack
            """if args.mixed:
                set_random_norm_mixed(args, model)
            else:"""
            model.set_rands()
            #set_random_weight(args, model)

            pgd_delta = attack_pgd(model, X, y, epsilon, alpha, attack_iters, restarts)

            # random select a path to infer
            """if args.mixed:
                set_random_norm_mixed(args, model)
            else:"""
            model.set_rands()
            #set_random_weight(args, model)

            with torch.no_grad():
                model.set_rands()
                output = model(X + pgd_delta)
                loss = F.cross_entropy(output, y)
                pgd_loss += loss.item() * y.size(0)
                pgd_acc += (output.max(1)[1] == y).sum().item()
                n += y.size(0)
                print(pgd_acc, n)

    return pgd_loss/n, pgd_acc/n


# random norm for entire network
def set_random_norm(args, model):
    norm_list = set_norm_list(args.num_group_schedule[0], args.num_group_schedule[1], args.random_type, args.gn_type)
    norm = np.random.choice(norm_list)
    model.module.set_normss(norm)
    return norm


# random norm for entire network
def set_random_weight(args, model):
    distribution_list = []#set_distribution_list(args.num_group_schedule[0], args.num_group_schedule[1], args.random_type, args.gn_type)
    distribution = []#np.random.choice(distribution_list)
    model.set_rands()
    return distribution

# setup random space for norms
def set_norm_list(min_group, max_group, random_type, gn_type):
    num_group_list = []
    for i in range(min_group, max_group + 1):
        num_group_list.append(2 ** i)
    # define norm list
    norm_list = []
    if 'bn' in random_type:
        norm_list.append('bn')
    if 'in' in random_type:
        norm_list.append('in')
    if '_' not in gn_type:
        for item in num_group_list:
            norm_list.append(gn_type + '_' + str(item))
    else:
        gn_str = gn_type[:gn_type.index('_')]
        gbn_str = gn_type[gn_type.index('_')+1:]
        for item in num_group_list:
            norm_list.append(gn_str + '_' + str(item))
            norm_list.append(gbn_str + '_' + str(item))

    return norm_list


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

