import torch
import argparse
import torch.nn as nn
from robustbench.data import get_preprocessing
from robustbench import benchmark

from utils import load_model, get_loaders

class DCSModel(nn.Module):
    def __init__(self, model_path, device):
        super().__init__()
        self.device = device
        
        
        args = argparse.Namespace(
            network='ResNet18', #WideResNet34 ResNet18
            
            none_random_training=True,
            rand_deform_training=False,
            randpos_deform_training=True,  # DCS
            randpos_multi_deform_training=False,
            only_adv_randpos_training=False,
            rand_path_training=False,
            
            data_dir='~/datasets/CIFAR100/', # placeholder
            batch_size=128,
            dataset='cifar100', #cifar10 cifar100
            worker=4,
            
            is_n_repeat=False,
            num_classes=100, # 10 100
            pretrain=model_path,
            # pos=0,
            # eot=True,
            # lb=2048
        )
        
        # Get the normalization layer from your utils.py
        _, _, dataset_normalization = get_loaders(args.data_dir, args.batch_size, dataset=args.dataset, worker=args.worker, norm=False)

        net = load_model(args=args)
        self.model = net(num_classes=args.num_classes, is_n_repeat=args.is_n_repeat, normalize=dataset_normalization, device=device).to(device)
        # self.model = net(num_classes=args.num_classes, normalize = dataset_normalization, device = device, pos = args.pos, eot = args.eot, lb = args.lb).to(device)
        print(self.model)
        
        # Load pretrained weights
        pretrained_model = torch.load(args.pretrain, map_location=device)
        if 'state_dict' in pretrained_model:
            partial = pretrained_model['state_dict']
        else:
            partial = pretrained_model
            
        state = self.model.state_dict()
        pretrained_dict = {k: v for k, v in partial.items() if k in list(state.keys()) and state[k].size() == partial[k].size()}
        state.update(pretrained_dict)
        self.model.load_state_dict(state)
        self.model.eval()

    def forward(self, x):
        # The hasattr check is good practice, but we know this model has the method.
        self.model.set_rand_mask()
        return self.model(x)

# Path to your pretrained model
# IMPORTANT: Please update this path to your actual model checkpoint
# PRETRAINED_MODEL_PATH = '/mnt/ssd_2/yxma/DCS/ckpt/cifar10/ResNet18/ckpt/model_20240928164626.pth'
# "/mnt/ssd_2/yxma/DCS/ckpt/cifar10/ResNet18/ckpt/model_20240928164626.pth" r18dcs c10
# PRETRAINED_MODEL_PATH = '/mnt/ssd_2/yxma/DCS/ckpt/cifar10/WideResNet34/ckpt/model_20241107185544.pth'
# "/mnt/ssd_2/yxma/DCS/ckpt/cifar10/WideResNet34/ckpt/model_20241107185544.pth" widedcs c10
# PRETRAINED_MODEL_PATH = '/mnt/ssd_2/yxma/DCS/ckpt/cifar10/ResNet18/ckpt/model_20240805133819.pth'
# "/mnt/ssd_2/yxma/DCS/ckpt/cifar10/ResNet18/ckpt/model_20240805133819.pth" pure at c10

PRETRAINED_MODEL_PATH = '/mnt/ssd_2/yxma/DCS/ckpt/cifar100/ResNet18/ckpt/model_20241107183716.pth'
# ""/mnt/ssd_2/yxma/DCS/ckpt/cifar100/ResNet18/ckpt/model_20241107183716.pth"" r18dcs c100
# PRETRAINED_MODEL_PATH = '/mnt/ssd_2/yxma/DCS/ckpt/cifar100/WideResNet34/ckpt/model_20241107185520.pth'
# "/mnt/ssd_2/yxma/DCS/ckpt/cifar100/WideResNet34/ckpt/model_20241107185520.pth" widedcs c100


device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

# The model name should follow the convention: <LastName><Year><FirstWordOfTheTitle>
model_name = "Ma2024Adversarial"

# Initialize your model
model = DCSModel(PRETRAINED_MODEL_PATH, device)

# Run the benchmark
clean_acc, robust_acc = benchmark(model,
                                  model_name=model_name,
                                  n_examples=10000,  # Using 1000 examples for a quicker test run
                                  dataset='cifar100',
                                  threat_model='Linf',
                                  eps=8/255,
                                  device=device,
                                  to_disk=True)

# print(f"Clean accuracy: {clean_acc:.2%}")
# print(f"Robust accuracy: {robust_acc:.2%}")