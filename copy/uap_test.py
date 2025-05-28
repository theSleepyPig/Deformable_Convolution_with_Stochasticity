import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import random
from tqdm import tqdm
from PIL import Image

def deepfool(image, model, device, num_classes=10, overshoot=0.02, max_iter=50):
    image = image.unsqueeze(0).to(device)
    pert_image = image.clone().detach().to(device)
    pert_image.requires_grad_()  # 关键：开启梯度计算

    model.eval()
    with torch.no_grad():
        output = model(image)
    pred_label = torch.argmax(output, dim=1).item()

    w = torch.zeros_like(image).to(device)
    loop_i = 0

    while loop_i < max_iter:
        pert_image.requires_grad_()
        output = model(pert_image)

        if torch.argmax(output) != pred_label:
            break

        loss = output[0, pred_label]
        model.zero_grad()
        loss.backward(retain_graph=True)

        if pert_image.grad is None:
            raise ValueError("pert_image.grad is None, check requires_grad settings!")

        grad = pert_image.grad.data
        w += grad

        pert_image = image + (1 + overshoot) * torch.sign(w)
        loop_i += 1

    return (pert_image - image).detach().cpu().numpy()



def proj_lp(v, xi, p=np.inf):
    """
    投影到 Lp 球
    """
    if p == np.inf:
        v = torch.clamp(v, -xi, xi)
    else:
        v = v * min(1, xi / (torch.norm(v, p) + 1e-8))
    return v


def universal_adversarial_perturbation(dataloader, model, device, xi=10, delta=0.2, max_iter_uni=10, num_classes=10):
    """
    计算 UAP（Universal Adversarial Perturbation）
    """
    v = torch.zeros(1, 3, 224, 224).to(device)
    fooling_rate = 0.0
    itr = 0

    while fooling_rate < 1 - delta and itr < max_iter_uni:
        for images, _ in tqdm(dataloader, desc=f"UAP Iter {itr}"):
            images = images.to(device)

            orig_preds = torch.argmax(model(images), dim=1)
            perturbed_images = images + v
            adv_preds = torch.argmax(model(perturbed_images), dim=1)

            for i in range(len(images)):
                if orig_preds[i] == adv_preds[i]:
                    dr = deepfool(images[i], model, device, num_classes)
                    dr = torch.from_numpy(dr).to(device)

                    v = v + dr
                    v = proj_lp(v, xi)

        fooling_rate = compute_fooling_rate(dataloader, model, v, device)
        print(f"Fooling Rate: {fooling_rate:.4f}")
        itr += 1

    return v


def compute_fooling_rate(dataloader, model, v, device):
    """
    计算欺骗率（Fooling Rate）
    """
    total, fooled = 0, 0
    for images, _ in dataloader:
        images = images.to(device)
        orig_preds = torch.argmax(model(images), dim=1)
        adv_preds = torch.argmax(model(images + v), dim=1)
        fooled += (orig_preds != adv_preds).sum().item()
        total += len(images)
    return fooled / total


# ======= 载入数据 & 运行 UAP  =======
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载预训练模型
model = models.resnet18(pretrained=True).to(device)
model.eval()

# 预处理
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 加载 CIFAR-10 数据集（可以换成 ImageNet 或其他）
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = DataLoader(trainset, batch_size=32, shuffle=True)

# 运行 UAP 生成通用扰动
uap = universal_adversarial_perturbation(trainloader, model, device)
print("Final UAP Generated!")
