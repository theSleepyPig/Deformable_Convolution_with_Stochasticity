from UAP.DeepFool.Python.deepfool import deepfool
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import numpy as np
import torch
import torch.optim as optim
import torch.utils.data as data_utils
# from torch.autograd.gradcheck import zero_gradients
import math
from PIL import Image
import torchvision.models as models
import sys
import random
import time
from tqdm import tqdm

def zero_gradients(x):
    if x.grad is not None:
        x.grad.detach_()
        x.grad.zero_()

def get_model(model,device):
    if model == 'vgg16':
        net = models.vgg16(pretrained=True)
    elif model =='resnet18':
        net = models.resnet18(pretrained=True)
    
    net.eval()
    net=net.to(device)
    return net

def data_input_init(xi):
    mean = [ 0.485, 0.456, 0.406 ]
    std = [ 0.229, 0.224, 0.225 ]
    transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean = mean,
                         std = std)])
    
    return (mean,std,transform)

def proj_lp(v, xi, p):
    # Project on the lp ball centered at 0 and of radius xi
    if p==np.inf:
            v=torch.clamp(v,-xi,xi)
    else:
        v=v * min(1, xi/(torch.norm(v,p)+0.00001))
    return v

def get_fooling_rate(data_list,v,model, device):
    """
    :data_list: list of image paths
    :v: Noise Matrix
    :model: target network
    :device: PyTorch device
    """
    tf = data_input_init(0)[2]
    num_images = len(data_list)
    
    fooled=0.0
    
    for name in tqdm(data_list):
        image = Image.open(name)
        image = tf(image)
        image = image.unsqueeze(0)
        image = image.to(device)
        _, pred = torch.max(model(image),1)
        _, adv_pred = torch.max(model(image+v),1)
        if(pred!=adv_pred):
            fooled+=1

    # Compute the fooling rate
    fooling_rate = fooled/num_images
    print('Fooling Rate = ', fooling_rate)
    for param in model.parameters():
        param.requires_grad = False
    
    return fooling_rate,model

def universal_adversarial_perturbation(data_list, model, device, xi=10, delta=0.2, max_iter_uni = 10, p=np.inf, 
                                       num_classes=10, overshoot=0.02, max_iter_df=10,t_p = 0.2):
    """
    :data_list: list of image paths
    :model: target network
    :device: PyTorch Device
    :param xi: controls the l_p magnitude of the perturbation
    :param delta: controls the desired fooling rate (default = 80% fooling rate)
    :param max_iter_uni: optional other termination criterion (maximum number of iteration, default = 10*num_images)
    :param p: norm to be used (default = np.inf)
    :param num_classes: For deepfool: num_classes (limits the number of classes to test against, by default = 10)
    :param overshoot: For deepfool: used as a termination criterion to prevent vanishing updates (default = 0.02).
    :param max_iter_df:For deepfool: maximum number of iterations for deepfool (default = 10)
    :param t_p:For deepfool: truth perentage, for how many flipped labels in a batch.(default = 0.2)
    
    :return: the universal perturbation matrix.
    """
    time_start = time.time()
    mean, std,tf = data_input_init(xi)
    v = torch.zeros(1,3,224,224).to(device)
    v.requires_grad_()
    
    
    fooling_rate = 0.0
    num_images =  len(data_list)
    itr = 0
    
    while fooling_rate < 1-delta and itr < max_iter_uni:
        random.shuffle(data_list)
        # Iterate over the dataset and compute the purturbation incrementally
        pbar=tqdm(data_list)
        pbar.set_description('Starting pass number '+ str(itr))
        for k,name in enumerate(pbar):
            # img = Image.open(name)
            img = name
            img = tf(img)
            img = img.to(device)
            img = img.unsqueeze(0)
            # _, pred = torch.max(model(img),1)
            # _, adv_pred = torch.max(model(img+v),1)
            features = model.forward_features(img)
            features.shape
            features = F.adaptive_avg_pool2d(features, (1, 1))  # 全局池化，确保匹配 FC 层
            features.shape
            features = features.view(features.size(0), -1)  # 变成 [batch, 512]
            print(features.shape)
            pred = torch.argmax(model.linear(features), dim=1)

            features_adv = model.forward_features(img + v)
            # print(features_adv.shape)
            # if features_adv.shape[2] < 4:  # 避免 avg_pool2d 计算后输出为 0
            #     features_adv = F.interpolate(features_adv, size=(4, 4), mode="bilinear", align_corners=False)
            print(features_adv.shape)
            features_adv = F.adaptive_avg_pool2d(features_adv, (1, 1))
            print(features_adv.shape)
            features_adv = features_adv.view(features_adv.size(0), -1)
            print(features_adv.shape)
            # features_adv.shape
            adv_pred = torch.argmax(model.linear(features_adv), dim=1)
            
            
            if(pred==adv_pred):
                # if features_adv.shape[0] < 4:  # 避免 avg_pool2d 计算后输出为 0
                #     features_adv = F.interpolate(features_adv, size=(4, 4), mode="bilinear", align_corners=False)
                dr, iter, _,_,_ = deepfool(features_adv ,model, device, num_classes= num_classes,
                                             overshoot= overshoot,max_iter= max_iter_df)
                if(iter<max_iter_df-1):
                    dr = torch.from_numpy(dr).to(device)
                    
                    # 如果 dr 维度不符合 (batch, channels, H, W)，需要扩展
                    if dr.dim() == 3:  # (channels, H, W) → (1, channels, H, W)
                        dr = dr.unsqueeze(0)
                    elif dr.dim() == 2:  # (H, W) → (1, 1, H, W)
                        dr = dr.unsqueeze(0).unsqueeze(0)

                    if dr.shape != v.shape:
                        dr = F.interpolate(dr, size=v.shape[2:], mode="bilinear", align_corners=False)
                        
                    v = v + dr
                    v = proj_lp(v, xi, p)
                    # v = v + torch.from_numpy(dr).to(device)
                    # v = proj_lp(v,xi,p)
                    
            if(k%10==0):
                pbar.set_description('Norm of v: '+str(torch.norm(v).detach().cpu().numpy()))
        fooling_rate,model = get_fooling_rate(data_list,v,model, device)
        itr = itr + 1
    
    # while fooling_rate < 1-delta and itr < max_iter_uni:
    #     random.shuffle(data_list)
    #     # Iterate over the dataset and compute the purturbation incrementally
    #     pbar=tqdm(data_list)
    #     pbar.set_description('Starting pass number '+ str(itr))
    #     for k, name in enumerate(pbar):
    #         img = name
    #         img = tf(img)
    #         # img = img.to(device)
    #         # img = img.unsqueeze(0) if img.dim() == 3 else img  # 确保 batch 维度
            
    #         img = img.to(device)
    #         img = img if img.dim() == 4 else img.unsqueeze(0)  # 确保 batch 维度

    #         # print("img shape before forward:", img.shape)  # Debug
    #         features = model.forward_features(img)


    #         # print("img shape before forward:", img.shape)s

    #         features = model.forward_features(img)
    #         # print("features.shape after forward_features:", features.shape)

    #         if features.shape[2] < 4:  # 避免 avg_pool2d 出现 0 维度
    #             features = F.avg_pool2d(features, 2)

    #         features = F.adaptive_avg_pool2d(features, (1, 1))
    #         features = features.view(features.size(0), -1)
    #         pred = torch.argmax(model.linear(features), dim=1)

    #         features_adv = model.forward_features(img + v)
    #         features_adv = F.adaptive_avg_pool2d(features_adv, (1, 1))
    #         features_adv = features_adv.view(features_adv.size(0), -1)
    #         adv_pred = torch.argmax(model.linear(features_adv), dim=1)

    #     if pred == adv_pred:
    #         img_adv = (img + v).detach()
    #         # print("img_adv shape before deepfool:", img_adv.shape)  # Debugging
    
    #         # 适配 ResNetPartmask5 的输入格式
    #         features_adv = model.forward_features(img_adv)  
    #         features_adv = F.adaptive_avg_pool2d(features_adv, (1, 1))
    #         features_adv = features_adv.view(features_adv.size(0), -1)  # 变为 [batch, 512]

    #         # print("features_adv shape before linear:", features_adv.shape)  # Debugging
    #         adv_pred = torch.argmax(model.linear(features_adv), dim=1)

    #         if pred == adv_pred:
    #             dr, iter, _, _, _ = deepfool(img_adv.squeeze(0), model, device, num_classes=num_classes,
    #                                          overshoot=overshoot, max_iter=max_iter_df)

    #         if iter < max_iter_df - 1:
    #             v = v + torch.from_numpy(dr).to(device)
    #             v = proj_lp(v, xi, p)

                    
    #         if(k%10==0):
    #             pbar.set_description('Norm of v: '+str(torch.norm(v).detach().cpu().numpy()))
    #     fooling_rate,model = get_fooling_rate(data_list,v,model, device)
    #     itr = itr + 1
        
        
    
    return v