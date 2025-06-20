from Universal_Adversarial_Perturbation_pytorch.DeepFool.Python.deepfool import deepfool
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
        # image = Image.open(name)
        image = name
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
    # mean, std,tf = data_input_init(xi)
    mean=[0.485, 0.456, 0.406]
    std=[0.229, 0.224, 0.225]
    v = torch.zeros(1,3,256,256).to(device)
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
            img=name
            # img = tf(img)
            img = img.to(device)
            img = img.unsqueeze(0)
            _, pred = torch.max(model(img),1)
            _, adv_pred = torch.max(model(img+v),1)
            
            if(pred==adv_pred):
                dr, iter, _,_,_ = deepfool((img+v).squeeze(0),model, device, num_classes= num_classes,
                                             overshoot= overshoot,max_iter= max_iter_df)
                if(iter<max_iter_df-1):
                    v = v + torch.from_numpy(dr).to(device)
                    v = proj_lp(v,xi,p)
                    
            if(k%10==0):
                pbar.set_description('Norm of v: '+str(torch.norm(v).detach().cpu().numpy()))
        fooling_rate,model = get_fooling_rate(data_list,v,model, device)
        itr = itr + 1
    
    return v