# -*- coding: utf-8 -*-
"""
Created on Mon Nov 18 23:16:41 2019

@author: User
"""

import torch
from model import netG, netD
from matplotlib.pyplot import imshow, imsave
import os
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
from torchvision.utils import save_image
import argparse

def get_sample_single_image(G, counts ,n_noise,DEVICE,file_path,save):
    #torch.radn(a,b) => b 크기의 데이터 a개를 만들어라
    z = torch.randn(counts, n_noise, 1, 1).to(DEVICE)
    y_hat = torch.squeeze(G(z), 1) # (100, 28, 28)
#    print(y_hat)
#    result = (y_hat.cpu().data.numpy()*255).astype(int)
#    print(result.shape)
#    print(type(result))
#    print(result)

            
            
    if save == True:
        for index in range(y_hat.shape[0]):
            new_image=y_hat.data[index]
            save_image(new_image,file_path+'/%s_%s.png'%(index,opt.cls_num.split('_')[0]))
            print(index)
            #print(new_image)
        
    return ''


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--cls_num', type=str)
    parser.add_argument('--nc', required=True,type=int, help='MNIST 1, cifar10 3')
    parser.add_argument('--ngf', type=int,default=32, help="path to netG (to continue training)")
    parser.add_argument('--counts', type=int)
    parser.add_argument('--model_name', help='Normal_GAN | AC_GAN')
    parser.add_argument('--model_epoch', type=str)
    parser.add_argument('--save', type=bool)
    opt = parser.parse_args()
    
    dir_path=r'../models/'
    path = dir_path+opt.model_name+'/'+opt.dataset+opt.cls_num+'/'+opt.model_epoch
#    path=r'.\models\G.pkl'

    DEVICE=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    n_noise=100 
    
    model=torch.nn.DataParallel(netG(n_noise, ngf=opt.ngf
                                     , nc=opt.nc), device_ids=[0])
    model.load_state_dict(torch.load(path))
    model.eval()

    file_path=r'../oversampled_data/' + opt.model_name + '/' + opt.dataset +'/'+ opt.cls_num
    if not os.path.exists(file_path):   
        os.makedirs(file_path)
    
    counts=opt.counts
    get_sample_single_image(model, opt.counts, n_noise,DEVICE,file_path,opt.save)


