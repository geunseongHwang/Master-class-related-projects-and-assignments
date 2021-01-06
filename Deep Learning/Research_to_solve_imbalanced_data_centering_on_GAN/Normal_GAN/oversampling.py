# -*- coding: utf-8 -*-
"""
Created on Mon Nov 18 23:16:41 2019

@author: User
"""

import torch
from model import Discriminator,Generator,cifar_Discriminator,cifar_Generator
from main import get_sample_image
from matplotlib.pyplot import imshow, imsave
import os
from PIL import Image
import numpy as np
import argparse
from torchvision.utils import save_image

def get_sample_single_image(G, counts ,n_noise,DEVICE,file_path,save):
    #torch.radn(a,b) => b 크기의 데이터 a개를 만들어라
    z = torch.randn(counts, n_noise).to(DEVICE)
    
    y_hat = torch.squeeze(G(z), 1) # (100, 28, 28)
#    print(y_hat)
#    result = (y_hat.cpu().data.numpy()*255).astype(int)
#    print(result.shape)
#    print(result.shape[0])
#    print(result)

            
            
    if save == True:
        for index in range(y_hat.shape[0]):
            #new_image=Image.fromarray(y_hat[0].detach().numpy())
            #new_image=new_image.convert(mode='1')
            new_image=y_hat.data[index]
            #imsave(file_path+'\%s_%s.png'%(index,1), new_image, cmap='gray')
            save_image(new_image,file_path+'/%s_%s.png'%(index,opt.cls_num.split('_')[0]))
    #            print(result[index][:2])
    #            print(result[index].shape)
            print(index)
        
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

    DEVICE=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    n_noise=100 
    
    if opt.dataset == 'MNIST':
        model=torch.nn.DataParallel(Generator(n_noise),device_ids=[0])
    elif opt.dataset == 'cifar10':
        model=torch.nn.DataParallel(cifar_Generator(n_noise),device_ids=[0])


    model.load_state_dict(torch.load(path))
    model.eval()
    
    file_path=r'../oversampled_data/' + opt.model_name +'/' + opt.dataset +'/'+ opt.cls_num
    if not os.path.exists(file_path):   
        os.makedirs(file_path)
    
    counts=opt.counts
    
    get_sample_single_image(model, opt.counts, n_noise,DEVICE,file_path,True)


    '''
    if save == True:
        for index in range(result.shape[0]):
            imsave(file_path+'\%s_%s.png'%(index, index), Image.fromarray((result[index]*255).astype(int).reshape(28,28)), cmap='gray')
'''

