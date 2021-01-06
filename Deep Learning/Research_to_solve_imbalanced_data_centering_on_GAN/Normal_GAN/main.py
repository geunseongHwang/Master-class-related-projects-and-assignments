# -*- coding: utf-8 -*-
"""
Created on Mon Nov 18 01:05:36 2019

@author: User
"""
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
from model import Discriminator,Generator,cifar_Discriminator,cifar_Generator
import time
import multiprocessing
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.datasets as datasets
from matplotlib.pyplot import imshow, imsave
import pickle
import argparse
import torchvision.utils as vutils
import matplotlib.pyplot as plt
import copy
import torchvision.transforms as transforms

def indexing_data(dataset,data_loader, idx_array,n_cpu):
    
    new_dataset=copy.deepcopy(dataset)
    if opt.dataset == 'MNIST':
        new_dataset.targets = torch.from_numpy(data_loader.dataset.targets.numpy()[idx_array])
        new_dataset.data = torch.from_numpy(data_loader.dataset.data.numpy()[idx_array])

    elif opt.dataset == 'cifar10':
        new_dataset.targets = torch.as_tensor(data_loader.dataset.targets).numpy()[idx_array]
        new_dataset.data = torch.as_tensor(data_loader.dataset.data).numpy()[idx_array]
            
    new_data_loader = torch.utils.data.DataLoader(new_dataset, batch_size=opt.batchSize, shuffle=True, num_workers=n_cpu,drop_last=True)
    
    if opt.dataset == 'MNIST':
        print('class :', np.unique(new_dataset.targets.numpy()))

    elif opt.dataset == 'cifar10':
        print('class :', np.unique(new_dataset.targets))

    print('num_of_sample :',len(new_data_loader.dataset))

    return new_dataset, new_data_loader



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True ,help='cifar10 | mnist', default='cifar10')
    parser.add_argument('--dataroot', help='path to dataset', default='./download_data')
    parser.add_argument('--batchSize', type=int, default=128, help='input batch size')
    parser.add_argument('--imageSize', type=int, default=28, help='Mnist 28, cifar10 32')
    parser.add_argument('--nz', type=int, default=100, help='size of the latent z vector')
    parser.add_argument('--ngf', type=int, default=32)
    parser.add_argument('--ndf', type=int, default=32)
    parser.add_argument('--niter', type=int, default=50, help='number of epochs to train for')
    parser.add_argument('--lr', type=float, default=0.0002, help='learning rate, default=0.0002')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam')
    parser.add_argument('--cls_num', required=True ,type=str, help='list_of_numbers', default='3_4')
    parser.add_argument('--imb_ratio', type=float,help='imb_ratio',default=0.8)
    
    opt = parser.parse_args()
    print(opt)
    
    n_cpu= multiprocessing.cpu_count()

    
    if opt.dataset == 'cifar10':
        train_dataset = datasets.CIFAR10(root=opt.dataroot, download=True, transform=transforms.Compose([transforms.Resize(opt.imageSize),transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),]))
    
    
    elif opt.dataset == 'MNIST':
        train_dataset = datasets.MNIST(root=opt.dataroot, download=True, train=True, transform=transforms.Compose([transforms.Resize(opt.imageSize),transforms.ToTensor(),transforms.Normalize(mean=[0.5], std=[0.5])]))
    
    assert train_dataset
    
    dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=opt.batchSize,shuffle=True, num_workers=n_cpu,drop_last=True)
    
    
    
    DEVICE=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    
    if opt.dataset == 'MNIST':
        D = torch.nn.DataParallel(Discriminator(), device_ids=[0,1]).to(DEVICE)
        G = torch.nn.DataParallel(Generator(), device_ids=[0,1]).to(DEVICE)

    elif opt.dataset == 'cifar10':
        D = torch.nn.DataParallel(cifar_Discriminator(), device_ids=[0,1]).to(DEVICE)
        G = torch.nn.DataParallel(cifar_Generator(), device_ids=[0,1]).to(DEVICE)


    criterion = nn.BCELoss().to(DEVICE)
    D_opt = torch.optim.Adam(D.parameters(), opt.lr, betas=(opt.beta1, 0.999))
    G_opt = torch.optim.Adam(G.parameters(), opt.lr, betas=(opt.beta1, 0.999))

    
    min_G_loss=float("inf")
    
    D_loss_list=[]
    G_loss_list=[]
    D_labels = torch.ones(opt.batchSize, 1).to(DEVICE) # Discriminator Label to real
    D_fakes = torch.zeros(opt.batchSize, 1).to(DEVICE) # Discriminator Label to fake
    
    
    dir_path=r'../dict/'
    new_dict_path = dir_path + opt.cls_num + '_' + opt.dataset
    with open(new_dict_path, 'rb') as filename:
        dictionary=pickle.load(filename)
    
    imbalanced_dataset,imbalanced_data_loader=indexing_data(train_dataset,dataloader, dictionary[list(dictionary.keys())[0]],n_cpu)

    c_time=0
    
    for epoch in range(opt.niter):
        
        
        G_loss=0
        D_loss=0
        step=0
        
        epoch_time=0
        for idx, (images, labels) in enumerate(imbalanced_data_loader):
            D.zero_grad()
            start=time.time()
            
            # Training Discriminator
            x = images.to(DEVICE)
#            print(x.shape)
            x_outputs = D(x)
#            print(x_outputs.shape)
            
            D_x_loss = criterion(x_outputs, D_labels)
            
            z = torch.randn(opt.batchSize, opt.nz).to(DEVICE)
            z_outputs = D(G(z))
            D_z_loss = criterion(z_outputs, D_fakes)
            
            D_loss += (D_x_loss + D_z_loss)/(idx+1)
            (D_x_loss + D_z_loss).backward()
            D_opt.step()
    
            # Training Generator
            G.zero_grad()
            z = torch.randn(opt.batchSize, opt.nz).to(DEVICE)
            z_outputs = D(G(z))
            G_z_loss = criterion(z_outputs, D_labels)
            G_loss += G_z_loss/(epoch+1)
            G_z_loss.backward()
            G_opt.step()
            step+=1
            c_time+=time.time() - start
            epoch_time += time.time() - start
            
            if step == len(imbalanced_data_loader)-1:
                print('Epoch: %s/%s, D Loss: %.3f, G Loss: %.3f, time: %.3f, total: %.3f'%(epoch, opt.niter, D_loss.item(), G_loss.item(), epoch_time, c_time))
                D_loss_list.append(D_loss)
                G_loss_list.append(G_loss)
            
                if G_loss < min_G_loss:
            
                    min_G_loss = G_loss
                    model_path=r'../models/Normal_GAN/'+opt.dataset+opt.cls_num
                    
                    if not os.path.exists(model_path):
                        os.makedirs(model_path)
        
                    torch.save(D.state_dict(), model_path+'/D_%s_epoch.pkl'%(epoch))
                    torch.save(G.state_dict(), model_path+'/G_%s_epoch.pkl'%(epoch))
                step += 1

        G.eval()
        file_path=r'../samples/Normal_GAN/Whole_Sampling/'+opt.dataset+opt.cls_num
        
        if not os.path.exists(file_path):
            os.makedirs(file_path)
        
        fake = G(z)
        vutils.save_image(fake.data, file_path + '/%s_epoch.png' % (epoch))
        G.train()

        
            
    plt.title('D&G'+' loss_change')
    plt.plot(np.arange(len(G_loss_list)), G_loss_list,color='b',label='G_loss')
    plt.plot(np.arange(len(D_loss_list)), D_loss_list,color='r',label='D_loss')
    plt.legend()
    plt.savefig(file_path+'/loss_graph.png')
    plt.close()
        
def get_sample_image(G, n_noise,DEVICE):
    """
        save sample 100 images
    """
    z = torch.randn(100, n_noise).to(DEVICE)
    y_hat = G(z).view(100, 28, 28) # (100, 28, 28)
    result = y_hat.cpu().data.numpy()
    img = np.zeros([280, 280])
    for j in range(10):
        img[j*28:(j+1)*28] = np.concatenate([x for x in result[j*10:(j+1)*10]], axis=-1)
    
    return img
