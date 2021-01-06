# -*- coding: utf-8 -*-
"""
Created on Mon Nov 18 01:05:36 2019

@author: User
"""
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
#from model import Discriminator,Generator,cifar_Discriminator,cifar_Generator
from model_dcgan import Discriminator,Generator
import time
import multiprocessing
#import dataset
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
    
    print('class :', np.unique(torch.as_tensor(new_dataset.targets).numpy()))
    print('num_of_sample :',len(new_data_loader.dataset))

    return new_dataset, new_data_loader
#%%
def main():
    
    if opt.dataset == 'cifar10':
        train_dataset = datasets.CIFAR10(root=opt.dataroot, download=True, transform=transforms.Compose([transforms.Resize(opt.imageSize),transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),]))
    
    elif opt.dataset == 'MNIST':
        train_dataset = datasets.MNIST(root=opt.dataroot, download=True, train=True, transform=transforms.Compose([transforms.Resize(opt.imageSize),transforms.ToTensor(),transforms.Normalize(mean=[0.5], std=[0.5])]))
    
    assert train_dataset
    
#%%    
    dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=opt.batchSize,shuffle=True, num_workers=n_cpu,drop_last=True)
    
    DEVICE=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # custom weights initialization called on netG and netD
    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            m.weight.data.normal_(0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            m.weight.data.normal_(1.0, 0.02)
            m.bias.data.fill_(0)
            
    if opt.dataset == 'MNIST':
        nc = 1
    else:
        nc = 3
        
    if opt.dataset == 'MNIST':
        D = Discriminator(opt.ngpu, nc, opt.ndf).to(DEVICE)
        G = Generator(opt.ngpu, opt.nz, opt.ngf, nc).to(DEVICE)
        D.apply(weights_init)
        G.apply(weights_init)
        
    elif opt.dataset == 'cifar10':
        D = Discriminator(opt.ngpu, nc, opt.ndf).to(DEVICE)
        G = Generator(opt.ngpu, opt.nz, opt.ngf, nc).to(DEVICE)
        D.apply(weights_init)
        G.apply(weights_init)
    
    D.to(DEVICE)
    G.to(DEVICE)

    criterion = nn.BCELoss().to(DEVICE)
    D_opt = torch.optim.Adam(D.parameters(), opt.lr, betas=(opt.beta1, 0.999))
    G_opt = torch.optim.Adam(G.parameters(), opt.lr, betas=(opt.beta1, 0.999))
    
    errG_list = []
    errD_r_list = []
    errD_f_list = []

    fixed_noise = torch.randn(128, opt.nz, 1, 1, device=DEVICE)
    
    dir_path=r'./dict/'
    new_dict_path = dir_path + opt.cls_num + '_' + opt.dataset
    with open(new_dict_path, 'rb') as filename:
        dictionary=pickle.load(filename)
    
    real_label = 1
    fake_label = 0

    imbalanced_dataset,imbalanced_data_loader=indexing_data(train_dataset,dataloader, dictionary[list(dictionary.keys())[0]],n_cpu)
    
    min_G_loss=float("inf")
    
    for epoch in range(opt.niter):
        for i, data in enumerate(imbalanced_data_loader, 0):
           
            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################
            # train with real
            D.zero_grad()
            real_cpu = data[0].to(DEVICE)
            batch_size = real_cpu.size(0)
            label = torch.full((batch_size,), real_label, device=DEVICE)
    
            output = D(real_cpu)
            errD_real = criterion(output, label)
            errD_real.backward()
            D_x = output.mean().item()
    
            # train with fake
            noise = torch.randn(batch_size, opt.nz, 1, 1, device=DEVICE)
            fake = G(noise)
            label.fill_(fake_label)
            output = D(fake.detach())
            errD_fake = criterion(output, label)
            errD_fake.backward()
            D_G_z1 = output.mean().item()
            errD = errD_real + errD_fake
            D_opt.step()
    
            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################
            G.zero_grad()
            label.fill_(real_label)  # fake labels are real for generator cost
            output = D(fake)
            errG = criterion(output, label)
            errG.backward()
            D_G_z2 = output.mean().item()
            G_opt.step()
            
            #save the output
            if i == len(imbalanced_data_loader)-1 :
                print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f' % (epoch, opt.niter, i, len(dataloader), errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))
                errG_list.append(errG)
                errD_r_list.append(errD_real)
                errD_f_list.append(errD_fake)
        
        # Check pointing for every epoch
                if errG < min_G_loss:
                        
                    min_G_loss = errG
                    model_path=r'./models/DC_GAN/'+opt.dataset
                    
                    if not os.path.exists(model_path):
                        os.makedirs(model_path)
    
                    torch.save(D.state_dict(), model_path+'/D_%s_epoch.pkl'%(epoch))
                    torch.save(G.state_dict(), model_path+'/G_%s_epoch.pkl'%(epoch))

        file_path=r'./samples/DC_GAN/Whole_Sampling/'+opt.dataset
        if not os.path.exists(file_path):
            os.makedirs(file_path)
            
        fake = G(fixed_noise)
        vutils.save_image(fake.detach(),r'./output_im/fake_samples_epoch_%03d.png' % (epoch),normalize=True)
#%% 그래프용

    plt.title('D&G'+' loss_change')
    plt.plot(np.arange(len(errG_list)), errG_list,color='b',label='errG')
    plt.plot(np.arange(len(errD_r_list)), errD_r_list,color='r',label='errD_r')
    plt.plot(np.arange(len(errD_f_list)), errD_f_list,color='g',label='errD_f')
    plt.legend()
    plt.savefig(file_path+'/loss_graph.png')
    plt.close()
    

#%%

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True,help='cifar10 | mnist', default='MNIST')
    parser.add_argument('--dataroot', help='path to dataset', default='./download_data')
    parser.add_argument('--batchSize', type=int, default=128, help='input batch size')
    parser.add_argument('--imageSize', type=int, default=64, help='the height / width of the input image to network')
    parser.add_argument('--nz', type=int, default=100, help='size of the latent z vector')
    parser.add_argument('--ngf', type=int, default=64)
    parser.add_argument('--ndf', type=int, default=64)
    parser.add_argument('--niter', type=int, default=50, help='number of epochs to train for')
    parser.add_argument('--lr', type=float, default=0.0002, help='learning rate, default=0.0002')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
    parser.add_argument('--cls_num', type=str, help='list_of_numbers', default='3_4')
    parser.add_argument('--imb_ratio', type=float,help='imb_ratio',default=0.8)
    parser.add_argument('--ngpu', type=int, help='num of gpu', default=1)
    opt = parser.parse_args()
    print(opt)
    
    #%%
    train_path=r'./data/untar/train/train/'
    n_cpu= multiprocessing.cpu_count()
    main()


    