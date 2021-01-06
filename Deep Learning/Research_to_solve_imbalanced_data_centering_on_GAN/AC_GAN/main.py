"""""""""
Pytorch implementation of Conditional Image Synthesis with Auxiliary Classifier GANs (https://arxiv.org/pdf/1610.09585.pdf).
This code is based on Deep Convolutional Generative Adversarial Networks in Pytorch examples : https://github.com/pytorch/examples/tree/master/dcgan
"""""""""
from __future__ import print_function
import argparse
import os
import numpy as np
import time
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
import multiprocessing
import model
import copy
import matplotlib.pyplot as plt
import pickle

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
    parser.add_argument('--dataset', required=True,help='cifar10 | mnist', default='MNIST')
    parser.add_argument('--dataroot', help='path to dataset', default='./download_data')
    parser.add_argument('--batchSize', type=int, default=128, help='input batch size')
    parser.add_argument('--imageSize', type=int, default=64, help='the height / width of the input image to network')
    parser.add_argument('--nz', type=int, default=100, help='size of the latent z vector')
    parser.add_argument('--ngf', type=int, default=32)
    parser.add_argument('--ndf', type=int, default=32)
    parser.add_argument('--niter', type=int, default=50, help='number of epochs to train for')
    parser.add_argument('--lr', type=float, default=0.0002, help='learning rate, default=0.0002')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
    parser.add_argument('--cls_num',required=True, help='list_of_numbers',default='3_4')
    parser.add_argument('--imb_ratio', type=float,help='imb_ratio')
    
    opt = parser.parse_args()
    print(opt)
    
    n_cpu= multiprocessing.cpu_count()
    DEVICE=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(opt.cls_num)
    cudnn.benchmark = True
    
    if opt.dataset == 'cifar10':
        train_dataset = dset.CIFAR10(root=opt.dataroot, download=True, transform=transforms.Compose([transforms.Resize(opt.imageSize),transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),]))
    
    
    elif opt.dataset == 'MNIST':
        train_dataset = dset.MNIST(root=opt.dataroot, download=True, train=True, transform=transforms.Compose([transforms.Resize(opt.imageSize),transforms.ToTensor(),transforms.Normalize(mean=[0.5], std=[0.5])]))
    
    
    assert train_dataset
    
    
    dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=opt.batchSize,shuffle=True, num_workers=n_cpu,drop_last=True)
    
    nz = int(opt.nz)
    ngf = int(opt.ngf)
    ndf = int(opt.ndf)
    if opt.dataset == 'MNIST':
        nc = 1
        nb_label = 10
    else:
        nc = 3
        nb_label = 10
    
    netG = torch.nn.DataParallel(model.netG(nz, ngf, nc), device_ids=[0,1])
    netD = torch.nn.DataParallel(model.netD(ndf, nc, nb_label), device_ids=[0,1])

    s_criterion = nn.BCELoss()
    c_criterion = nn.NLLLoss()
    
    if opt.dataset =='MNIST':
        input = torch.FloatTensor(opt.batchSize, 1, opt.imageSize, opt.imageSize)
    else:
        input = torch.FloatTensor(opt.batchSize, 3, opt.imageSize, opt.imageSize)
        
        
    real_label = 1
    fake_label = 0
    
    netD.to(DEVICE)
    netG.to(DEVICE)
    s_criterion.to(DEVICE)
    c_criterion.to(DEVICE)
    
    # setup optimizer
    optimizerD = optim.Adam(netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
    
    
    #test 함수가 있는 자리
    def test(predict, labels):
        correct = 0
        pred = predict.data.max(1)[1]
        correct = pred.eq(labels.data).cpu().sum()
        return correct, len(labels.data)
    
    time_start=0


    dir_path=r'../dict/'
    new_dict_path = dir_path + opt.cls_num + '_' + opt.dataset
    with open(new_dict_path, 'rb') as filename:
        dictionary=pickle.load(filename)
        
    imbalanced_dataset,imbalanced_data_loader=indexing_data(train_dataset,dataloader, dictionary[list(dictionary.keys())[0]],n_cpu)
    
    c_time=0
    
    errG_list = []
    errD_r_list = []
    errD_f_list = []

    min_G_loss=float("inf")
    for epoch in range(opt.niter):
        errG = 0
        errD_r=0
        errD_f= 0
        epoch_time=0
        for i, data in enumerate(imbalanced_data_loader):
            start=time.time()
            ##########################
            
            noise = torch.FloatTensor(opt.batchSize, nz, 1, 1).normal_(0,1)
            s_label = torch.FloatTensor(opt.batchSize)
            c_label = torch.LongTensor(opt.batchSize)
            
            
            input, s_label = input.to(DEVICE), s_label.to(DEVICE)
            c_label = c_label.to(DEVICE)
            noise = noise.to(DEVICE)
            
            input = Variable(input)
            s_label = Variable(s_label)
            c_label = Variable(c_label)
            noise = Variable(noise)

            random_label = np.random.randint(0, nb_label, opt.batchSize)
            #print('fixed label:{}'.format(random_label))
            
            random_onehot = np.zeros((opt.batchSize, nb_label))
            random_onehot[np.arange(opt.batchSize), random_label] = 1    
            
            ###########################
            # (1) Update D network
            ###########################
            # train with real
            netD.zero_grad()
            img, label = data
                        
            batch_size = img.size(0)
            input.data.resize_(img.size()).copy_(img).to(DEVICE)
            s_label.data.resize_(batch_size).fill_(real_label).to(DEVICE)
            c_label.data.resize_(batch_size).copy_(label).to(DEVICE)
            
            s_output, c_output = netD(input)
            s_errD_real = s_criterion(s_output, s_label).to(DEVICE)
            c_errD_real = c_criterion(c_output, c_label).to(DEVICE)
#            errD_real = s_errD_real + c_errD_real
            (s_errD_real + c_errD_real).backward()
            errD_r += (s_errD_real + c_errD_real)/(i+1)
            D_x = s_output.data.mean()
            
            correct, length = test(c_output, c_label)
            # train with fake
            noise.data.resize_(batch_size, nz, 1, 1)
            noise.data.normal_(0, 1)
    
            label = np.random.randint(0, nb_label, batch_size)
            noise_ = np.random.normal(0, 1, (batch_size, nz))
            label_onehot = np.zeros((batch_size, nb_label))
            label_onehot[np.arange(batch_size), label] = 1
            noise_[np.arange(batch_size), :nb_label] = label_onehot[np.arange(batch_size)]
            
            noise_ = (torch.from_numpy(noise_)).to(DEVICE)
            noise_ = noise_.resize_(batch_size, nz, 1, 1).to(DEVICE)
            noise.data.copy_(noise_).to(DEVICE)
    
            c_label.data.resize_(batch_size).copy_(torch.from_numpy(label)).to(DEVICE)
    
            fake = netG(noise).to(DEVICE)
            s_label.data.fill_(fake_label)
            s_output,c_output = netD(fake.detach())
            s_errD_fake = s_criterion(s_output, s_label)
            c_errD_fake = c_criterion(c_output, c_label)
            errD_f+= (s_errD_fake + s_errD_fake)/(i+1)
            (s_errD_fake + c_errD_fake).backward()
            optimizerD.step()
    
            ###########################
            # (2) Update G network
            ###########################
            netG.zero_grad()
            s_label.data.fill_(real_label)  # fake labels are real for generator cost
            s_output,c_output = netD(fake)
            s_errG = s_criterion(s_output, s_label)
            c_errG = c_criterion(c_output, c_label)
            
            errG += (s_errG + c_errG)/(i+1)
            (s_errG + c_errG).backward()
            optimizerG.step()
            
            epoch_time += time.time() - start
            c_time += time.time()-start

            if i == len(imbalanced_data_loader)-1 :
                print('[%d/%d] Loss_D_r: %.4f Loss_D_f: %.4f Loss_G: %.4f Accuracy: %.4f / %.4f = %.4f, time : %.4f, c_time=%.4f'% (epoch, opt.niter, errD_r.data,errD_f.data, errG.data, correct, length, 100.* correct / length, epoch_time, c_time))
                errG_list.append(errG)
                errD_r_list.append(errD_r)
                errD_f_list.append(errD_f)
                
                if errG < min_G_loss:
                    
                    min_G_loss = errG
                    model_path=r'../models/AC_GAN/'+opt.dataset+opt.cls_num
                    
                    if not os.path.exists(model_path):
                        os.makedirs(model_path)
    
                    torch.save(netD.state_dict(), model_path+'/D_%s_epoch.pkl'%(epoch))
                    torch.save(netG.state_dict(), model_path+'/G_%s_epoch.pkl'%(epoch))

        file_path=r'../samples/AC_GAN/Whole_Sampling/'+opt.dataset+opt.cls_num
        if not os.path.exists(file_path):
            os.makedirs(file_path)
            
        fake = netG(noise)
        vutils.save_image(fake.data, file_path + '/%s_epoch.png' % (epoch))
                
    plt.title('D&G'+' loss_change')
    plt.plot(np.arange(len(errG_list)), errG_list,color='b',label='errG')
    plt.plot(np.arange(len(errD_r_list)), errD_r_list,color='r',label='errD_r')
    plt.plot(np.arange(len(errD_f_list)), errD_f_list,color='g',label='errD_f')
    plt.legend()
    plt.savefig(file_path+'/loss_graph.png')
    plt.close()