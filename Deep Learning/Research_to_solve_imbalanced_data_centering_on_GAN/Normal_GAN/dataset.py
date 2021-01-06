# -*- coding: utf-8 -*-
"""
Created on Mon Nov 18 03:42:31 2019

@author: User
"""
import tarfile,os
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torchvision.transforms as transforms
import torchvision.datasets
from PIL import Image





class MNIST(Dataset):
    """ MNIST dataset

        To write custom datasets, refer to
        https://pytorch.org/tutorials/beginner/data_loading_tutorial.html

    Args:
        data_dir: directory path containing images

    Note:
        1) Each image should be preprocessed as follows:
            - First, all values should be in a range of [0,1]
            - Substract mean of 0.1307, and divide by std 0.3081
            - These preprocessing can be implemented using torchvision.transforms
        2) Labels can be obtained from filenames: {number}_{label}.png
    """

    def __init__(self, data_dir):

        # write your codes here
        
        
        self.trans=transforms.Compose([transforms.ToTensor(),
                                 transforms.Normalize((0.5,),(0.5,))])

        self.name_list=os.listdir(data_dir)
        self.data_dir=data_dir
        
    def __len__(self):

        return len(self.name_list)

    def __getitem__(self, idx):
        img_name=self.data_dir+self.name_list[idx]
        img=self.trans(Image.fromarray(np.array(Image.open(img_name).convert("L"))/255))

        label= int(self.name_list[idx].split('_')[1].split('.')[0])
            
        # write your codes here
        
        return img, label

#%%
if __name__ == '__main__':
    train_path = r'./data/untar/train/train/'
    test_path = r'../../data/untar/test/test/'
    
    batch_size=250
    
    MNIST_dataset = MNIST(data_dir=train_path)
    MNIST_loader= DataLoader(MNIST_dataset, batch_size=batch_size, shuffle=True)

    for batch_idx, (x,target) in enumerate(MNIST_loader):
        if batch_idx % 10 ==0:
            print(target[:10])
            print(x.shape, target.size()[0])
            print(len(MNIST_loader.dataset))
            
            
    '''
    train_dataset = torchvision.datasets.CIFAR10(root=r'D:', transform=torchvision.transforms.ToTensor(),download=True)
    target = train_dataset.train_labels
    class_sample_count = np.unique(target, return_counts=True)[1]
    print(class_sample_count)
    '''

    
    
    
    