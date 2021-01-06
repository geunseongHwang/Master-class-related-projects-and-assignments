
import dataset
from model import LeNet5, CustomMLP

# import some packages you need here
import numpy as np

from torch.utils.data import DataLoader, Dataset
import torch

import torch.optim as optim
import matplotlib.pyplot as plt

def train(model, trn_loader, device, criterion, optimizer):
    """ Train function

    Args:
        model: network
        trn_loader: torch.utils.data.DataLoader instance for training
        device: device for computing, cpu or gpu
        criterion: cost function
        optimizer: optimization method, refer to torch.optim

    Returns:
        trn_loss: average loss value
        acc: accuracy
    """

    trn_loss = 0 
    acc = 0
    
    total_batch = len(trn_loader)# 전체 batch가 몇이 되는지 알 수 있음
    
    for X, Y in trn_loader:

        X_train = X.to(device) # input # cuda연산을 진행하려면 torch.cuda.Tensor가 되어야함
        Y_train = Y.to(device) # label # cuda연산을 진행하려면 torch.cuda.Tensor가 되어야함, 그래야 명령어가 실행 가능
        
        prediction = model(X_train) # output값이 가설이 됌 
        
        cost = criterion(prediction, Y_train) # hypothesis와 label사이에차이 계산
        optimizer.zero_grad()
        cost.backward()
        optimizer.step() # model 학습
        
        correct_prediction = torch.argmax(prediction, 1) == Y_train
        trn_loss += criterion(prediction, Y_train)
        acc += correct_prediction.float().mean()
        #print("Loss : [{0}], ACC : [{1}]".format(trn_loss, acc))

    # write your codes here
    acc = acc/total_batch
    trn_loss = trn_loss/total_batch
    
    #print("Training Loss : [{0}], ACC : [{1}]".format(trn_loss, acc))

    return trn_loss, acc

def test(model, tst_loader, device, criterion):
    """ Test function

    Args:
        model: network
        tst_loader: torch.utils.data.DataLoader instance for testing
        device: device for computing, cpu or gpu
        criterion: cost function

    Returns:
        tst_loss: average loss value
        acc: accuracy
    """
    tst_loss = 0
    acc = 0
    
    total_batch = len(tst_loader)
    
    with torch.no_grad():
        
        for X,Y in tst_loader:
            
            X_test = X.to(device)
            Y_test = Y.to(device)
            
            prediction = model(X_test)
            correct_prediction = torch.argmax(prediction, 1) == Y_test 
            tst_loss += criterion(prediction, Y_test)
            acc += correct_prediction.float().mean()
            #print("Loss : [{0}], ACC : [{1}]".format(tst_loss, acc))
    
    # write your codes here
    tst_loss = tst_loss/total_batch
    acc = acc/total_batch
    
    #print("test Loss : [{0}], ACC : [{1}]".format(tst_loss, acc))
    
    return tst_loss, acc


def main():
    """ Main function

        Here, you should instantiate
        1) Dataset objects for training and test datasets
        2) DataLoaders for training and testing
        3) model
        4) optimizer: SGD with initial learning rate 0.01 and momentum 0.9
        5) cost function: use torch.nn.CrossEntropyLoss

    """
    #공통
    train_losses_list = []
    train_accs_list = []
    test_losses_list = []
    test_accs_list = []
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    criterion = torch.nn.CrossEntropyLoss().to(device)
    
    model = CustomMLP().to(device)
    # train
    train_path = '../data/train_path/train/'
    
    dataset_train = dataset.MNIST(train_path)
    trn_loader = DataLoader(dataset_train, batch_size = 150)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum = 0.9)    
    
    # test
    test_path = '../data/test_path/test/'

    dataset_test = dataset.MNIST(test_path)
    
    tst_loader = DataLoader(dataset_test, batch_size = 150)
    
    # loss, accuracy 및 epoch
    n_epoch = 10
  
    for epoch in range(n_epoch):
        train_function = train(model, trn_loader, device, criterion, optimizer)
        print('train_cost =', '{:.9f}'.format(train_function[0]))
        print('train_accuracy =', '{:.9f}'.format(train_function[1]))
        
        test_function = test(model, tst_loader, device, criterion)
        print('test_cost =', '{:.9f}'.format(test_function[0]))
        print('test_accuracy =', '{:.9f}'.format(test_function[1]))
        
        train_losses = train_function[0].item()
        train_accs = train_function[1].item()
        test_losses = test_function[0].item()
        test_accs = test_function[1].item()
        
        train_losses_list.append(train_losses)
        train_accs_list.append(train_accs)
        test_losses_list.append(test_losses)
        test_accs_list.append(test_accs)
    
    
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111)
    ax1.plot(train_losses_list)
    plt.xticks(np.arange(0, n_epoch, step=1))
    plt.title('CustomMLP Training Loss at Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Training Loss')
    
    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111)
    ax2.plot(train_accs_list)
    plt.xticks(np.arange(0, n_epoch, step=1))
    plt.title('CustomMLP Training Acc at Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Training Acc')
   
    fig3 = plt.figure()
    ax3 = fig3.add_subplot(111)
    ax3.plot(test_losses_list)
    plt.xticks(np.arange(0, n_epoch, step=1))
    plt.title('CustomMLP Test Loss at Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Test Loss')

    fig4 = plt.figure()
    ax4 = fig4.add_subplot(111)
    ax4.plot(test_accs_list)
    plt.xticks(np.arange(0, n_epoch, step=1))
    plt.title('CustomMLP Test Acc at Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Test Acc')
        
    # write your codes here


if __name__ == '__main__':
    main()
    

    
    

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

