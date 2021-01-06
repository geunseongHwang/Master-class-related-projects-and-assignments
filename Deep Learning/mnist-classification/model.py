
import torch.nn as nn

class LeNet5(nn.Module):
    """ LeNet-5 (LeCun et al., 1998)

        - For a detailed architecture, refer to the lecture note
        - Freely choose activation functions as you want
        - For subsampling, use max pooling with kernel_size = (2,2)
        - Output should be a logit vector
    """

    def __init__(self):
        super().__init__()
        self.Conv1 = nn.Conv2d(1,6,5,stride=1)
        self.MP1 = nn.MaxPool2d(2)
        self.Conv2 = nn.Conv2d(6,16,5,stride=1)
        self.MP2 = nn.MaxPool2d(2)
        
        self.fc1 = nn.Linear(400, 120, bias=True)
        self.fc2 = nn.Linear(120, 84, bias=True)
        self.fc3 = nn.Linear(84, 10, bias=True)
        
        # write your codes here

    def forward(self, img):
        output = self.Conv1(img)
        output = self.MP1(output) 
        output = self.Conv2(output)
        output = self.MP2(output)
        
        output = output.view(output.size(0), -1)
        output = self.fc1(output)
        output = self.fc2(output)
        output = self.fc3(output)
        # write your codes here

        return output


class CustomMLP(nn.Module):
    """ Your custom MLP model

        - Note that the number of model parameters should be about the same
          with LeNet-5
    """
    
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(1024, 58, bias=True)
        self.linear2 = nn.Linear(58, 34, bias=True)
        self.linear3 = nn.Linear(34, 10, bias=True)

        # write your codes here

    def forward(self, img):
        output = img.view(-1, 1024)
        output = self.linear1(output)
        output = self.linear2(output)
        output = self.linear3(output)
        # write your codes here

        return output




#from torchsummary import summary
#net = LeNet5()
#summary(net.cuda(), input_size=(1, 32, 32), batch_size=1)