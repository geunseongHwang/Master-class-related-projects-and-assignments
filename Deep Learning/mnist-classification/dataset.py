
# import some packages you need here

import torchvision.transforms as transforms

import  tarfile,os

from torch.utils.data import DataLoader, Dataset

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
        # 변수 지정
        self.data = os.listdir(data_dir)
        self.folder = data_dir
        self.transform_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize((0.1307, ), (0.3081,))
       
    def __len__(self):
        # 데이터 길이 # 데이터 범위 지정
        return len(self.data)
               
    def __getitem__(self, idx):
        # loader로 넘기고 실제 data를 리턴 
        img = self.data[idx]
        label = self.data[idx][-5] 
        label = int(label)
                
        img = Image.open(self.folder + "/" + img)
        
        # 28 X 28 picture => 32 X 32 picture
        Resize = transforms.Resize((32,32))
             
        img = Resize(img)
        
        img = self.transform_tensor(img)
        img = self.normalize(img)
        
        return img, label

if __name__ == '__main__':
    
    if os.path.isdir("../data/train_path") == True:
        pass

    else: 
        train_set = '../data/train.tar'
        train_path = '../data/train_path'
        os.makedirs(train_path)
        with tarfile.TarFile(train_set, 'r') as file:
            file.extractall(train_path)
    
    if os.path.isdir("../data/test_path") == True:
        pass
    
    else: 
        test_set = '../data/test.tar'
        test_path = '../data/test_path'
        os.makedirs(test_path)
        with tarfile.TarFile(test_set, 'r') as file:
            file.extractall(test_path)
    

