import torch
from torchvision import datasets
from torchvision import transforms
from torch import nn, optim
from torch.nn import functional as F
import matplotlib.pyplot as plt
import scipy.io as sio
import numpy as np
from torch.utils.data import Dataset,DataLoader
from sklearn.metrics import confusion_matrix
import os
import time

EPOCH = 30
BATCH_SIZE = 1024
LR = 1e-3
WD = 0#1e-4

class MyDataset(Dataset):
    def __init__(self, ext):
        data = sio.loadmat(ext)
        self.data = data['data']
        self.label = data['label']
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        data = self.data[idx]
        label = self.label[idx]
        return {'data':torch.from_numpy(data).float(),
                'label':torch.from_numpy(label).long(),
                }

class ANN_O(nn.Module):
    def __init__(self, N, num_classes):
        super(ANN_O, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(N,16,3,1,1),nn.BatchNorm2d(16),nn.ReLU(inplace=True),
            nn.Conv2d(16,16,3,1,1),nn.BatchNorm2d(16),nn.ReLU(inplace=True),
            nn.MaxPool2d(2,2),
            nn.Conv2d(16,32,3,1,1),nn.BatchNorm2d(32),nn.ReLU(inplace=True),
            nn.Conv2d(32,32,3,1,1),nn.BatchNorm2d(32),nn.ReLU(inplace=True),
            nn.MaxPool2d(2,2),
        )
        self.fc1 = nn.Sequential(
            nn.Linear(64,128),
            nn.ReLU(inplace=True),
            nn.Dropout(),
        )
        self.fc2 = nn.Sequential(
            nn.Linear(128,64),
            nn.ReLU(inplace=True),
            nn.Dropout(),
        )
        self.fc3 = nn.Linear(64,num_classes)
        
    def forward(self, x):
        
        x = x.view(x.size(0), 1, 10, 6)
        out = self.features(x)
        
        out = out.view(out.size(0), -1)
        out_1 = self.fc1(out)
        out_2 = self.fc2(out_1)
        out_3 = self.fc3(out_2)
        
        return out

def deploy(epoch, device):
    model.eval()
    test_loss = 0
    correct = 0
    LABEL, PRED = np.array([]), np.array([])
    with torch.no_grad():
        for batch_idx, Data in enumerate(test_loader):    
            data = Data['data']
            target = Data['label'][:,0]

            data, target = data.to(device), target.to(device)
            output = model(data)
            
            pred = output.argmax(dim=1)
            correct += torch.eq(pred, target).float().sum().item()
            test_loss += criterion(output, target)
            
            LABEL = np.append(LABEL, target.cpu().numpy())
            PRED = np.append(PRED, pred.cpu().numpy())

    print('\nTest set: \nAverage loss: {:.6f} \nAccuracy: {}/{} ({:.4f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))
    
    C2 = confusion_matrix(LABEL, PRED, labels=range(11))
    print(C2)
    CORRECT = (PRED == LABEL).sum()
    print('\nEpoch: {} [({:.5f}%)]'.format(epoch, 100.0 * CORRECT / len(LABEL)))
    
    if not os.path.exists(path+'/pred'):
        os.makedirs(path+'/pred')
    sio.savemat(path+'/pred/pred_test_%d.mat' % epoch, {'pred':PRED, 'label':LABEL})
    
    return test_loss, 100. * correct / len(test_loader.dataset)

def main(start, end):
    t0 = time.time()
    acc_list = np.array([])
    for epoch in range(start, end):
        model_name = path+'/'+str(epoch)+'_model.pkl' 
        model.load_state_dict(torch.load(model_name))
        loss, test_acc = deploy(epoch, DEVICE)
        acc_list = np.append(acc_list, test_acc)
        t1 = time.time()
        print(t1-t0)
    print(acc_list)
    sio.savemat(path+'/acc_test.mat', {'acc':acc_list}) 

def show():
    data = sio.loadmat(path+'/acc_test.mat')
    acc_list = data['acc']
    print(acc_list[0])
    plt.ylim(0, 100)
    plt.plot(range(len(acc_list[0])),acc_list[0])
    plt.show()

def feature(epoch, device):
    model_name = path+'/'+str(epoch-1)+'_model.pkl' 
    model.load_state_dict(torch.load(model_name))
    model.eval()
    data_train, data_test = np.array([]), np.array([])
    label_train, label_test = np.array([]), np.array([])
    with torch.no_grad():
        for batch_idx, Data in enumerate(train_loader):    
            data = Data['data']
            target = Data['label'][:,0]

            data, target = data.to(device), target.to(device)
            output = model(data)
            
            data_train = np.append(data_train, output.cpu().numpy())
            label_train = np.append(label_train, target.cpu().numpy())
            
        for batch_idx, Data in enumerate(test_loader):    
            data = Data['data']
            target = Data['label'][:,0]

            data, target = data.to(device), target.to(device)
            output = model(data)
            
            data_test = np.append(data_test, output.cpu().numpy())
            label_test = np.append(label_test, target.cpu().numpy())
        
    data_train = data_train.reshape(-1,64)
    data_test = data_test.reshape(-1,64)
    label_train = label_train.reshape(-1,1)
    label_test = label_test.reshape(-1,1)
    print(data_train.shape,data_test.shape)
    print(label_train.shape,label_test.shape)
    
    sio.savemat('./data/odor_feature_train.mat', 
                {'data':data_train, 'label':label_train})
    sio.savemat('./data/odor_feature_test.mat', 
                {'data':data_test, 'label':label_test})

if __name__ == '__main__':
    
    train_data = MyDataset('./data/odor_train.mat')
    test_data = MyDataset('./data/odor_test.mat')
    
    train_loader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE,\
                               shuffle=False, num_workers=0)
    test_loader = DataLoader(dataset=test_data, batch_size=BATCH_SIZE,\
                               shuffle=False, num_workers=0)
        
    path = './model/ANN_O'
    if not os.path.exists(path):
        os.makedirs(path)
        
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = ANN_O(1, 11).to(DEVICE)

    print(model)
    
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WD)   
    criterion = nn.CrossEntropyLoss().to(DEVICE) 
    
    start, end = 0, 30
    
#    main(start, end)
    
    show()
    
#    feature(end, DEVICE)
    
    