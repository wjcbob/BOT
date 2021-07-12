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
from CNNs import ANN_T

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

def train_class(model, device, train_loader, optimizer, epoch):
    model.train()
    correct = 0
    for batch_idx, Data in enumerate(train_loader):    
        data = Data['data']
        target = Data['label'][:,0]
        data, target = data.to(device), target.to(device)
        output = model(data)

        loss = criterion(output, target)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if(batch_idx+1)%10 == 0: 
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
        pred = output.argmax(dim=1)
        correct += torch.eq(pred, target).float().sum().item()
    print('\nTrain set: \nAccuracy: {}/{} ({:.4f}%)\n'.format(
            correct, len(train_loader.dataset),
            100. * correct / len(train_loader.dataset)))
    return 100. * correct / len(train_loader.dataset)

def test_class(model, device, test_loader, epoch):
    model.eval()
    test_loss = 0
    correct = 0
    LABEL, PRED = np.array([]), np.array([])
    torch.save(model.state_dict(), path + '/' + str(epoch) + '_model.pkl')
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
    print(CORRECT / len(LABEL))
    return test_loss, 100. * correct / len(test_loader.dataset)    

def main(epoch):
    train_acc = train_class(model, DEVICE, train_loader, optimizer, epoch)
    loss, test_acc = test_class(model, DEVICE, test_loader, epoch)

if __name__ == '__main__':
    
    train_data = MyDataset('./data/tactile_train.mat')
    test_data = MyDataset('./data/tactile_test.mat')
    
    train_loader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE,\
                               shuffle=True, num_workers=0)
    test_loader = DataLoader(dataset=test_data, batch_size=BATCH_SIZE,\
                               shuffle=False, num_workers=0)
        
    path = './model/ANN_T'
    if not os.path.exists(path):
        os.makedirs(path)
        
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = ANN_T(10, 11).to(DEVICE)

    print(model)
    
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WD)   
    criterion = nn.CrossEntropyLoss().to(DEVICE) 
    
    t0 = time.time()
    for epoch in range(EPOCH):
        
        main(epoch)
        
        t1 = time.time()
        print('Time: ', t1-t0)