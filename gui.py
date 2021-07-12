import PySimpleGUI as sg
import numpy as np
import matplotlib.pyplot as plt
import time
import os
import scipy.io as sio

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from torch import nn, optim
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from CNNs import ANN_OT, ANN_OTF

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

class ANN_T(nn.Module):
    def __init__(self, N, num_classes):
        super(ANN_T, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(N,16,3,1,1),nn.BatchNorm2d(16),nn.ReLU(inplace=True),
            nn.Conv2d(16,16,3,1,1),nn.BatchNorm2d(16),nn.ReLU(inplace=True),
            nn.MaxPool2d(2,2),
            nn.Conv2d(16,32,3,1,1),nn.BatchNorm2d(32),nn.ReLU(inplace=True),
            nn.Conv2d(32,32,3,1,1),nn.BatchNorm2d(32),nn.ReLU(inplace=True),
            nn.MaxPool2d(2,2),
        )
        self.fc1 = nn.Sequential(
            nn.Linear(96,128),
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
        
        out = self.features(x)
        
        out = out.view(out.size(0), -1)
        out_1 = self.fc1(out)
        out_2 = self.fc2(out_1)
        out_3 = self.fc3(out_2)
        
        return out

def ann_ot(odor, tactile):
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ANN_OT(1, 10, 11).to(DEVICE)
    model_name = './model/ANN_OT/0_model.pkl' 
    model.load_state_dict(torch.load(model_name))
    
    model.eval()
    odor, tactile = torch.from_numpy(odor).float().to(DEVICE), torch.from_numpy(tactile).float().to(DEVICE)
    output = model(odor, tactile)
    pred = output.argmax(dim=1)
    return pred.cpu().numpy()

def ann_otf(odor, tactile):
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    odor, tactile = torch.from_numpy(odor).float().to(DEVICE), torch.from_numpy(tactile).float().to(DEVICE)
    
    model = ANN_O(1, 11).to(DEVICE)
    model_name = './model/ANN_O/29_model.pkl' 
    model.load_state_dict(torch.load(model_name))
    model.eval()
    odor_feature = model(odor)
    
    model = ANN_T(10, 11).to(DEVICE)
    model_name = './model/ANN_T/29_model.pkl' 
    model.load_state_dict(torch.load(model_name))
    model.eval()
    tactile_feature = model(tactile)
    
    model = ANN_OTF(11).to(DEVICE)
    model_name = './model/ANN_OTF/29_model.pkl'
    model.load_state_dict(torch.load(model_name))
    model.eval()
    output = model(odor_feature, tactile_feature)
    
    pred = output.argmax(dim=1)
    return pred.cpu().numpy()


def deploy(path):
    DATA = sio.loadmat(path)
    odor = DATA['odor']
    tactile = DATA['tactile']
    label = DATA['label']
    
    for i in range(odor.shape[1]):
        odor[:,i] = odor[:,i] / np.max(odor[:,i])
    '''
    MAXX = [3.277, 3.4284, 1.5498, 1.2351, 1.2139, 1.4592, 1.2838,  
            1.0325, 1.3013, 1.2142, -0.4755, 1.192, 2.2637, 2.6088]
    MINN = [-2.2186, -2.2395, -2.2158, -2.2157, -2.3153, -2.355, -2.3177,
            -2.3045, -2.1806, -2.022, -2.0261, -1.771, -1.7644, -1.2846]
    for i in range(tactile.shape[2]):
        tactile[:,:,i] = (tactile[:,:,i] - MINN[i]) / (MAXX[i] - MINN[i])
    '''    
    tactile = (tactile + 3) / 7
    
    LENGTH = len(odor)
    NUMBER = 10
    ODOR = np.zeros((1,1,NUMBER,6))
    index = np.random.randint(0,int(LENGTH / NUMBER),NUMBER)
    for j in range(len(index)):
        index[j] = index[j] + j * int(LENGTH / NUMBER)
    ODOR[0,0] = odor[index]
    
    TACTILE = np.zeros((1,NUMBER,5,14))
    index = np.random.randint(0,int(LENGTH / NUMBER),NUMBER)
    for j in range(len(index)):
        index[j] = index[j] + j * int(LENGTH / NUMBER)
    TACTILE[0] = tactile[index]
    
    LABEL = label[0]   
    
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    PRED = ann_ot(ODOR, TACTILE)
    
#    PRED = ann_otf(ODOR, TACTILE)
    
    image = np.zeros((3,6))
    image[0] = ODOR[0,0,0]
    image[1] = ODOR[0,0,int(NUMBER / 2)]
    image[2] = ODOR[0,0,NUMBER-1]
    plt.figure(figsize=(3, 2), dpi=100)
    plt.title('Odor')
    plt.axis('off')
    plt.imshow(image, cmap='Blues', vmin=0, vmax=1)
    plt.savefig('6.png')
    plt.close()
    
    for i in range(tactile.shape[1]):
        for j in range(tactile.shape[2]):
            maxx = np.max(tactile[:,i,j])
            minn = np.min(tactile[:,i,j])
            TACTILE[:,:,i,j] = (TACTILE[:,:,i,j] - minn) / (maxx - minn)
    
    image = TACTILE[0,0]
    plt.figure(figsize=(1, 2), dpi=100)
    plt.title('Tactile_1')
    plt.axis('off')
    plt.imshow(image.T, cmap='Blues', vmin=0, vmax=1)
    plt.savefig('3.png')
    plt.close()
    
    image = TACTILE[0,3]
    plt.figure(figsize=(1, 2), dpi=100)
    plt.title('Tactile_2')
    plt.axis('off')
    plt.imshow(image.T, cmap='Blues', vmin=0, vmax=1)
    plt.savefig('4.png')
    plt.close()
    
    image = TACTILE[0,6]
    plt.figure(figsize=(1, 2), dpi=100)
    plt.title('Tactile_3')
    plt.axis('off')
    plt.imshow(image.T, cmap='Blues', vmin=0, vmax=1)
    plt.savefig('5.png')
    plt.close()
    
    print('Deploy Finish')
    return LABEL, PRED

def process(path):
    
    DATA = sio.loadmat(path)
    odor = DATA['odor']
    tactile = DATA['tactile']
    label = DATA['label']
    '''
    for i in range(odor.shape[1]):
        odor[:,i] = odor[:,i] / np.max(odor[:,i])
    
    for i in range(tactile.shape[1]):
        for j in range(tactile.shape[2]):
            minn = np.min(tactile[:,i,j])
            maxx = np.max(tactile[:,i,j])
            tactile[:,i,j] = (tactile[:,i,j] - minn) / (maxx - minn)
    '''
    '''
    tactile_sum = np.zeros((tactile.shape[0],tactile.shape[1]))
    for i in range(tactile.shape[0]):
        for j in range(tactile.shape[1]):
            tactile_sum[i,j] = tactile[i,j].sum()
    '''    
    tactile_sum = np.zeros((tactile.shape[0],1))
    for i in range(tactile.shape[0]):
        tactile_sum[i] = tactile[i].sum()
            
    image = np.copy(tactile_sum)
    plt.figure(figsize=(4, 2), dpi=100)
    plt.title('Tactile')
    plt.axis('off')
    for i in range(image.shape[1]):
        plt.plot(range(image.shape[0]), image[:,i], linewidth=1)
#    plt.legend(['finger1','finger2','finger3','finger4','finger5'])
    plt.savefig('1.png')
    plt.close()
    image = np.copy(odor)
    plt.figure(figsize=(4, 2), dpi=100)
    plt.title('Odor')
    plt.axis('off')
    for i in range(image.shape[1]):
        plt.plot(range(image.shape[0]), image[:,i], linewidth=1)
    plt.savefig('2.png')
    plt.close()

if __name__ == '__main__':
    
    print('Start!')
    
    classes = ('Orange', 'Towel', 'Arm', 'Stone', 'Head', 
               'Leg', 'Mouse', 'Clothes', 'Cup', 'Can', 'Carton')

first_column = [
    [sg.Text('Source Data',size=(10,1), auto_size_text=False, justification='center'), 
     sg.InputText('', size=(30,1), key='file1'), sg.FileBrowse('choose', key = 'choose1')],
    [sg.Button('Process', size=(50, 1))],
    [sg.Image(size=(400, 200), key='-IMAGE1-')],
    [sg.Image(size=(400, 200), key='-IMAGE2-')],
]

second_column = [
    [sg.Image(size=(100, 200), key='-IMAGE3-'),
     sg.Image(size=(100, 200), key='-IMAGE4-'),
     sg.Image(size=(100, 200), key='-IMAGE5-'),],
    [sg.Image(size=(300, 200), key='-IMAGE6-')],
    [sg.Button('Pred', size=(40, 1))],
    [sg.Button('Exit', size=(40, 1))],
]

third_column = [
    [sg.Text(key='-TOUT1-', font=16, size=(20, 1))],
    [sg.Text(key='-TOUT2-', font=16, size=(20, 1))],
    [sg.Text(key='-TOUT3-', font=16, size=(20, 1))],
    [sg.Text(key='-TOUT4-', font=16, size=(20, 1))],
]

layout = [
    [sg.Column(first_column),
     sg.VSeperator(),
     sg.Column(second_column),
     sg.VSeperator(),
     sg.Column(third_column),]
]

window = sg.Window("Object Recognition by BOT-M", layout)

while True:
    event, values = window.read()
    if event == "Exit" or event == sg.WIN_CLOSED:
        print('End!')
        break
    if (event == 'Process'):
        process(values['choose1'])
        window["-IMAGE1-"].update(filename='./1.png')
        window["-IMAGE2-"].update(filename='./2.png')
    if (event == 'Pred'):
        time_start = time.time()
        label, pred = deploy(values['choose1'])
        window["-IMAGE3-"].update(filename='./3.png')
        window["-IMAGE4-"].update(filename='./4.png')
        window["-IMAGE5-"].update(filename='./5.png')
        window["-IMAGE6-"].update(filename='./6.png')
        time_end = time.time()
        if (label == pred):
            result = 'Success'
        else:
            result = 'Failure'  
        window["-TOUT1-"].update('%8s %2.5fs' % ('  TIME:', time_end-time_start))
        window["-TOUT2-"].update('%8s %8s' % (' LABEL:', int(label[0])))
        window["-TOUT3-"].update('%8s %8s' % ('  PRED:', int(pred[0])))
        window["-TOUT4-"].update('%8s %8s' % ('RESULT:', result))
    
window.close()