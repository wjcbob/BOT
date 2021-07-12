import scipy.io as sio
import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt
from scipy import interpolate

def load_odor():
    NAME = ['橘子','毛巾','人体胳膊','石头','头','腿','小鼠','衣服','直筒水杯','易拉罐','纸盒']
    LENGTH = 5000
    NUMBER = 10
    data_odor_train, label_odor_train = np.array([]), np.array([])
    data_odor_test, label_odor_test = np.array([]), np.array([])
    for name in range(len(NAME)):
        path = './data/odor/'+NAME[name]
        for num in range(4):
            execl_name = path+'/'+str(num+1)+'.xlsx'
            print(execl_name)
            data = pd.read_excel(execl_name, header=None)
            data = np.copy(data)
            data = data[1:,1:]
            for i in range(data.shape[1]):
                data[:,i] = data[:,i] / np.max(data[:,i])
            
            data_sum = np.zeros((data.shape[0],1))
            for i in range(data.shape[0]):
                data_sum[i] = data[i].sum()
            plt.ylim(0,6)
            plt.plot(range(len(data_sum)),data_sum[:,0])
            plt.title(str(name)+'  '+str(num))
            plt.show()
            
            ####差值
            DATA = np.zeros((LENGTH, 6))
            Label = np.zeros((LENGTH,1)) + name
            x = np.linspace(0,data.shape[0]-1,data.shape[0])
            xnew = np.linspace(0,data.shape[0]-1,LENGTH)
            for i in range(data.shape[1]):
                f = interpolate.interp1d(x,data[:,i],kind='cubic')
                DATA[:,i] = f(xnew)
            
            Data = np.zeros((LENGTH,NUMBER,6))
            for i in range(Data.shape[0]):
                index = np.random.randint(0,int(LENGTH / NUMBER),NUMBER)
                for j in range(len(index)):
                    index[j] = index[j] + j * int(LENGTH / NUMBER)
                Data[i] = DATA[index]
            print(data.shape,DATA.shape,Data.shape)
            
            if (num == 3):
                data_odor_test = np.append(data_odor_test, Data)
                label_odor_test = np.append(label_odor_test, Label)
            else:
                data_odor_train = np.append(data_odor_train, Data)
                label_odor_train = np.append(label_odor_train, Label)
            
    data_odor_train = data_odor_train.reshape(-1,NUMBER,6)
    data_odor_test = data_odor_test.reshape(-1,NUMBER,6)
    label_odor_train = label_odor_train.reshape(-1,1)
    label_odor_test = label_odor_test.reshape(-1,1)
    print(data_odor_train.shape,label_odor_train.shape)
    
#    for i in range(11):
#        for j in range(3):
#            plt.ylim(0,1)
#            plt.plot(range(LENGTH),data_odor_train[LENGTH*(i*3+j):LENGTH*(i*3+j+1)])
#            plt.show()
            
    sio.savemat('./data/odor_train.mat', {'data':data_odor_train ,'label':label_odor_train})
    sio.savemat('./data/odor_test.mat', {'data':data_odor_test ,'label':label_odor_test})
            
def load_tactile():
    NAME = ['橘子','毛巾','人体胳膊','石头','头','腿','小鼠','衣服','直筒水杯','易拉罐','纸盒']
    FINGER = ['拇指', '食指', '中指', '无名指', '小指']
    LENGTH = 5000
    NUMBER = 10
    data_tactile_train, label_tactile_train = np.array([]), np.array([])
    data_tactile_test, label_tactile_test = np.array([]), np.array([])
    for name in range(len(NAME)):
        path = './data/tactile/'+NAME[name]
        for num in range(4):
            file_name = path+'/'+'第'+str(num+1)+'组'
            DATA = np.zeros((LENGTH, 5, 14))
            Label = np.zeros((LENGTH,1)) + name
            for finger in range(5):
                execl_name = file_name+'/'+FINGER[finger]+'.xlsx'
                print(execl_name)
                data = pd.read_excel(execl_name, header=None)
                data = np.copy(data)
                if (data.shape[1] == 15):
                    data = np.delete(data, 10, 1)
                
                list_data = pd.date_range('1/1/2000', periods=data.shape[0], freq='T') 
                data_1 = pd.DataFrame(data)
                data_1.index = list_data
                data_obj = data_1.resample('%dT' % int(data.shape[0] / LENGTH), label='right') 
                data_new = data_obj.asfreq()[0:]
                print(data_new.shape, data.shape)

                DATA[:,finger,:] = data_new[:LENGTH]
                
            Data = np.zeros((LENGTH, NUMBER, 5, 14))
            for i in range(Data.shape[0]):
                index = np.random.randint(0,int(LENGTH / NUMBER),NUMBER)
                for j in range(len(index)):
                    index[j] = index[j] + j * int(LENGTH / NUMBER)
                Data[i] = DATA[index]
            
            if (num == 3):
                data_tactile_test = np.append(data_tactile_test, Data)
                label_tactile_test = np.append(label_tactile_test, Label)
            else:
                data_tactile_train = np.append(data_tactile_train, Data)
                label_tactile_train = np.append(label_tactile_train, Label)
                
    data_tactile_train = data_tactile_train.reshape(-1,NUMBER,5,14)
    data_tactile_test = data_tactile_test.reshape(-1,NUMBER,5,14)
    label_tactile_train = label_tactile_train.reshape(-1,1)
    label_tactile_test = label_tactile_test.reshape(-1,1)
    print(data_tactile_train.shape,label_tactile_train.shape)
    
    sio.savemat('./data/tactile_train_original.mat', {'data':data_tactile_train ,'label':label_tactile_train})
    sio.savemat('./data/tactile_test_original.mat', {'data':data_tactile_test ,'label':label_tactile_test})
                

if __name__ == '__main__' :
#    load_odor()
    load_tactile()