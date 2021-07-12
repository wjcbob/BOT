import scipy.io as sio
import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt
from scipy import interpolate
import scipy

def visual():
    
    train = sio.loadmat('./data/tactile_train_original.mat')
    test = sio.loadmat('./data/tactile_test_original.mat')
    tactile_data_train = train['data']
    tactile_data_test = test['data']
    
    print(tactile_data_train.shape)
    
    for i in range(tactile_data_train.shape[3]):
        print(np.max(tactile_data_train[:,:,:,i]),np.min(tactile_data_train[:,:,:,i]))
        
    for i in range(tactile_data_test.shape[3]):
        print(np.max(tactile_data_test[:,:,:,i]),np.min(tactile_data_test[:,:,:,i]))
        
    tactile_data_train = (tactile_data_train + 3) / 7
    tactile_data_test = (tactile_data_test + 3) / 7
    
    sio.savemat('./data/tactile_train.mat', 
                {'data':tactile_data_train, 'label':train['label']})
    sio.savemat('./data/tactile_test.mat', 
                {'data':tactile_data_test, 'label':test['label']})

def transform():
    train = sio.loadmat('./data/tactile_train_original.mat')
    test = sio.loadmat('./data/tactile_test_original.mat')
    tactile_data_train = train['data']
    tactile_data_test = test['data']
    
    print(tactile_data_train.shape)
    '''
    for i in range(tactile_data_train.shape[3]):
        maxx = np.max(tactile_data_train[:,:,:,i])
        minn = np.min(tactile_data_train[:,:,:,i])
        print(maxx, minn)
        tactile_data_train[:,:,:,i] = (tactile_data_train[:,:,:,i] - minn) / (maxx - minn)
        
    for i in range(tactile_data_test.shape[3]):
        maxx = np.max(tactile_data_test[:,:,:,i])
        minn = np.min(tactile_data_test[:,:,:,i])
        print(maxx, minn)
        tactile_data_test[:,:,:,i] = (tactile_data_test[:,:,:,i] - minn) / (maxx - minn)
    '''
    tactile_data_train = (tactile_data_train + 3) / 7
    tactile_data_test = (tactile_data_test + 3) / 7
    
    i_max = tactile_data_train.shape[0]
    j_max = tactile_data_train.shape[1]
    Index = np.array(range(i_max))
    random.shuffle(Index)
    length = int (len(Index) / 8)
    for i in range(0,length):
        tactile_data_train[Index[i],:,:,4:] = tactile_data_train[Index[i],:,:,:10]
        tactile_data_train[Index[i],:,:,:4] = 0
    for i in range(length,2*length):
        tactile_data_train[Index[i],:,:,:10] = tactile_data_train[Index[i],:,:,4:]
        tactile_data_train[Index[i],:,:,10:] = 0
    for i in range(2*length,3*length):
        tactile_data_train[Index[i],:,:3,:] = tactile_data_train[Index[i],:,2:,:]
        tactile_data_train[Index[i],:,3:,:] = 0
    for i in range(3*length,4*length):
        tactile_data_train[Index[i],:,2:,:] = tactile_data_train[Index[i],:,:3,:]
        tactile_data_train[Index[i],:,:2,:] = 0
    for i in range(4*length,5*length):
        tactile_data_train[Index[i]] = tactile_data_train[Index[i],:,::-1]
    for i in range(5*length,6*length):
        tactile_data_train[Index[i]] = tactile_data_train[Index[i],:,:,::-1]
    for i in range(6*length,7*length):
        tactile_data_train[Index[i]] = tactile_data_train[Index[i],:,::-1,::-1]
    '''
    i_max = tactile_data_test.shape[0]
    j_max = tactile_data_test.shape[1]
    Index = np.array(range(i_max))
    random.shuffle(Index)
    length = int (len(Index) / 8)
    for i in range(0,length):
        tactile_data_test[Index[i],:,:,4:] = tactile_data_test[Index[i],:,:,:10]
        tactile_data_test[Index[i],:,:,:4] = 0
    for i in range(length,2*length):
        tactile_data_test[Index[i],:,:,:10] = tactile_data_test[Index[i],:,:,4:]
        tactile_data_test[Index[i],:,:,10:] = 0
    for i in range(2*length,3*length):
        tactile_data_test[Index[i],:,:3,:] = tactile_data_test[Index[i],:,2:,:]
        tactile_data_test[Index[i],:,3:,:] = 0
    for i in range(3*length,4*length):
        tactile_data_test[Index[i],:,2:,:] = tactile_data_test[Index[i],:,:3,:]
        tactile_data_test[Index[i],:,:2,:] = 0
    for i in range(4*length,5*length):
        tactile_data_test[Index[i],:] = tactile_data_test[Index[i],:,::-1]
    for i in range(5*length,6*length):
        tactile_data_test[Index[i],:] = tactile_data_test[Index[i],:,:,::-1]
    for i in range(6*length,7*length):
        tactile_data_test[Index[i],:] = tactile_data_test[Index[i],:,::-1,::-1]
    '''
    sio.savemat('./data/tactile_train.mat', 
                {'data':tactile_data_train, 'label':train['label']})
    sio.savemat('./data/tactile_test.mat', 
                {'data':tactile_data_test, 'label':test['label']})

def load_odor():
    NAME = ['橘子','毛巾','人体胳膊','石头','头','腿','小鼠','衣服','直筒水杯','易拉罐','纸盒']
    LENGTH = 5000
    data_odor_train, label_odor_train = np.array([]), np.array([])
    data_odor_test, label_odor_test = np.array([]), np.array([])
    for name in range(len(NAME)):
        path = './data/odor/'+NAME[name]
        for num in range(3,4):
            execl_name = path+'/'+str(num+1)+'.xlsx'
            print(execl_name)
            data = pd.read_excel(execl_name, header=None)
            data = np.copy(data)
            data = data[1:,1:]
            
            ####差值
            DATA = np.zeros((LENGTH, 6))
            Label = np.zeros((LENGTH,1)) + name
            x = np.linspace(0,data.shape[0]-1,data.shape[0])
            xnew = np.linspace(0,data.shape[0]-1,LENGTH)
            for i in range(data.shape[1]):
                f = interpolate.interp1d(x,data[:,i],kind='cubic')
                DATA[:,i] = f(xnew)
            
            Data = np.copy(DATA)
            
            if (num == 3):
                data_odor_test = np.append(data_odor_test, Data)
                label_odor_test = np.append(label_odor_test, Label)
            else:
                data_odor_train = np.append(data_odor_train, Data)
                label_odor_train = np.append(label_odor_train, Label)
            
    data_odor_train = data_odor_train.reshape(-1,6)
    data_odor_test = data_odor_test.reshape(-1,6)
    label_odor_train = label_odor_train.reshape(-1,1)
    label_odor_test = label_odor_test.reshape(-1,1)
    print(data_odor_test.shape,label_odor_test.shape)
    
#    for i in range(11):
#        for j in range(3):
#            plt.ylim(0,1)
#            plt.plot(range(LENGTH),data_odor_train[LENGTH*(i*3+j):LENGTH*(i*3+j+1)])
#            plt.show()
    
    sio.savemat('./data/odor.mat', {'data':data_odor_test ,'label':label_odor_test})
            
def load_tactile():
    NAME = ['橘子','毛巾','人体胳膊','石头','头','腿','小鼠','衣服','直筒水杯','易拉罐','纸盒']
    FINGER = ['拇指', '食指', '中指', '无名指', '小指']
    LENGTH = 5000
    data_tactile_train, label_tactile_train = np.array([]), np.array([])
    data_tactile_test, label_tactile_test = np.array([]), np.array([])
    for name in range(len(NAME)):
        path = './data/tactile/'+NAME[name]
        for num in range(3,4):
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
                
                #########下采样
                list_data = pd.date_range('1/1/2000', periods=data.shape[0], freq='T') 
                data_1 = pd.DataFrame(data)
                data_1.index = list_data
                data_obj = data_1.resample('%dT' % int(data.shape[0] / LENGTH), label='right') 
                data_new = data_obj.asfreq()[0:]
                print(data_new.shape, data.shape)

                DATA[:,finger,:] = data_new[:LENGTH]
               
            Data = np.copy(DATA)
            
            if (num == 3):
                data_tactile_test = np.append(data_tactile_test, Data)
                label_tactile_test = np.append(label_tactile_test, Label)
            else:
                data_tactile_train = np.append(data_tactile_train, Data)
                label_tactile_train = np.append(label_tactile_train, Label)
                
    data_tactile_train = data_tactile_train.reshape(-1,5,14)
    data_tactile_test = data_tactile_test.reshape(-1,5,14)
    label_tactile_train = label_tactile_train.reshape(-1,1)
    label_tactile_test = label_tactile_test.reshape(-1,1)
    print(data_tactile_test.shape,label_tactile_test.shape)
    
    sio.savemat('./data/tactile.mat', {'data':data_tactile_test ,'label':label_tactile_test})


def gui():
    
    odor_test = sio.loadmat('./data/odor.mat')
    tactile_test = sio.loadmat('./data/tactile.mat')
    odor = odor_test['data']
    tactile = tactile_test['data']
    label = odor_test['label']
    LENGTH = 5000
    '''
    for i in range(int(odor.shape[0] / LENGTH)):
        for j in range(odor.shape[1]):
            maxx = np.max(odor[i*LENGTH:(i+1)*LENGTH,j])
            plt.plot(range(LENGTH), odor[i*LENGTH:(i+1)*LENGTH,j])
        plt.show()
    '''
    tactile_sum = np.zeros((tactile.shape[0],tactile.shape[1]))
    for i in range(tactile.shape[0]):
        for j in range(tactile.shape[1]):
            tactile_sum[i,j] = tactile[i,j].sum()
    for i in range(int(tactile.shape[0] / LENGTH)):
        for j in range(tactile.shape[1]):
            plt.plot(range(LENGTH), tactile_sum[i*LENGTH:(i+1)*LENGTH,j])
        plt.show()
    '''
    for i in range(int(tactile.shape[0] / LENGTH)):
        for j in range(tactile.shape[1]):
            for k in range(tactile.shape[2]):
                maxx = np.max(tactile[i*LENGTH:(i+1)*LENGTH,j,k])
                minn = np.min(tactile[i*LENGTH:(i+1)*LENGTH,j,k])
                plt.plot(range(LENGTH), tactile[i*LENGTH:(i+1)*LENGTH,j,k])
        plt.show()
    '''
    for i in range(int(tactile.shape[0] / LENGTH)):
        ODOR = odor[i*LENGTH:(i+1)*LENGTH]
        TACTILE = tactile[i*LENGTH:(i+1)*LENGTH]
        LABEL = label[i*LENGTH:(i+1)*LENGTH]
        sio.savemat('./data/test/test%d.mat'%i, 
                    {'odor':ODOR, 'tactile':TACTILE, 'label':LABEL})
   
def ab2c(a,b,lengtha,lengthb,d):
    index_a, index_b = np.array(range(d)), np.array(range(d)) 
    random.shuffle(index_a)
    v_a = index_a[:lengtha]
    s_a = (np.random.randint(1,3,lengtha) - 1.5) * 2
    A = np.zeros((d,))
    for i in range(lengtha):
        A[v_a[i]] = A[v_a[i]] + a[i] * s_a[i]
    
    random.shuffle(index_b)
    v_b = index_b[:lengthb]
    s_b = (np.random.randint(1,3,lengthb) - 1.5) * 2
    B = np.zeros((d,))
    for i in range(lengthb):
        B[v_b[i]] = B[v_b[i]] + b[i] * s_b[i]
    
    C = np.convolve(A, B, 'same')
    C = np.sqrt(np.abs(C))
    l2_norm = np.sqrt(np.multiply(C, C).sum())
    C = C / l2_norm
    return C

if __name__ == '__main__':
    
#    visual()
    
    transform()
 
#    load_odor()
#    load_tactile()
#    gui()
    '''
    a = np.random.randint(0,100,16)
    b = np.random.randint(0,100,32)
    print(a)
    print(b)
    
    A = np.fft.ifft(np.fft.fft(a))
    print(len(A))
    print(A)
    B = np.fft.irfft(np.multiply(np.fft.rfft(a),np.fft.rfft(a)))
    print(B)
    print(len(B))
    print(np.convolve(a, a, 'same'))
    print(np.multiply(np.fft.fft(a),np.fft.fft(a)))
    
    c = np.convolve(a, a, 'same')
    c = np.sqrt(c)
    l2_norm = np.sqrt(np.multiply(c, c).sum())
    c = c / l2_norm
    print(a)
    print(c)
    e = ab2c(a,b,len(a),len(b),64)
    print(e)
    '''