import scipy.io as sio
import numpy as np
import random

def fusion():
    
    train = sio.loadmat('./data/odor_train.mat')
    test = sio.loadmat('./data/odor_test.mat')
    odor_train = train['data']
    odor_test = test['data']
    
    train = sio.loadmat('./data/tactile_train.mat')
    test = sio.loadmat('./data/tactile_test.mat')
    tactile_train = train['data']
    tactile_test = test['data']
    
    sio.savemat('./data/fusion_train.mat', 
                {'odor':odor_train, 'tactile':tactile_train, 'label':train['label']})
    sio.savemat('./data/fusion_test.mat', 
                {'odor':odor_test, 'tactile':tactile_test, 'label':test['label']})
    
    train = sio.loadmat('./data/odor_feature_train.mat')
    test = sio.loadmat('./data/odor_feature_test.mat')
    odor_feature_train = train['data']
    odor_feature_test = test['data']
    
    train = sio.loadmat('./data/tactile_feature_train.mat')
    test = sio.loadmat('./data/tactile_feature_test.mat')
    tactile_feature_train = train['data']
    tactile_feature_test = test['data']
    
    sio.savemat('./data/fusion_feature_train.mat', 
                {'odor':odor_feature_train, 'tactile':tactile_feature_train, 'label':train['label']})
    sio.savemat('./data/fusion_feature_test.mat', 
                {'odor':odor_feature_test, 'tactile':tactile_feature_test, 'label':test['label']})
    
def ab2C(a,b,lengtha,lengthb,d):
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

def fusion_convolve():
    train = sio.loadmat('./data/odor_feature_train.mat')
    test = sio.loadmat('./data/odor_feature_test.mat')
    odor_feature_train = train['data']
    odor_feature_test = test['data']
    
    train = sio.loadmat('./data/tactile_feature_train.mat')
    test = sio.loadmat('./data/tactile_feature_test.mat')
    tactile_feature_train = train['data']
    tactile_feature_test = test['data']
    
    length_train = odor_feature_train.shape[0]
    Convolve_train = np.array([])
    for i in range(length_train):
        convolve_train = ab2C(odor_feature_train[i], tactile_feature_train[i], 64, 96, 128)
        Convolve_train = np.append(Convolve_train, convolve_train)
    Convolve_train = Convolve_train.reshape(-1,128)
    
    length_test = odor_feature_test.shape[0]
    Convolve_test = np.array([])
    for i in range(length_test):
        convolve_test = ab2C(odor_feature_test[i], tactile_feature_test[i], 64, 96, 128)
        Convolve_test = np.append(Convolve_test, convolve_test)
    Convolve_test = Convolve_test.reshape(-1,128)
    
    print(Convolve_train.shape, Convolve_test.shape)
    
    sio.savemat('./data/fusion_convolve_train.mat', 
                {'data':Convolve_train, 'label':train['label']})
    sio.savemat('./data/fusion_convolve_test.mat', 
                {'data':Convolve_test, 'label':test['label']})
    
if __name__ == '__main__':
    
    fusion()
    
#    fusion_convolve()