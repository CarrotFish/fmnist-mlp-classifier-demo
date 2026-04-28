# 定义所需要的运算函数(带batch)
from nn import config

if config.CUDA: import cupy as np
else: import numpy as np


limit = 1e-12


# 损失函数
def cross(predict:np.ndarray, label:np.ndarray)->float:
    '''交叉熵'''
    predict = np.clip(predict, limit, 1)
    return -np.mean(np.sum(label*np.log(predict), axis=1))


# Softmax
def softmax(x:np.ndarray)->np.ndarray:
    '''Softmax'''
    x = x-np.max(x, axis=1, keepdims=True)
    tmp = np.exp(x)
    return tmp/np.sum(tmp, axis=1, keepdims=True)


# 可用激活函数
def sigmoid(x:np.ndarray)->np.ndarray:
    '''Sigmoid'''
    return 1/(1+np.exp(-x))

def reLU(x:np.ndarray)->np.ndarray:
    '''ReLU'''
    return np.maximum(0, x)

# 梯度计算
def dCrossSoftmax(predict:np.ndarray, label:np.ndarray)->np.ndarray:
    '''交叉熵+softmax层的梯度计算'''
    # print(predict, label)
    return (predict-label)/predict.shape[0]
def dSigmoid(y:np.ndarray)->np.ndarray:
    '''Sigmoid激活层的梯度计算'''
    return y*(1-y)
def dReLU(z:np.ndarray)->np.ndarray:
    '''ReLU激活层的梯度计算'''
    return (z>0).astype(float)

# 线性层梯度
def dW(X:np.ndarray, dz:np.ndarray)->np.ndarray:
    '''W参数梯度计算'''
    return X.T @ dz / X.shape[0]
def db(dz:np.ndarray)->float:
    '''b参数梯度计算'''
    return np.mean(dz, axis=0)