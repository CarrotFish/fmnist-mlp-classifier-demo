# 该文件定义模型层级及其行为(规定模型的输出层必须为SoftMax)
from nn import config

if config.CUDA: import cupy as np
else: import numpy as np
from nn import funcs

class Layer:
    '''模型层基类'''
    name: str = 'CustomLayer'
    tmp: np.ndarray
    trainable: bool = False
    def __init__(self):
        pass
    def predict(self, x:np.ndarray)->np.ndarray:
        '''计算输出'''
        self.tmp = x
        return x
    def back(self, grad:np.ndarray)->np.ndarray:
        '''计算反向传播(grad: 传下来的梯度)'''
        raise NotImplementedError()
    def train(self, lr:float, l2:float=0):
        '''进行训练 l2:L2正则化系数'''
        return


# Softmax输出层
class SoftMax(Layer):
    '''back直接传入label'''
    name = 'SoftMax'
    tmp_predict:np.ndarray
    def predict(self, x:np.ndarray)->np.ndarray:
        super().predict(x)
        self.tmp_predict = funcs.softmax(x)
        return self.tmp_predict
    def back(self, grad:np.ndarray)->np.ndarray:
        return funcs.dCrossSoftmax(self.tmp_predict, grad)

# 激活层
class Sigmoid(Layer):
    name = 'Sigmoid'
    tmp_predict:np.ndarray
    def predict(self, x:np.ndarray)->np.ndarray:
        super().predict(x)
        self.tmp_predict = funcs.sigmoid(x)
        return self.tmp_predict
    def back(self, grad:np.ndarray)->np.ndarray:
        return grad*funcs.dSigmoid(self.tmp_predict)
class ReLU(Layer):
    name = 'ReLU'
    def predict(self, x:np.ndarray)->np.ndarray:
        super().predict(x)
        return funcs.reLU(x)
    def back(self, grad:np.ndarray)->np.ndarray:
        return grad*funcs.dReLU(self.tmp)

# 参数层
class BaseLayer(Layer):
    '''必须传入维度信息'''
    name = 'Base'
    w:np.ndarray
    b:np.ndarray
    dW:np.ndarray
    db:np.ndarray
    trainable = True
    def __init__(self, in_dim:int=0, out_dim:int=0, init_scale:float=0.01):
        self.w = np.random.randn(in_dim, out_dim)*init_scale
        self.b = np.zeros(out_dim)
    def predict(self, x:np.ndarray)->np.ndarray:
        super().predict(x)
        return x@self.w+self.b
    def back(self, grad:np.ndarray)->np.ndarray:
        self.dW = funcs.dW(self.tmp, grad)
        self.db = funcs.db(grad)
        return grad@self.w.T
    def train(self, lr:float, l2:float=0):
        self.w -= lr*(self.dW+l2*self.w)
        self.b -= lr*self.db

# Dropout
class Dropout(Layer):
    mask:np.ndarray
    dropout:float
    def __init__(self, dropout:float=0.5):
        self.dropout = dropout
    def predict(self, x:np.ndarray)->np.ndarray:
        self.mask = (np.random.rand(*x.shape) >= self.dropout).astype(np.float32)
        return self.mask*x/(1-self.dropout)
    def back(self, grad:np.ndarray)->np.ndarray:
        return self.mask*grad/(1-self.dropout)



# 多层包装
class LayerStack(Layer):
    name = 'Stack'
    layers: list[Layer]
    trainable = True
    def __init__(self, layers:list[Layer]):
        self.layers = layers
    def predict(self, x:np.ndarray)->np.ndarray:
        super().predict(x)
        y = x
        for i in range(len(self.layers)):
            y = self.layers[i].predict(y)
        return y
    def back(self, grad:np.ndarray)->np.ndarray:
        g = grad
        for i in range(len(self.layers)-1, -1, -1):
            g = self.layers[i].back(g)
        return g
    def train(self, lr:float, l2:float=0):
        for i in range(len(self.layers)):
            # 是否进行训练
            if self.layers[i].trainable:
                self.layers[i].train(lr, l2)



# 通过layer name获取对应类
LAYER_NAMES = {}
LAYER_LIST = [SoftMax, LayerStack, Sigmoid, ReLU, BaseLayer]
for layer in LAYER_LIST:
    LAYER_NAMES[layer.name] = layer