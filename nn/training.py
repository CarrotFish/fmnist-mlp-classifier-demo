# 用于自动化训练
from nn import config

if config.CUDA: import cupy as np
else: import numpy as np
from nn import modeling
from nn import funcs
import math
import numpy

class EpochResult:
    '''存储每个Epoch训练情况'''
    loss: float
    loss_test: float

class Trainer:
    '''对训练器做一个完整的包装'''
    model: modeling.LayerStack
    x_train: np.ndarray
    y_train: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray
    batch_size: int
    indices: np.ndarray

    def __init__(self, model:modeling.LayerStack, x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, y_test: np.ndarray, batch_size: int):
        self.model = model
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.batch_size = batch_size

    def randSet(self):
        '''随机生成训练集'''
        self.indices = numpy.random.permutation(self.x_train.shape[0])

    def epoch(self)->EpochResult:
        '''单次训练'''
        result = EpochResult()
        # 生成训练集
        xt, yt = self.randSet()
        # 先进行预测
        predict = self.model.predict(xt)
        cross_loss = funcs.cross(predict, yt)
        result.loss = cross_loss
        # 反向传播
        self.model.back(yt)
        # 进行训练
        self.model.train(1)
        return result

class DynamicLRTrainer(Trainer):
    '''采取带预热的指数衰减方法'''
    l2: float = 1e-3
    epoch_count: int = 0
    warmup_epoch: int = 200
    lr_max: float = 1e-2
    lr_min: float = 1e-4
    lr_delta: float = 0.98 # lr衰减率
    lr: float = lr_max

    start_decrease: bool = False

    decrease_flag: bool = False
    def epoch(self)->EpochResult:
        result = EpochResult()
        result.loss = 0
        # 随机训练集顺序
        self.randSet()
        for i in range(int(self.x_train.shape[0]/self.batch_size)):
            # 取得训练集
            xt, yt = self.x_train[self.indices[i*self.batch_size:(i+1)*self.batch_size]], self.y_train[self.indices[i*self.batch_size:(i+1)*self.batch_size]]
            # 先进行预测
            predict = self.model.predict(xt)
            cross_loss = funcs.cross(predict, yt)
            result.loss += cross_loss
            # 反向传播
            self.model.back(yt)
            # 进行训练
            if self.epoch_count < self.warmup_epoch:
                # 进行warmup
                self.lr = self.lr_min + self.epoch_count*(self.lr_max-self.lr_min)/self.warmup_epoch
            elif self.start_decrease:
                self.lr = max(self.lr_min, self.lr*self.lr_delta)
            else:
                self.lr = self.lr_max
            self.model.train(self.lr, self.l2)
        self.epoch_count += 1
        # loss计算
        predict_test = self.model.predict(self.x_test)
        result.loss_test = funcs.cross(predict_test, self.y_test)
        result.loss /= self.x_train.shape[0]/self.batch_size
        return result