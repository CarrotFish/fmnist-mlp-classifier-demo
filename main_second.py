from nn import loading, training

from nn import config
if config.CUDA: import cupy as np
else: import numpy as np

# 数据加载
from fashion_mnist_master.utils import mnist_reader
x_train, y_train = mnist_reader.load_mnist('./fashion_mnist_master/data/fashion', kind='train')
x_test, y_test = mnist_reader.load_mnist('./fashion_mnist_master/data/fashion', kind='t10k')
# 数据预处理(flatten)
x_train = x_train.reshape(x_train.shape[0], -1).astype(np.float64)/255
x_test = x_test.reshape(x_test.shape[0], -1).astype(np.float64)/255
x_train = (x_train-0.5)/0.5
x_test = (x_test-0.5)/0.5
y_train = np.eye(10)[y_train]
y_test = np.eye(10)[y_test]
if config.CUDA:
    x_train = np.asarray(x_train)
    x_test = np.asarray(x_test)
    y_train = np.asarray(y_train)
    y_test = np.asarray(y_test)
print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)

# 模型微调
model = loading.load('best_0.9021.nnpt')
train = training.DynamicLRTrainer(model, x_train, y_train, x_test, y_test, 32)
train.decrease_flag = True
train.lr_max = 0.0003
train.lr_min = 0.0000001
train.lr_delta = 0.98
train.warmup_epoch = 0
train.l2 = 1e-4

def culculate_accuracy():
    predict = model.predict(x_test)
    pred_class = predict.argmax(axis=1)
    true_class = y_test.argmax(axis=1)
    return (pred_class == true_class).mean()

max_acc = 0.9021
for epoch in range(10000):
    result = train.epoch()
    acc = culculate_accuracy()
    if acc > max_acc:
        # 最优
        max_acc = acc
        loading.save(f'best_{acc}.nnpt', model, 'Mnist Fashion 10分类器', 'CarrotFish1024', f'超参数如下：<layer1={512}> <layer2={256}> <acc={acc}>')
        print('[发现最优] 已保存')
    # print(epoch, result.loss, acc)
    print(f'[{epoch}/{10000}] acc={acc}')