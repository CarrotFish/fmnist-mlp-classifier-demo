from nn import training, funcs, loading, modeling
import numpy
from nn import config
if config.CUDA: import cupy as np
else: import numpy as np

numpy.random.seed(0)
np.random.seed(0)

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

# 加载模型
model = loading.load('best_0.9021.nnpt')
print('模型加载完毕')


# 正确率评估
def culculate_accuracy()->tuple[np.ndarray, float]:
    predict = model.predict(x_test)
    pred_class = predict.argmax(axis=1)
    true_class = y_test.argmax(axis=1)
    is_correct = (pred_class == true_class)
    return is_correct, is_correct.mean()

is_correct, acc = culculate_accuracy()
print('测试集正确率:', acc)