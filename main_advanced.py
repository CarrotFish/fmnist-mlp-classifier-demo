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

def train(layer1:int, layer2:int, lr_max:float, lr_min:float, l2:float, lr_delta:float, epoch_num:int=30):
    # 模型定义
    model = modeling.LayerStack([
        modeling.BaseLayer(x_train.shape[1], layer1, 0.1),
        modeling.ReLU(),
        # modeling.Dropout(0.2),
        modeling.BaseLayer(layer1, layer2, 0.1),
        modeling.ReLU(),
        # modeling.Dropout(0.2),
        modeling.BaseLayer(layer2, 10, 0.1),
        modeling.SoftMax()
    ])

    # 模型基本信息
    model_name = 'Mnist Fashion 10分类器'
    author = 'CarrotFish1024'
    model_desc = f'超参数如下：<layer1={layer1}> <layer2={layer2}> <lr_max={lr_max}> <lr_min={lr_min}> <l2={l2}> <lr_delta={lr_delta}>'

    # 正确率评估
    def culculate_accuracy():
        predict = model.predict(x_test)
        pred_class = predict.argmax(axis=1)
        true_class = y_test.argmax(axis=1)
        return (pred_class == true_class).mean()

    # 训练循环
    trainer = training.DynamicLRTrainer(model, x_train, y_train, x_test, y_test, 16)
    trainer.warmup_epoch = 3
    trainer.lr_max = lr_max
    trainer.lr_min = lr_min
    trainer.l2 = l2
    trainer.lr_delta = lr_delta
    # trainer.lr_delta = 0
    max_acc = 0.89
    for epoch in range(epoch_num):
        result = trainer.epoch()
        # if (epoch+1)%1==0:
        acc = culculate_accuracy()
        if acc > max_acc:
            # 最优
            max_acc = acc
            loading.save(f'result/mnist_fashion_{layer1}_{layer2}_{lr_max}_{lr_min}_{l2}_{lr_delta}.nnpt', model, model_name, author, model_desc + f' <acc={acc}>')
            print('[发现最优] 已保存')
        # print(epoch, result.loss, acc)
        print(f'[{epoch}/{epoch_num}] acc={acc}')
        if acc > 0.8:
            trainer.start_decrease = True
    # 测试评估
    return max_acc


# 自动查找
with open('log.csv', 'w', encoding='utf-8') as log:
    log.write('layer1, layer2, lr_max, lr_min, l2, lr_delta, max_acc')
    for layer1 in [512, 256]:
        for layer2 in [256, 64]:
            for lr_max in [1, 0.5]:
                for lr_min in [0.01]:
                    for l2 in [1e-4, 1e-5]:
                        for lr_delta in [0.9, 0.95, 0.85]:
                            print()
                            print(f'[搜索超参] <layer1={layer1}> <layer2={layer2}> <lr_max={lr_max}> <lr_min={lr_min}> <l2={l2}> <lr_delta={lr_delta}>')
                            acc = train(layer1, layer2, lr_max, lr_min, l2, lr_delta)
                            print(f'[最终精度] {acc}')
                            log.write(f'{layer1}, {layer2}, {lr_max}, {lr_min}, {l2}, {lr_delta}, {acc}\n')
                            print()

