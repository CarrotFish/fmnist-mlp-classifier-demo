# 结果可视化

from nn import training, funcs, loading, modeling
import numpy
from nn import config
if config.CUDA: import cupy as np
else: import numpy as np

numpy.random.seed(0)
np.random.seed(0)

# 数据加载
from fashion_mnist_master.utils import mnist_reader
_x_train, _y_train = mnist_reader.load_mnist('./fashion_mnist_master/data/fashion', kind='train')
_x_test, _y_test = mnist_reader.load_mnist('./fashion_mnist_master/data/fashion', kind='t10k')
# 数据预处理(flatten)
x_train = _x_train.reshape(_x_train.shape[0], -1).astype(np.float64)/255
x_test = _x_test.reshape(_x_test.shape[0], -1).astype(np.float64)/255
x_train = (x_train-0.5)/0.5
x_test = (x_test-0.5)/0.5
y_train = np.eye(10)[_y_train]
y_test = np.eye(10)[_y_test]
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
predict = model.predict(x_test)
pred_class = predict.argmax(axis=1)
true_class = y_test.argmax(axis=1)
is_correct = (pred_class == true_class)
acc = is_correct.mean()
if config.CUDA:
    is_correct = is_correct.get()
print('测试集正确率:', acc)

# 读取训练曲线
with open('log_main.txt', 'r', encoding='utf-8') as f:
    txt = f.read()
x, y1, y2, y3 = [], [], [], []
for line in txt.split('\n'):
    try:
        elems = line.split(' ')[1:]
        loss = float(elems[0].split('=')[1])
        loss_test = float(elems[1].split('=')[1])
        acc = float(elems[2].split('=')[1])
        x.append(len(x))
        y1.append(loss)
        y2.append(loss_test)
        y3.append(acc)
    except:
        pass

import matplotlib.pyplot as plt

fig, ax1 = plt.subplots()

ax1.plot(x[1:], y1[1:], label='Train Loss', color='blue')
ax1.plot(x[1:], y2[1:], label='Test Loss', color='orange')
ax1.set_xlabel('epoch')
ax1.set_ylabel('loss')

ax2 = ax1.twinx()
ax2.plot(x[1:], y3[1:], label='Accuracy', color='green')
ax2.set_ylabel('acc')

lines_1, labels_1 = ax1.get_legend_handles_labels()
lines_2, labels_2 = ax2.get_legend_handles_labels()
ax1.legend(lines_1 + lines_2, labels_1 + labels_2)

plt.title('Training Process')
plt.grid(True)
plt.show()


# 参数可视化
images = model.layers[0].w
if config.CUDA:
    images = images.get()
rows, cols = 16, 32
h, w = 28, 28
gap = 2

images = images.T
print(images.min(), images.max(), images.shape)

# 创建带空隙的大画布
grid_h = rows * h + (rows - 1) * gap
grid_w = cols * w + (cols - 1) * gap

grid = numpy.zeros((grid_h, grid_w))
images = (images - images.min()) / (images.max() - images.min() + 1e-8)

idx = 0
for i in range(rows):
    for j in range(cols):
        img = images[idx]
        
        top = i * (h + gap)
        left = j * (w + gap)
        grid[top:top+h, left:left+w] = img.reshape(28, 28)
        idx += 1

plt.figure(figsize=(20, 10))
plt.imshow(grid, cmap='gray', vmin=images.min(), vmax=images.max())
plt.axis('off')
plt.show()

# 获取前10个错误案例
wrongs = numpy.array(range(len(is_correct)))[is_correct == 0][:10]
print(wrongs)
# 拼接这10个错误案例并标注错误/正确分类
num = 10
wrong_imgs = [_x_test[wrongs[i]] for i in range(num)]
labels = [f'{pred_class[i]}/{true_class[i]}' for i in wrongs]
fig, axes = plt.subplots(1, num, figsize=(num * 2, 3))
for i in range(num):
    ax = axes[i]
    img = wrong_imgs[i].reshape(h, w)
    img = (img - img.min()) / (img.max() - img.min() + 1e-8)
    ax.imshow(img, cmap='gray', vmin=0, vmax=1)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel(labels[i], fontsize=10)
    for spine in ax.spines.values():
        spine.set_visible(False)

plt.subplots_adjust(wspace=0.3, hspace=0.2)
plt.show()