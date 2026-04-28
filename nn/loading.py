# 负责模型参数以及信息的储存

from nn import modeling, config
import struct
if config.CUDA: import cupy as np
else: import numpy as np

class ModelLayerSaver:
    '''模型层数据的保存'''
    name: str
    data: list[float]

    def __init__(self, name: str):
        self.name = name
        self.data = []

class ModelInfo:
    name: str # 模型名称
    author: str # 模型作者
    desc: str # 模型描述
    layers: list[ModelLayerSaver] # 模型结构及参数的保存

    def __init__(self, name:str, author:str, desc:str, layers: list[ModelLayerSaver]):
        self.name = name
        self.author = author
        self.desc = desc
        self.layers = layers
    
    # 保存为二进制文件
    def write(self, filename: str):
        with open(filename, 'wb') as f:
            f.write(self.name.encode('utf-8')+b'\0')
            f.write(self.author.encode('utf-8')+b'\0')
            f.write(self.desc.encode('utf-8')+b'\0')
            for layer in self.layers:
                f.write(layer.name.encode('utf-8')+b'\0')
                if len(layer.data) > 0:
                    # 写入权重数据
                    for data in layer.data:
                        b = struct.pack('<d', data)
                        f.write(b)
    
    # 转换为Model
    def toModel(self)->modeling.LayerStack:
        stack_layers = []
        for layer in self.layers:
            model_layer: modeling.Layer = modeling.LAYER_NAMES[layer.name]()
            if layer.name == 'Base':
                # 解析参数
                height, width = map(int, layer.data[:2])
                w = np.array(layer.data[2:2+height*width]).reshape((height, width))
                b = np.array(layer.data[-width:])
                model_layer.w = w
                model_layer.b = b
            stack_layers.append(model_layer)
        return modeling.LayerStack(stack_layers)


    
def from_model(model: modeling.LayerStack, name: str, author: str, desc: str = '')->ModelInfo:
    '''将模型数据转换为存储结构'''
    result = ModelInfo(name, author, desc, [])
    def transformStack(stack: modeling.LayerStack):
        for layer in stack.layers:
            saver = ModelLayerSaver(layer.name)
            if layer.name == 'Base':
                # 需要存储数据的BaseLayer层
                tmp_layer: modeling.BaseLayer = layer
                # 存储w
                height, width = tmp_layer.w.shape[0], tmp_layer.w.shape[1]
                saver.data.append(height)
                saver.data.append(width)
                for y in range(height):
                    for x in range(width):
                        saver.data.append(tmp_layer.w[y][x])
                # 存储b
                for x in range(width):
                    saver.data.append(tmp_layer.b[x])
                result.layers.append(saver)
            elif layer.name == 'Stack':
                # 递归
                transformStack(layer)
            else:
                result.layers.append(saver)
    transformStack(model)
    return result

def from_file(filename: str)->ModelInfo:
    '''从文件中读取'''
    name = ''
    author = ''
    desc = ''
    layers = []
    with open(filename, 'rb') as f:
        data = bytearray()
        flag = 0
        while True:
            if flag < 3:
                # 在读取字符串信息
                b = f.read(1)
                if b == b'\0':
                    if flag == 0: name = data.decode('utf-8')
                    elif flag == 1: author = data.decode('utf-8')
                    else: desc = data.decode('utf-8')
                    data.clear()
                    flag += 1
                else:
                    data.extend(b)
            else:
                # 读取模型层信息
                b = f.read(1)
                if b == b'\0':
                    layername = data.decode('utf-8')
                    data.clear()
                    layer = ModelLayerSaver(layername)
                    if layername == 'Base':
                        # 需要读取w和b
                        height, width = struct.unpack('<2d', f.read(16))
                        layer.data.extend([height, width])
                        layer.data.extend(struct.unpack('<%dd'%(int((height+1)*width)), f.read(int(8*(height+1)*width))))
                        # print(layer.data)
                    layers.append(layer)
                elif b == b'':
                    break
                else:
                    data.extend(b)
    return ModelInfo(name, author, desc, layers)


def save(filename:str, model: modeling.LayerStack, name: str, author: str, desc: str = ''):
    from_model(model, name, author, desc).write(filename)
def load(filename:str)->modeling.LayerStack:
    info = from_file(filename)
    print(f'Model Loaded: {info.name} from {info.author}\n{info.desc}')
    return info.toModel()