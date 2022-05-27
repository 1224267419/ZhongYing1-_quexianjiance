import itertools as it
import logging
import math
import os
import time
import warnings

import numpy as np
import pandas as pd
import torch
import torchvision.transforms as tf
from PIL import Image
from tqdm import tqdm

warnings.filterwarnings('ignore')
logging.basicConfig(format='%(message)s', level=logging.INFO)
LOGGER = logging.getLogger(__name__)

Loader = torch.utils.data.DataLoader
Adam = torch.optim.Adam
Listener = torch.optim.lr_scheduler.ReduceLROnPlateau


def autopad(k, p=None):  # kernel, padding
    # Pad to 'same'
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


def make_divisible(x, divisor=8):
    # Returns x evenly divisible by divisor
    return math.ceil(x / divisor) * divisor


def read_image(file):
    ''' 图像加载'''
    img = Image.open(file)
    size = len(img.mode), *img.size
    if size[0] == 1:
        transform = tf.Compose([
            tf.Resize([256, 256]),
            tf.ToTensor()
        ])
    else:
        transform = tf.Compose([
            tf.ToTensor()
        ])
    return transform(img)


def read_target(file):
    df = pd.read_csv(file)
    df.index = df['Image_name\t/(.bmp)']
    return df['categories']


def param_init(neural_net):
    ''' 网络参数初始化'''
    parameters = neural_net.state_dict()
    for key in parameters:
        if len(parameters[key].shape) >= 2:
            parameters[key] = torch.nn.init.kaiming_normal_(parameters[key], a=0, mode='fan_in',
                                                            nonlinearity='leaky_relu')


def dataset_split(data_set, batch_size: int, scale: int, classes=None, verbose=True):
    ''' 对原始数据集进行批处理、分割
        data_set: 原始数据集
        batch_size: 批处理时的批大小
        scale: 数据集的分割比例
             train_set: eval_set = (scale-1): scale
        classes: 分类数
        return: train_set, eval_set'''
    if classes:
        train_data, eval_data = [], []
        for category in data_set:
            data_len = len(category)
            eval_len = data_len // scale
            train_len = data_len - eval_len
            train_data_, eval_data_ = map(list, torch.utils.data.random_split(
                category, [train_len, eval_len], generator=torch.manual_seed(0)))
            train_data += train_data_
            eval_data += eval_data_
    else:
        data_len = len(data_set)

        eval_len = data_len // scale
        train_len = data_len - eval_len
        train_data, eval_data = torch.utils.data.random_split(data_set, [train_len, eval_len],
                                                              generator=torch.manual_seed(0))
    train_set = Loader(train_data, batch_size=batch_size, shuffle=True, drop_last=True)
    eval_set = Loader(eval_data, batch_size=batch_size, shuffle=True, drop_last=True)
    print(train_set)
    # print(train_data)
    if verbose:
        print(f'train_set: {len(train_set) * batch_size}\teval_set: {len(eval_set) * batch_size}')

    return train_set, eval_set


class Clock:
    ''' 时钟: 测试运行时间'''
    iterations = 1
    profix = 'Cost'

    def __init__(self, fun, *args, **kwargs):
        start = time.time()
        for _ in range(self.iterations):
            result = fun(*args, **kwargs)
            self.result = result
        self.cost = round((time.time() - start) * 1000, 0)
        print(f'{self.profix}: {self.cost:.0f} ms')

    def __str__(self):
        return str(self.result)

    __repr__ = __str__


class Chain:

    def __init__(self, loaders):
        self.loaders = loaders
        self.len = sum(map(len, loaders))

    def __iter__(self):
        return it.chain(*self.loaders)

    def __len__(self):
        return self.len


class Trainer:
    ''' 训练器
        net: 网络模型
        net_file: 网络模型保存路径 (.pth)
        loss_fun: 损失函数
        lr: 学习率'''

    def __init__(self, net, net_file: str, lr: float):
        # 设置系统参数
        self.net = net.cuda()
        self._net_file = net_file
        self._min_loss = np.inf
        # 载入网络参数
        if os.path.isfile(self._net_file):
            state_dict = torch.load(self._net_file)
            self.net.load_state_dict(state_dict)
        else:
            param_init(self.net)
        # 实例化优化器
        parameters = self.net.parameters()
        self._optimizer = Adam(parameters, lr=lr)
        self._lr_listener = Listener(self._optimizer,
                                     factor=0.2,
                                     patience=10,
                                     min_lr=1e-7,
                                     verbose=True)

    def train(self, train_set, prefix='train'):
        assert self._min_loss != np.inf, '请先运行 eval 函数'
        self.net.train()
        avg_loss = self._forward(train_set, train=True, prefix=prefix)

        self._lr_listener.step(avg_loss)
        return avg_loss

    @torch.no_grad()
    def eval(self, eval_set, prefix='eval ', save=True):
        self.net.eval()

        avg_loss = self._forward(eval_set, train=False, prefix=prefix)
        # 保存在测试集上表现最好的网络
        if save and avg_loss <= self._min_loss:
            self._min_loss = avg_loss
            torch.save(self.net.state_dict(), self._net_file)
            #TODO
            torch.save(self.net, '123.pth')
            torch.onnx.export(self.net,  torch.randn(10,3,256,256), '123.onnx', export_params=True)

        return avg_loss

    def loss(self, batch):
        ''' # 对 batch 进行解包
            image, target = batch
            # 将数据传到 GPU 上
            image, target = image.cuda(), target.cuda()
            # 调用神经网络
            logits = self.net(image)
            # 使用交叉熵损失
            loss = F.cross_entropy(logits, target)
            return loss'''
        pass

    def _forward(self, data_set, train: bool, prefix: str):
        ''' data_set: 数据集
            train: 训练 (bool)
            prefix: 进度条前缀
            return: 平均损失'''
        # 批信息
        batch_num = len(data_set)
        print(data_set)
        # batch_size = data_set.batch_size
        loss_sum = 0
        # 启动进度条
        pbar = tqdm(enumerate(data_set), total=batch_num)
        for idx, batch in pbar:
            # 计算损失值
            loss = self.loss(batch)
            loss_value = loss.item()
            loss_sum += loss_value
            # loss 反向传播梯度，并迭代
            if train:
                loss.backward()
                self._optimizer.step()
                self._optimizer.zero_grad()
            # 输出平均损失
            pbar.set_description(('%-10s' + '%-10.4g') %
                                 (prefix, loss_sum / (idx + 1)))
        avg_loss = loss_sum / batch_num
        return avg_loss