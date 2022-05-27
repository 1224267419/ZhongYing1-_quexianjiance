from .model import *
import torch.nn.functional as F


class Pr_Counter:
    ''' 模型精度计算器
        batch_num: 批数量
        prefix: 进度条前缀'''
    classes = None
    string_mode = None

    def __init__(self, batch_num, prefix):
        self._loss_sum = 0
        self._prefix = prefix
        self.cross_tab = pd.crosstab(*[np.arange(0, self.classes)] * 2) * 0
        self.pbar = tqdm(range(batch_num))

    def update(self, idx, logits, target, loss):
        ''' idx: logits 索引'''
        # 取 logits 的最大值索引做为 result
        pred = logits.argmax(1)
        # 叠加列联表
        self.cross_tab += pd.crosstab(
            index=np.concatenate([target.cpu().data.numpy(), np.arange(0, self.classes)]),
            columns=np.concatenate([pred.cpu().data.numpy(), np.arange(0, self.classes)])
        ) - np.eye(self.classes)
        # 计算得出 avg_loss
        loss = loss.item()
        self._loss_sum += loss
        avg_loss = self._loss_sum / (idx + 1)
        # 输出 train / eval 数据
        self.pbar.set_description(self.string_mode %
                                  (self._prefix, avg_loss,
                                   *(self.cross_tab[i][i] / self.cross_tab[i].sum()
                                     for i in range(self.classes))))
        self.pbar.update()
        return avg_loss


class Classifier(Trainer):
    ''' 分类器
        net: 网络模型
        net_file: 网络模型保存路径 (.pt)
        adam: 使用 Adam 优化器
        bar_len: 进度条长度'''

    def __init__(self, net, net_file: str, lr: float, classes: int):
        super(Classifier, self).__init__(net, net_file, lr)
        # 设置模型精度计算器的类属性
        Pr_Counter.classes = classes
        string_mode = '%-10s' + '%-15s' + '%-6s' * classes
        # 输出数值标题
        LOGGER.info(string_mode % ('', 'Avg Loss',
                                   *(f'Pr {i}' for i in range(classes))))
        Pr_Counter.string_mode = '%-10s' + '%-15.8f' + '%-6.2f' * classes
        self._pos_weight = torch.ones([8]).cuda()  # torch.tensor([1.2, 1.5, 1, 1.2, 1, 1.2, 1.5, 1]).cuda()

    def loss(self, batch):
        # 对 batch 进行解包
        image, target = batch
        # 将数据传到 GPU 上
        image, target = image.cuda(), target.cuda()
        # 调用神经网络
        logits = self.net(image)
        # 使用交叉熵损失
        loss = F.cross_entropy(logits, target, weight=self._pos_weight)
        return loss, logits, target

    def _forward(self, data_set, train: bool, prefix: str):
        # 初始化精度计算器
        counter = Pr_Counter(len(data_set), prefix)
        for idx, batch in enumerate(data_set):
            loss, logits, target = self.loss(batch)
            if train:
                # loss 反向传播梯度，并迭代
                loss.backward()
                self._optimizer.step()
                self._optimizer.zero_grad()
            # 更新 acc 计算器
            avg_loss = counter.update(idx, logits, target, loss)
        counter.pbar.close()
        return avg_loss