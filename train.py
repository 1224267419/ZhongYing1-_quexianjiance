from pathlib import Path

from basic import *

PATH = Path().resolve()
TRAIN_IMG = PATH / 'images/train'
TRAIN_LABEL = read_target(PATH / 'images/training.csv')

BATCH_SIZE = 10
LEARN_RATE = 5e-3
CLASSES = 8


def load_dataset(reload=False):
    data_file = 'dataset.dat'
    if reload:
        data_set_gray = [[] for _ in range(CLASSES)]
        data_set_common = [[] for _ in range(CLASSES)]
        for file in TRAIN_IMG.iterdir():
            # 按照文件名找到对应的分类
            index = int(file.stem[1:])
            label = torch.tensor(TRAIN_LABEL.loc[index], dtype=torch.long)
            # 读取图像
            img = read_image(file)
            info = [img, label]
            #print(img.shape)
            if img.shape[0] == 1:
                data_set_gray[label].append(info)
            else:
                data_set_common[label].append(info)
        torch.save([data_set_gray, data_set_common], data_file)
    else:
        data_set_gray, data_set_common = torch.load(data_file)
    return data_set_gray, data_set_common


data_set_gray, data_set_common = load_dataset(reload=True)
# 分割数据集
kwargs = dict(batch_size=BATCH_SIZE, scale=5, classes=CLASSES)
train_set_gray, eval_set_gray = dataset_split(data_set_gray, **kwargs)
train_set_common, eval_set_common = dataset_split(data_set_common, **kwargs)
# 拼接数据集
train_set = Chain([train_set_common, train_set_gray])
eval_set = Chain([eval_set_common, eval_set_gray])

net = Net(shrink=6, expansion=0.75)
clf = Classifier(net, net_file='model.pth', lr=LEARN_RATE, classes=CLASSES)

for epoch in range(1 << 64):
    clf.eval(eval_set, prefix=f'eval -{epoch}')
   # clf.train(train_set, prefix=f'train-{epoch}')