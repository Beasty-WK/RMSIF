import numpy as np
import torch
import os, glob
import random, csv
import time
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import torch.nn.functional as F
from torchvision import transforms, models
from PIL import Image
from torchvision import models
import torch.nn as nn
from RMSIF_NET import RMSIF_NET, BasicBlock
import warnings

warnings.filterwarnings("ignore")  # 忽略警告


torch.manual_seed(2023)   # 为了每次的实验结果一致

np.random.seed(2023)


# 数据地址
dataset_dir = r""
dataset_name = r""



# number of subprocesses to use for data loading

# 设置超参数
# 设置学习次数
total_epochs = 10
# 学习率
learning_rate = 0.01  #0.03
# number of subprocesses to use for data loading
num_workers = 8
# 采样批次
batch_size = 32
# 验证集比例


# test、train


dataset = np.load(dataset_dir)
dataset_order = np.load(dataset_name)






train_on_gpu = torch.cuda.is_available()

transform = transforms.Compose([
    # transforms.ToTensor(),
    transforms.ToTensor(),
    ])


class RNA_Mg_Dataset(Dataset):
    def __init__(self, dataset, order):
        self.list_data_path = dataset
        self.list_data_order = order
        self.transform = transform

    def __getitem__(self, item):
        data = np.load(self.list_data_path[item], allow_pickle=True)
        order = int(self.list_data_order[item])
        return self.transform(data), torch.LongTensor([order])

    def __len__(self):
        return len(self.list_data_path)




# 1.2.3确定验证集随机采样
indices = list(range(len(dataset)))
test_sampler = SubsetRandomSampler(indices)

# 测试集、训练集、验证

data_set = RNA_Mg_Dataset(dataset, dataset_order)


#1.2.4确定数据集

test_loader = torch.utils.data.DataLoader(data_set, batch_size=batch_size, sampler=test_sampler, shuffle=False, num_workers=num_workers)


# 初始化ResNet18
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ResNet18(BasicBlock=BasicBlock)
model.to(device)

optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)  # 随机梯度下降
valid_loss_min = np.Inf  # np.inf表示正无穷大，-np.inf表示负无穷大

predict_per = {}


if __name__ == '__main__':

    # 测试
    # 1、设置参数
    test_loss = 0.0  # 设置好test_loss认定为一个精度值
    class_correct = list(0. for i in range(2))  # [0.0, 0.0]
    class_total = list(0. for i in range(2))  # [0.0, 0.0]

    # 2、开始测试
    model.eval()  # model不反向传播
    state_dict = torch.load('model/RMSIF_NET.pt')
    model.load_state_dict(state_dict)
    for step, (data, order) in enumerate(test_loader):
        data = data.cuda()
        order = order.cuda().int()
        output = model(data)

        for i in range(output.shape[0]):  # 这里再按照output.shape[0]的值遍历
            predict_per[order[i].item()] = output[i].item() #float(format(output[i].item(), '.2g')) #np.round(output[i].item(), 2)



    #endregion


    np.save("loss/predict_per_G1G2TDA.npy", predict_per)




