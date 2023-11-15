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
dataset_lable_dir = r""




# number of subprocesses to use for data loading

# 设置超参数
# 设置学习次数
total_epochs = 100
# 学习率
learning_rate = 0.02  #0.03
# number of subprocesses to use for data loading
num_workers = 8
# 采样批次
batch_size = 32
# 验证集比例


# test、train


dataset = np.load(dataset_dir)
dataset_lable = np.load(dataset_lable_dir)
# print(dataset[0])





train_on_gpu = torch.cuda.is_available()

transform = transforms.Compose([
    # transforms.ToTensor(),
    transforms.ToTensor(),
    ])


class RNA_Mg_Dataset(Dataset):
    def __init__(self, dataset, lables):
        self.list_data_path = dataset
        self.list_data_label = lables
        self.transform = transform

    def __getitem__(self, item):
        data = np.load(self.list_data_path[item], allow_pickle=True)
        # data = data[:, :, 2:4]
        label = self.list_data_label[item]
        return self.transform(data), torch.LongTensor([label]) 

    def __len__(self):
        return len(self.list_data_path)






# 1.2.3确定验证集随机采样

# 测试集、训练集、验证集。
indices = list(range(len(dataset)))
np.random.shuffle(indices)  # 打乱顺序

train_idx = indices[: int(len(dataset)*0.8)]
valid_idx = indices[int(len(dataset)*0.8): int(len(dataset)*0.9)]
test_idx = indices[int(len(dataset)*0.9):]



train_sampler = SubsetRandomSampler(train_idx)  # 确定采样的顺序，后面制作train_loader的时用这个列表得索引值取样本
valid_sampler = SubsetRandomSampler(valid_idx)
test_sampler = SubsetRandomSampler(test_idx)

data_set = RNA_Mg_Dataset(dataset, dataset_lable)


#1.2.4确定数据集
train_loader = torch.utils.data.DataLoader(data_set, batch_size=batch_size, sampler=train_sampler, num_workers=num_workers)
valid_loader = torch.utils.data.DataLoader(data_set, batch_size=batch_size, sampler=valid_sampler, num_workers=num_workers)
test_loader = torch.utils.data.DataLoader(data_set, batch_size=batch_size, sampler=test_sampler, num_workers=num_workers)

classes = ['disbinding_site', 'binding_site']

# 初始化RMSIF_NET
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = RMSIF_NET(BasicBlock=BasicBlock)
model.to(device)

optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)  # 随机梯度下降
valid_loss_min = np.Inf  # np.inf表示正无穷大，-np.inf表示负无穷大

save_train_loss = []
save_Valid_loss = []

if __name__ == '__main__':
    # 开始循环训练
    for epoch in range(total_epochs):
        if epoch > 30:
            learning_rate = 0.01

        if epoch > 90:
            learning_rate = 0.001

        if epoch > 200:
            learning_rate = 0.0001
        # 设置损失
        train_loss = 0.0
        valid_loss = 0.0

        ###################
        #      训练模块     #
        ###################

        model.train()
        for step, (data, target) in enumerate(train_loader):
            data, target = data.cuda(), target.cuda().float()  #.unsqueeze(1)

            optimizer.zero_grad()  # 将梯度清零
            output = model(data)  # 得到输出
            # 如果，模型中用的sigmoid，那输出必须都在0-1，否则就有问题
            loss = F.binary_cross_entropy(output,
                                          target)  
            loss.backward()  # 求方向传播得梯度
            optimizer.step()  # 反向传播
            train_loss += loss.item() * data.size(0) 

        ######################
        #       验证模块       #
        ######################

        model.eval()  # 下面的不做反向传播，即为验证部分
        for step, (data, target) in enumerate(valid_loader):
            data, target = data.cuda(), target.cuda().float()   #.unsqueeze(1)
            output = model(data)
            loss = F.binary_cross_entropy(output, target)  # 计算loss
            valid_loss += loss.item() * data.size(0)  # 计算验证损失

        train_loss = train_loss / len(train_loader.dataset)  # 计算平均损失
        valid_loss = valid_loss / len(valid_loader.dataset)

        save_train_loss.append(train_loss)
        save_Valid_loss.append(valid_loss)

        # 输出tain_loss和valid_loss
        print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
            epoch+1, train_loss, valid_loss))

        # 保存模型权重
        if valid_loss <= valid_loss_min:
            print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
                valid_loss_min,
                valid_loss))
            torch.save(model.state_dict(), 'model/RMSIF_NET.pt')
            valid_loss_min = valid_loss
    # 循环训练结束


    # 测试
    # 1、设置参数
    test_loss = 0.0  # 设置好test_loss认定为一个精度值
    class_correct = list(0. for i in range(2))  # [0.0, 0.0]
    class_total = list(0. for i in range(2))  # [0.0, 0.0]

    # 2、开始测试
    model.eval()  # model不反向传播
    state_dict = torch.load('model/RMSIF_NET.pt')
    model.load_state_dict(state_dict)
    for step, (data, target) in enumerate(test_loader):
        data, target = data.cuda(), target.cuda().float()
        output = model(data)
        # target1 = target.float().unsqueeze(1)
        loss = F.binary_cross_entropy(output, target)
        test_loss += loss.item() * data.size(0)

        pred = torch.tensor([[1] if num[0] >= 0.5 else [0] for num in output]).cuda()

        # 获得预测结果的正确与否
        correct_tensor = pred.eq(target.data.view_as(pred))
        # x.eq(y)判断x和y的值是否相等，作比较以后输出新的tensor 这里的view_as和view一样都是调整格式只不过view是将target调整成和pred一样的格式
        correct = np.squeeze(correct_tensor.numpy()) if not train_on_gpu else np.squeeze(
            correct_tensor.cpu().numpy())  # 改成np格式[ True  True False False False False]

        for i in range(output.shape[0]):  # 这里再按照output.shape[0]的值遍历
            label = target.data[i]  # 第一轮就是第一个pocket的标签是1 tensor([1.], device='cuda:0')
            label = label.cuda().int()
            class_correct[label] += correct[i].item()  
            class_total[label] += 1
    test_loss = test_loss / len(test_loader.dataset)  # 计算平均损失
    print('Test Loss: {:.6f}\n'.format(test_loss))
    # 测试结束

    # 获取精确度
    #通过得到的class_correct[i]和class_total[i]计算精度
    for i in range(2):
        if class_total[i] > 0:
            print('Test Accuracy of %5s: %2d%% (%2d/%2d)' % (
                classes[i], 100 * class_correct[i] / class_total[i],
                np.sum(class_correct[i]), np.sum(class_total[i])))  
        else:
            print('Test Accuracy of %5s: N/A (no training examples)' % (classes[i]))

    print('\nTest Accuracy (Overall): %2d%% (%2d/%2d)' % (
        100. * np.sum(class_correct) / np.sum(class_total),
        np.sum(class_correct), np.sum(class_total)))
    # endregion
#
#
    np.save("loss/save_train_loss.npy", save_train_loss)
    np.save("loss/save_Valid_loss.npy", save_Valid_loss)



