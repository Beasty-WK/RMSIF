from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os




class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=[1,1], padding=1) -> None:
        super(BasicBlock, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_channels,out_channels,kernel_size=3,stride=stride[0],padding=padding,bias=False),  # 尺寸不变
            nn.BatchNorm2d(out_channels),  # 数据归一化
            nn.ReLU(inplace=True),  # 原地替换 节省内存开销
            nn.Conv2d(out_channels, out_channels,kernel_size=3,stride=stride[1],padding=padding,bias=False),  # 尺寸不变
            nn.BatchNorm2d(out_channels)
        )

        self.shortcut = nn.Sequential()
        if stride[0] != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride[0], bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = self.layer(x)
        out += self.shortcut(x)  
        out = F.relu(out)
        return out

class RMSIF_NET(nn.Module):
    def __init__(self, BasicBlock, num_classes=1) -> None: 
        super(RMSIF_NET, self).__init__()
        self.in_channels = 64
 
        self.conv1 = nn.Sequential(
            nn.Conv2d(5,64,kernel_size=7,stride=2,padding=3,bias=False),  
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)  
        )
        # conv2_x
        self.conv2 = self._make_layer(BasicBlock,64,[[1,1],[1,1]])  

        # conv3_x
        self.conv3 = self._make_layer(BasicBlock,128,[[2,1],[1,1]])  

        # conv4_x
        self.conv4 = self._make_layer(BasicBlock,256,[[2,1],[1,1]])  

        # conv5_x
        self.conv5 = self._make_layer(BasicBlock,512,[[2,1],[1,1]])

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)


    def _make_layer(self, block, out_channels, strides):
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels
        return nn.Sequential(*layers)
    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.conv5(out)

        out = self.avgpool(out)
        out = out.reshape(x.shape[0], -1)
        out = torch.sigmoid(self.fc(out)) 


