# -*- coding:utf-8 -*-
"""
作者：洪成健
日期：2022年07月27日
"""
# 本网络是基于Unet的双分支网络

from operator import mul
import os
from typing import MutableMapping
from sklearn import multiclass
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.utils.layers import *
from models.utils.init_weights import init_weights
from models.nets.model_part.ASPP_channel import DAFM

class Double_UNet(nn.Module):

    def __init__(self, in_channels=1, n_classes=4, feature_scale=4, is_deconv=True, is_batchnorm=True):
        super().__init__()
        self.is_deconv = is_deconv
        self.in_channels = in_channels
        self.is_batchnorm = is_batchnorm
        self.feature_scale = feature_scale

        filters = [64, 128, 256, 512, 1024]
        filters = [int(x / self.feature_scale) for x in filters]

        # downsampling
        self.conv1 = unetConv2(self.in_channels, filters[0], self.is_batchnorm)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)

        self.conv2 = unetConv2(filters[0], filters[1], self.is_batchnorm)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)

        self.conv3 = unetConv2(filters[1], filters[2], self.is_batchnorm)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2)

        self.conv4 = unetConv2(filters[2], filters[3], self.is_batchnorm)
        self.maxpool4 = nn.MaxPool2d(kernel_size=2)

        self.center = unetConv2(filters[3], filters[4], self.is_batchnorm)

        # upsampling1
        self.up_concat4 = unetUp(filters[4], filters[3], self.is_deconv,n_concat=3)
        self.up_concat3 = unetUp(filters[3], filters[2], self.is_deconv,n_concat=3)
        self.up_concat2 = unetUp(filters[2], filters[1], self.is_deconv,n_concat=3)
        self.up_concat1 = unetUp(filters[1], filters[0], self.is_deconv,n_concat=3)
          # 插入一个模块
        self.muiti_p = DAFM(512)
        # upsample2 这里记录的是边缘的损失
        self.add_concat4 = unetUp_edge(filters[4], filters[3], self.is_deconv)
        self.add_concat3 = unetUp_edge(filters[3], filters[2], self.is_deconv)
        self.add_concat2 = unetUp_edge(filters[2], filters[1], self.is_deconv)
        self.add_concat1 = unetUp_edge(filters[1], filters[0], self.is_deconv)

        # final conv (without any concat)
        self.final_1 = nn.Conv2d(filters[0], n_classes, 1)
        # final_2是给边缘loss卷积的
        self.final_2 = nn.Conv2d(filters[0], 2, 1)
        # self.final_3 = nn.Conv2d(filters[0], n_classes, 1)

        # initialise weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init_weights(m, init_type='kaiming')
            elif isinstance(m, nn.BatchNorm2d):
                init_weights(m, init_type='kaiming')

    def forward(self, inputs):
        conv1 = self.conv1(inputs)  # 16*512*1024
        maxpool1 = self.maxpool1(conv1)  # 16*256*512

        conv2 = self.conv2(maxpool1)  # 32*256*512
        maxpool2 = self.maxpool2(conv2)  # 32*128*256

        conv3 = self.conv3(maxpool2)  # 64*128*256
        maxpool3 = self.maxpool3(conv3)  # 64*64*128

        conv4 = self.conv4(maxpool3)  # 128*64*128
        maxpool4 = self.maxpool4(conv4)  # 128*32*64

        center = self.center(maxpool4)  # 256*32*64

        
        
        # 插入多尺度的模块
        mupti = self.muiti_p(center)

        edge4 = self.add_concat4(mupti,conv4)
        edge3 = self.add_concat3(edge4,conv3)
        edge2 = self.add_concat2(edge3,conv2)
        edge1 = self.add_concat1(edge2,conv1)
        up4 = self.up_concat4(mupti, conv4,edge4)  # 128*64*128
        up3 = self.up_concat3(up4, conv3,edge3)  # 64*128*256
        up2 = self.up_concat2(up3, conv2,edge2)  # 32*256*512
        up1 = self.up_concat1(up2, conv1,edge1)  # 16*512*1024
        # edge4 = self.add_concat4(mupti, conv4)
        # up4 = self.up_concat4(mupti, conv4, edge4)  # 128*64*128
        # edge3 = self.add_concat3(up4, conv3)
        # up3 = self.up_concat3(up4, conv3, edge3)  # 64*128*256
        # edge2 = self.add_concat2(up3, conv2)
        # up2 = self.up_concat2(up3, conv2, edge2)  # 32*256*512
        # edge1 = self.add_concat1(up2, conv1)
        # up1 = self.up_concat1(up2, conv1, edge1)  # 16*512*1024


        final_1 = self.final_1(up1)
        final_2 = self.final_2(edge1)
        return final_1,final_2




# ------------------------------------------------------------------------------------------------------

if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    net = Double_UNet(in_channels=1, n_classes=4, is_deconv=True).cuda()
    print(net)
    x = torch.rand((4, 1, 256, 128)).cuda()
    forward = net.forward(x)
    print(forward)
    print(type(forward))

#    net = resnet34_unet(in_channel=1,out_channel=4,pretrain=False).cuda()
#    torchsummary.summary(net, (1, 512, 512))
