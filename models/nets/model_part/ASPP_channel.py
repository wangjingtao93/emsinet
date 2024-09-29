# -*- coding:utf-8 -*-
"""
作者：洪成健
日期：2022年07月18日
"""
import torch.nn as nn
import torch.nn.functional as F
import torch


## 第一个模块是关于实现ASPP的关键实现模块，跟普通的ASPP不同，它关注了每个通道的注意力。

class ConvBnRelu(nn.Module):
    def __init__(self, in_planes, out_planes, ksize, stride=1, pad=0, dilation=1,
                 groups=1, has_bn=True, norm_layer=nn.BatchNorm2d,
                 has_relu=True, inplace=True, has_bias=False):
        super(ConvBnRelu, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=ksize,
                              stride=stride, padding=pad,
                              dilation=dilation, groups=groups, bias=has_bias)
        self.has_bn = has_bn
        if self.has_bn:
            self.bn = nn.BatchNorm2d(out_planes)
        self.has_relu = has_relu
        if self.has_relu:
            self.relu = nn.ReLU(inplace=inplace)

    def forward(self, x):
        x = self.conv(x)
        if self.has_bn:
            x = self.bn(x)
        if self.has_relu:
            x = self.relu(x)

        return x


# 双注意混合多尺度模块
class DAFM(nn.Module):
    def __init__(self, in_channels):
        super(DAFM, self).__init__()
        self.conv3x3 = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, dilation=1, kernel_size=3,
                                 padding=1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.softmax = nn.Softmax(dim=1)
        self.bn = nn.ModuleList([nn.BatchNorm2d(in_channels), nn.BatchNorm2d(in_channels),
                                 nn.BatchNorm2d(in_channels), nn.BatchNorm2d(in_channels // 2)])
        self.lineal1 = nn.Linear(in_channels, in_channels // 2)
        self.lineal2 = nn.ModuleList([nn.Linear(in_channels // 2, in_channels),
                                      nn.Linear(in_channels // 2, in_channels),
                                      nn.Linear(in_channels // 2, in_channels)])
        self.softmax = nn.Softmax(dim=1)
        self.relu = nn.ReLU()
        self.conv1 = ConvBnRelu(in_channels*3, 1, 1, has_bn=False,has_relu=False)
        self.dconv = ConvBnRelu(in_channels*3, in_channels, 1, 1, has_bn=False)
        self.sigmoid = nn.Sigmoid()
        self.sigma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        x_size = x.size()

        branches_1 = self.conv3x3(x)
        branches_1 = self.bn[0](branches_1)
        branches_2 = F.conv2d(x, self.conv3x3.weight, padding=2, dilation=2)
        branches_2 = self.bn[1](branches_2)
        branches_3 = F.conv2d(x, self.conv3x3.weight, padding=4, dilation=4)
        branches_3 = self.bn[2](branches_3)
        # 1.配合上sk——net的多尺度网络
        feat_fus = branches_1 + branches_2 + branches_3
        feat_channel_mid = self.avg_pool(feat_fus) + self.max_pool(feat_fus)
        feat_channel_mid = self.relu(self.lineal1(feat_channel_mid.squeeze(-1).squeeze(-1)))
        # 中间三通道特征，运用softmax
        feat_channel = torch.cat(
            (self.lineal2[0](feat_channel_mid).unsqueeze(-1), self.lineal2[1](feat_channel_mid).unsqueeze(-1),
             self.lineal2[2](feat_channel_mid).unsqueeze(-1)), dim=2)
        feat_channel = self.softmax(feat_channel)
        branches_1_o = feat_channel[:, :, 0].unsqueeze(-1).unsqueeze(-1) * branches_1
        branches_2_o = feat_channel[:, :, 1].unsqueeze(-1).unsqueeze(-1) * branches_2
        branches_3_o = feat_channel[:, :, 2].unsqueeze(-1).unsqueeze(-1) * branches_3
        feat_o_1 = branches_2_o + branches_3_o + branches_1_o

        # 2. 条状池化的实现
        feat_con = torch.cat((branches_1, branches_2, branches_3), dim=1)
        feat_space = self.sigmoid(self.conv1(feat_con))
        feat_ = self.dconv(feat_con)
        feat_o_2 = feat_ * feat_space
        # 输出
        return torch.relu(x + self.sigma * (feat_o_1) + (1 - self.sigma)*feat_o_2)

        

if __name__ == '__main__':
    input_tensor = torch.randn((8, 256, 32, 32)).cuda()
    model = DAFM(256).cuda()

    model.eval()
    out = model(input_tensor)
    print(out.shape)
