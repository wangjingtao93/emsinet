from audioop import mul
from re import S
from turtle import forward
import torch
import torch.nn as nn
import torch.functional as F


# ASPP的实现
class ASPP_module(nn.Module):
    def __init__(self, inplanes, planes, os):
        super(ASPP_module, self).__init__()
        if os == 16:
            dilations = [1, 6, 12, 18]
        elif os == 8:
            dilations = [1, 12, 24, 36]

        # 空洞率为1，6，12，18，padding为1、6、12、18的空洞卷积
        # 四组aspp的卷积块特征图的输出相同
        self.aspp1 = nn.Sequential(nn.Conv2d(inplanes, planes, kernel_size=1, stride=1, padding=0,
                                             dilation=dilations[0], bias=False),
                                   nn.BatchNorm2d(planes),
                                   nn.ReLU())

        self.aspp2 = nn.Sequential(nn.Conv2d(inplanes, planes, kernel_size=3, stride=1, padding=dilations[1],
                                             dilation=dilations[1], bias=False),
                                   nn.BatchNorm2d(planes),
                                   nn.ReLU())

        self.aspp3 = nn.Sequential(
            nn.Conv2d(inplanes, planes, kernel_size=3, stride=1, padding=dilations[2], dilation=dilations[2],
                      bias=False),
            nn.BatchNorm2d(planes),
            nn.ReLU())

        self.aspp4 = nn.Sequential(
            nn.Conv2d(inplanes, planes, kernel_size=3, stride=1, padding=dilations[3], dilation=dilations[3],
                      bias=False),
            nn.BatchNorm2d(planes),
            nn.ReLU())

        self.global_avg_pool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                             nn.Conv2d(512, 256, 1, stride=1, bias=False),
                                             nn.BatchNorm2d(256),
                                             nn.ReLU())

    def forward(self, x):
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x5 = self.global_avg_pool(x)

        x5 = F.interpolate(x5, size=x4.size()[2:], mode='bilinear', align_corners=True)
        x = torch.cat((x1, x2, x3, x4, x5), dim=1)
        return x


class Nonlocal(nn.Module):
    def __init__(self, channel):
        super(Nonlocal, self).__init__()
        # self.scale = scale
        self.inter_channel = channel // 2  # this notes the embedding dim
        self.conv_k = nn.Conv2d(in_channels=channel, out_channels=self.inter_channel, kernel_size=1, stride=1,
                                padding=0, bias=False)
        self.conv_q = nn.Conv2d(in_channels=channel, out_channels=self.inter_channel, kernel_size=1, stride=1,
                                padding=0, bias=False)
        self.conv_v = nn.Conv2d(in_channels=channel, out_channels=self.inter_channel, kernel_size=1, stride=1,
                                padding=0, bias=False)
        self.softmax = nn.Softmax(dim=1)
        self.conv_mask = nn.Conv2d(in_channels=self.inter_channel, out_channels=channel, kernel_size=1, stride=1,
                                   padding=0, bias=False)

    def forward(self, x):
        # (N,C,H,W)
        b, c, h, w = x.size()
        # (N,c/2,HW)
        x_q = self.conv_k(x).view(b, -1, h*w)
        # (N,HW,C/2)
        x_k = self.conv_q(x).view(b, -1, h*w).permute(0, 2, 1).contiguous()
        x_v = self.conv_v(x).view(b, -1, h*w).permute(0, 2, 1).contiguous()
        # (N,HW,HW)
        sim_matrix = self.softmax(torch.matmul(x_k, x_q))
        # (N,HW,C/2)
        sim_out = torch.matmul(sim_matrix, x_v)
        # (N,c/2,HW)
        sim_out = sim_out.permute(0, 2, 1).contiguous().view(b, self.inter_channel, h, w)

        # (N,C,H,W)
        mask = self.conv_mask(sim_out)
        out = mask + x
        return out


if __name__ == '__main__':
    model = Nonlocal(channel=16)
    print(model)

    input = torch.randn(1, 16, 64, 64)
    out = model(input)
    print(out.shape)
