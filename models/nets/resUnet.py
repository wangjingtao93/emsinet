# -*- coding:utf-8 -*-
"""
作者：洪成健
日期：2022年05月25日
"""
# 实现一个将UNet的网络编码器变成ResNet的过程。

from torchsummary import summary
import torch
import torch.nn as nn
from models.nets.resnet import resnet34
from torch.nn import init





class ResUnet(nn.Module):
    def __init__(self, inC, outC, expand_feature=1):
        filters = [64, 64, 128, 256, 512]
        filters = [int(x / expand_feature) for x in filters]
        super().__init__()
        self.backbone = resnet34(pretrained=True)
        # self.backbone.conv1 = nn.Conv2d(3, 32, 3, 1, 1)
        # self.backbone.bn1 = nn.BatchNorm2d(32)
        self.up_3 = DecoderBlock(filters[4], filters[3])
        self.up_2 = DecoderBlock(filters[3], filters[2])
        self.up_1 = DecoderBlock(filters[2], filters[1])
        self.up_0 = DecoderBlock(filters[1], filters[0])
        self.finalDeconv = nn.ConvTranspose2d(filters[0], filters[0], kernel_size=2, stride=2)
        self.convlast = nn.Sequential(nn.Conv2d(filters[0], filters[0]//2, kernel_size=3,padding=1,bias=False),
                                      nn.BatchNorm2d(filters[0]//2),
                                      nn.ReLU(inplace=True),
                                      nn.Conv2d(filters[0]//2, outC,kernel_size=1,bias=False),
                                      )
        self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 4, 2, 1)
        self.finalrelu1 = nn.ReLU(inplace=True)
        self.finalconv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.finalrelu2 = nn.ReLU(inplace=True)
        self.finalconv3 = nn.Conv2d(32, outC, 3, padding=1)


    def forward(self, x):
        x0 = self.backbone.conv1(x)
        x0 = self.backbone.bn1(x0)
        x0 = self.backbone.relu(x0)
        x0_l = self.backbone.maxpool(x0)
        x1 = self.backbone.layer1(x0_l)
        x2 = self.backbone.layer2(x1)
        x3 = self.backbone.layer3(x2)
        x4 = self.backbone.layer4(x3)
        x3_r = self.up_3(x4) + x3
        x2_r = self.up_2(x3_r) + x2
        x1_r = self.up_1(x2_r) + x1
        x0_r = self.up_0(x1_r) + x0
        out = self.finaldeconv1(x0_r)
        out = self.finalrelu1(out)
        out = self.finalconv2(out)
        out = self.finalrelu2(out)
        out = self.finalconv3(out)
        return out,out





class DecoderBlock(nn.Module):
    def __init__(self, in_channels, n_filters):
        super(DecoderBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, in_channels // 4, 1)
        self.norm1 = nn.BatchNorm2d(in_channels // 4)
        self.relu1 = nn.ReLU(inplace=True)

        self.deconv2 = nn.ConvTranspose2d(in_channels // 4, in_channels // 4, 3, stride=2, padding=1, output_padding=1)
        self.norm2 = nn.BatchNorm2d(in_channels // 4)
        self.relu2 = nn.ReLU(inplace=True)

        self.conv3 = nn.Conv2d(in_channels // 4, n_filters, 1)
        self.norm3 = nn.BatchNorm2d(n_filters)
        self.relu3 = nn.ReLU(inplace=True)

        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         init_weights(m, init_type='kaiming')
        #     elif isinstance(m, nn.BatchNorm2d):
        #         init_weights(m, init_type='kaiming')


    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)
        x = self.deconv2(x)
        x = self.norm2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.norm3(x)
        x = self.relu3(x)
        return x


class Res_mul_Net(nn.Module):
    def __init__(self, inC, outC, expand_feature=1):
        filters = [64, 64, 128, 256, 512]
        filters = [int(x / expand_feature) for x in filters]
        super().__init__()
        self.backbone = resnet34(pretrained=True)
        # self.backbone.conv1 = nn.Conv2d(3, 32, 3, 1, 1)
        # self.backbone.bn1 = nn.BatchNorm2d(32)
        self.up_3 = DecoderBlock(filters[4], filters[3])
        self.up_2 = DecoderBlock(filters[3], filters[2])
        self.up_1 = DecoderBlock(filters[2], filters[1])
        self.up_0 = DecoderBlock(filters[1], filters[0])
        self.finalDeconv = nn.ConvTranspose2d(filters[0], filters[0], kernel_size=2, stride=2)
        self.convlast = nn.Sequential(nn.Conv2d(filters[0], filters[0] // 2, kernel_size=3, padding=1, bias=False),
                                      nn.BatchNorm2d(filters[0] // 2),
                                      nn.ReLU(inplace=True),
                                      nn.Conv2d(filters[0] // 2, outC, kernel_size=1, bias=False),
                                      )
        self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 4, 2, 1)
        self.finalrelu1 = nn.ReLU(inplace=True)
        self.finalconv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.finalrelu2 = nn.ReLU(inplace=True)
        self.finalconv3 = nn.Conv2d(32, outC, 3, padding=1)
        self.multi_scale = LocalMultiScale(filters[4])



    def forward(self, x):
        x0 = self.backbone.conv1(x)
        x0 = self.backbone.bn1(x0)
        x0 = self.backbone.relu(x0)
        x0_l = self.backbone.maxpool(x0)
        x1 = self.backbone.layer1(x0_l)
        x2 = self.backbone.layer2(x1)
        x3 = self.backbone.layer3(x2)
        x4 = self.backbone.layer4(x3)
        x4 = self.multi_scale(x2,x3, x4)
        x3_r = self.up_3(x4) + x3
        x2_r = self.up_2(x3_r) + x2
        x1_r = self.up_1(x2_r) + x1
        x0_r = self.up_0(x1_r) + x0
        out = self.finaldeconv1(x0_r)
        out = self.finalrelu1(out)
        out = self.finalconv2(out)
        out = self.finalrelu2(out)
        out = self.finalconv3(out)
        return out, out


class LocalMultiScale(nn.Module):   # 3-4层与第五层的融合。
    def __init__(self,inChannels):
        super(LocalMultiScale, self).__init__()
        self.down1 = nn.Sequential(nn.Conv2d(inChannels//4, inChannels//2,kernel_size=3,stride=2,padding=1),
                                   nn.BatchNorm2d(inChannels//2),
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(inChannels // 2, inChannels, kernel_size=3, stride=2, padding=1),
                                   nn.BatchNorm2d(inChannels),
                                   )

        self.down2 = nn.Sequential(nn.Conv2d(inChannels//2,inChannels,kernel_size=3,stride=2,padding=1),
                                   nn.BatchNorm2d(inChannels),
                                   )
        self.convFirst = nn.Sequential(nn.Conv2d(inChannels, inChannels, kernel_size=1),
                                      nn.BatchNorm2d(inChannels),
                                      nn.ReLU(inplace=True),
                                      nn.Conv2d(inChannels, inChannels, kernel_size=1),
                                      nn.BatchNorm2d(inChannels),
                                      nn.ReLU(inplace=True))
        self.conv5X5 = nn.Sequential(nn.Conv2d(inChannels,5*5,kernel_size=3,padding=1),
                                     nn.BatchNorm2d(5*5))
        self.conv3X3 = nn.Sequential(nn.Conv2d(inChannels,3*3,kernel_size=3,padding=1),
                                     nn.BatchNorm2d(3*3))
        self.conv1X1 = nn.Sequential(nn.Conv2d(inChannels,1,kernel_size=3,padding=1),
                                     nn.BatchNorm2d(1))
        self.Softmax = nn.Softmax(dim=1)
        self.relu = nn.ReLU(inplace=True)
        self.alpha = nn.Parameter(torch.zeros(1))
        self.convlast = nn.Sequential(nn.Conv2d(inChannels*3, inChannels, kernel_size=1),
                                      nn.BatchNorm2d(inChannels),
                                      nn.ReLU(inplace=True),
                                      nn.Conv2d(inChannels,inChannels,kernel_size=1),
                                      nn.BatchNorm2d(inChannels),
                                      nn.ReLU(inplace=True))
        self.globalPool = nn.AvgPool2d(1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init_weights(m, init_type='kaiming')
            elif isinstance(m, nn.BatchNorm2d):
                init_weights(m, init_type='kaiming')

    def forward(self,x3, x4, x5):

        # x5_pre 是一个上下文的信息。是一个指导信息。
        x3_down = self.down1(x3)
        x4_down = self.down2(x4)
        x3_4 = self.relu(x3_down + x4_down)
        x5_pre = self.convFirst(x3_4)

        x5_5X5 = nn.Unfold(kernel_size=5,dilation=1,padding=2,stride=1)(x5)
        x5_5x5 = x5_5X5.view(x5_5X5.shape[0],x5.shape[1],5*5,x5.shape[2],x5.shape[3])
        s5x5 = self.conv5X5(x5_pre)
        s5x5 = self.Softmax(s5x5)
        x5_1 = s5x5.unsqueeze(1) * x5_5x5
        x5_1 = torch.sum(x5_1,dim=2)

        x5_3X3 = nn.Unfold(kernel_size=3, dilation=1, padding=1)(x5)
        x5_3x3 = x5_3X3.view(x5_3X3.shape[0], x5.shape[1], 3* 3, x5.shape[2], x5.shape[3])
        s3x3 = self.conv3X3(x5_pre)
        s3x3 = self.Softmax(s3x3)
        x5_2 = s3x3.unsqueeze(1) * x5_3x3
        x5_2 = torch.sum(x5_2, dim=2)



        s1x1 = self.conv1X1(x5_pre)
        x5_3 = s1x1 *x5
        x5_com = torch.cat((x5_1,x5_2,x5_3),dim=1)
        x5_com = self.convlast(x5_com)
        return x5_com




# class LocalMultiScale(nn.Module):   # 与第五层的融合。
#     def __init__(self,inChannels):
#         super(LocalMultiScale, self).__init__()
#         self.down1 = nn.Sequential(nn.Conv2d(inChannels//4,inChannels//2,kernel_size=2,stride=2),
#                                    nn.BatchNorm2d(inChannels//2),
#                                    nn.ReLU(inplace=True))
#         self.down2 = nn.Sequential(nn.Conv2d(inChannels,inChannels,kernel_size=2,stride=2),
#                                    nn.BatchNorm2d(inChannels),
#                                    nn.ReLU(inplace=True))
#         self.conv5X5 = nn.Sequential(nn.Conv2d(inChannels,5*5,kernel_size=3,padding=1),
#                                      nn.BatchNorm2d(5*5))
#         self.conv3X3 = nn.Sequential(nn.Conv2d(inChannels,3*3,kernel_size=3,padding=1),
#                                      nn.BatchNorm2d(3*3))
#         self.conv1X1 = nn.Sequential(nn.Conv2d(inChannels,1,kernel_size=3,padding=1),
#                                      nn.BatchNorm2d(1))
#         self.Softmax = nn.Softmax(dim=1)
#         self.relu = nn.ReLU(inplace=True)
#         self.convlast = nn.Sequential(nn.Conv2d(inChannels,inChannels,kernel_size=1),
#                                       nn.BatchNorm2d(inChannels))
#         self.alpha = nn.Parameter(torch.zeros(1))
#
#
#     def forward(self, x5):
#         x5_5X5 = nn.Unfold(kernel_size=5,dilation=1,padding=2,stride=1)(x5)
#         x5_5x5 = x5_5X5.view(x5_5X5.shape[0],x5.shape[1],5*5,x5.shape[2],x5.shape[3])
#         s5x5 = self.conv5X5(x5)
#         s5x5 = self.Softmax(s5x5)
#         x5_1 = s5x5.unsqueeze(1) * x5_5x5
#         x5_1 = torch.sum(x5_1,dim=2)
#
#         x5_3X3 = nn.Unfold(kernel_size=3, dilation=1, padding=1)(x5)
#         x5_3x3 = x5_3X3.view(x5_3X3.shape[0], x5.shape[1], 3* 3, x5.shape[2], x5.shape[3])
#         s3x3 = self.conv3X3(x5)
#         s3x3 = self.Softmax(s3x3)
#         x5_2 = s3x3.unsqueeze(1) * x5_3x3
#         x5_2 = torch.sum(x5_2, dim=2)
#         s1x1 = self.conv1X1(x5)
#         x5_3 = s1x1 *x5
#         x5_com = self.convlast(x5_1+x5_2+x5_3)
#         # print(self.alpha)
#         return self.relu((1-self.alpha)*x5 + self.alpha*x5_com)


def init_weights(net, init_type='normal'):
    # print('initialization method [%s]' % init_type)
    if init_type == 'kaiming':
        net.apply(weights_init_kaiming)
    else:
        raise NotImplementedError('initialization method [%s] is not implemented' % init_type)


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('BatchNorm') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


if __name__ == '__main__':
    input = torch.rand(8, 3, 512, 512).cuda()
    count = 0
    model = ResUnet(3, 1).cuda()
    # print(summary(model, (3, 512, 512)))
    out_ = model(input)
    print(out_.shape)
