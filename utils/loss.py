import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np


def loss_builder(loss_type, multitask):
    if loss_type == 'mix_dice':
        weight_1 = torch.Tensor([1, 10, 10])
        weight_2 = torch.Tensor([1,15])
        # criterion_1 = nn.NLLLoss(weight=None,ignore_index=255)             #log_softmax的输出与Label对应的那个值拿出来，再去掉负号求均值(weight=None),即无权重的多类交叉熵平均导致过分关注背景
        criterion_1 = nn.CrossEntropyLoss(weight=weight_1, ignore_index=255)
        # criterion_2 = SoftDiceLoss()
        criterion_2 = DiceLoss()
        # criterion_2 = nn.CrossEntropyLoss()
        if multitask == True:
            criterion_3 = nn.CrossEntropyLoss(weight_2)
    elif loss_type == 'mix_eldice':
        weight_1 = torch.Tensor([1, 5, 10, 20])
        criterion_1 = nn.NLLLoss(weight=weight_1, ignore_index=255)
        criterion_2 = EL_DiceLoss()
        if multitask == True:
            criterion_3 = nn.BCEWithLogitsLoss()

    if loss_type in ['mix_dice', 'mix_eldice']:
        criterion_1.cuda()
        criterion_2.cuda()
        criterion = [criterion_1, criterion_2]
        if multitask == True:
            criterion_3.cuda()
            criterion.append(criterion_3)

    return criterion


class DiceLoss(nn.Module):
    '''
    如果某一类不存在，返回该类dice=1,dice_loss正常计算（平均值可能会偏高）
    如果所有类都不存在，返回dice_loss=0
    理论上smooth应该用不到，正常计算存在的某类dice时，union一定不为0
    '''

    def __init__(self, class_num=3, smooth=0.001):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        self.class_num = class_num

    def forward(self, input, target):
        input = F.softmax(input,dim=1)  # 网络输出log_softmax
        self.smooth = 0.001
        # torch.Tensor((3))生成一个size=3的随机张量，torch.Tensor([3])生成一个size=1值为3的张量
        Dice = Variable(torch.Tensor([0]).float()).cuda()
        for i in range(1, self.class_num):
            input_i = input[:, i, :, :]
            target_i = (target == i).float()
            intersect = (input_i * target_i).sum()
            union = torch.sum(input_i) + torch.sum(target_i)
            if target_i.sum() == 0:
                dice = torch.Tensor([1]).float().cuda()
            else:
                dice = (2 * intersect + self.smooth) / (union + self.smooth)
            Dice += dice

        dice_loss = 1 - Dice / (self.class_num - 1)

        return dice_loss


class SoftDiceLoss(nn.Module):

    def __init__(self, class_num=6, smooth=1e-5):
        super(SoftDiceLoss, self).__init__()
        self.smooth = smooth
        self.class_num = class_num

    def forward(self, input, target):
        # print(input.shape)
        input = torch.exp(input)
        self.smooth = 0.
        class_dice = []
        Dice = Variable(torch.Tensor([0]).float()).cuda()

        for i in range(1, self.class_num):
            pred = input[:, i:i + 1, :]  # [4,6,256,256]
            gt = target[:, i:i + 1, :]
            # print(pred.shape)
            # print(0)
            # print(gt.shape)
            # print("gt.type ========= {}".format(gt.dtype))
            intersection = (pred * gt.float()).sum()
            unionset = pred.sum() + gt.sum()
            dice = (2 * intersection + self.smooth) / (unionset + self.smooth)
            Dice += dice

        dice_loss = 1 - Dice / (self.class_num - 1)
        return dice_loss


class EL_DiceLoss(nn.Module):
    def __init__(self, class_num=6, smooth=1, gamma=0.5):
        super(EL_DiceLoss, self).__init__()
        self.smooth = smooth
        self.class_num = class_num
        self.gamma = gamma

    def forward(self, input, target):
        input = torch.exp(input)
        self.smooth = 0.
        Dice = Variable(torch.Tensor([0]).float()).cuda()
        for i in range(1, self.class_num):
            input_i = input[:, i, :, :]
            target_i = (target == i).float()
            intersect = (input_i * target_i).sum()
            union = torch.sum(input_i) + torch.sum(target_i)
            if target_i.sum() == 0:
                dice = Variable(torch.Tensor([1]).float()).cuda()
            else:
                dice = (2 * intersect + self.smooth) / (union + self.smooth)
            Dice += (-torch.log(dice)) ** self.gamma
        dice_loss = Dice / (self.class_num - 1)
        return dice_loss


palette = [0, 1, 2]


def mask_to_onehot(mask, palette):
    """
    Converts a segmentation mask (H, W, C) to (H, W, K) where the last dim is a one
    hot encoding vector, C is usually 1 or 3, and K is the number of class.
    """
    semantic_map = []
    for colour in palette:
        class_map = (mask == colour)
        semantic_map.append(class_map)
    semantic_map = torch.cat(semantic_map, dim=1)
    semantic_map = semantic_map.int()
    return semantic_map
