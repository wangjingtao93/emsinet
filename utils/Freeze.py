##### System library #####
import os
import argparse
import time
from datetime import datetime
import tqdm
import socket
import numpy as np
from  PIL import Image
##### pytorch library #####
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
##### My own library #####
from data.OCT_Fluid import OCT_Fluid
from utils.loss import loss_builder
import utils.utils as u
from models.net_builder import net_builder
from utils.config import DefaultConfig
import utils
from visdom import Visdom
# from numpy import *

def Frozen_zone(model=None):
    # 编码器1    0——13
    model.module.conv1.conv1._modules['0'].weight.requires_grad = False
    model.module.conv1.conv1._modules['0'].bias.requires_grad = False
    model.module.conv1.conv1._modules['1'].weight.requires_grad = False
    model.module.conv1.conv1._modules['1'].bias.requires_grad = False
    model.module.conv1.conv1._modules['1'].running_mean.requires_grad = False
    model.module.conv1.conv1._modules['1'].running_var.requires_grad = False
    model.module.conv1.conv1._modules['1'].num_batches_tracked.requires_grad = False
    model.module.conv1.conv2._modules['0'].weight.requires_grad = False
    model.module.conv1.conv2._modules['0'].bias.requires_grad = False
    model.module.conv1.conv2._modules['1'].weight.requires_grad = False
    model.module.conv1.conv2._modules['1'].bias.requires_grad = False
    model.module.conv1.conv2._modules['1'].running_mean.requires_grad = False
    model.module.conv1.conv2._modules['1'].running_var.requires_grad = False
    model.module.conv1.conv2._modules['1'].num_batches_tracked.requires_grad = False

    #编码器2     14——27
    model.module.conv2.conv1._modules['0'].weight.requires_grad = False
    model.module.conv2.conv1._modules['0'].bias.requires_grad = False
    model.module.conv2.conv1._modules['1'].weight.requires_grad = False
    model.module.conv2.conv1._modules['1'].bias.requires_grad = False
    model.module.conv2.conv1._modules['1'].running_mean.requires_grad = False
    model.module.conv2.conv1._modules['1'].running_var.requires_grad = False
    model.module.conv2.conv1._modules['1'].num_batches_tracked.requires_grad = False
    model.module.conv2.conv2._modules['0'].weight.requires_grad = False
    model.module.conv2.conv2._modules['0'].bias.requires_grad = False
    model.module.conv2.conv2._modules['1'].weight.requires_grad = False
    model.module.conv2.conv2._modules['1'].bias.requires_grad = False
    model.module.conv2.conv2._modules['1'].running_mean.requires_grad = False
    model.module.conv2.conv2._modules['1'].running_var.requires_grad = False
    model.module.conv2.conv2._modules['1'].num_batches_tracked.requires_grad = False

    #编码器3     28——41
    model.module.conv3.conv1._modules['0'].weight.requires_grad = False
    model.module.conv3.conv1._modules['0'].bias.requires_grad = False
    model.module.conv3.conv1._modules['1'].weight.requires_grad = False
    model.module.conv3.conv1._modules['1'].bias.requires_grad = False
    model.module.conv3.conv1._modules['1'].running_mean.requires_grad = False
    model.module.conv3.conv1._modules['1'].running_var.requires_grad = False
    model.module.conv3.conv1._modules['1'].num_batches_tracked.requires_grad = False
    model.module.conv3.conv2._modules['0'].weight.requires_grad = False
    model.module.conv3.conv2._modules['0'].bias.requires_grad = False
    model.module.conv3.conv2._modules['1'].weight.requires_grad = False
    model.module.conv3.conv2._modules['1'].bias.requires_grad = False
    model.module.conv3.conv2._modules['1'].running_mean.requires_grad = False
    model.module.conv3.conv2._modules['1'].running_var.requires_grad = False
    model.module.conv3.conv2._modules['1'].num_batches_tracked.requires_grad = False

    #编码器4     42——55
    model.module.conv4.conv1._modules['0'].weight.requires_grad = False
    model.module.conv4.conv1._modules['0'].bias.requires_grad = False
    model.module.conv4.conv1._modules['1'].weight.requires_grad = False
    model.module.conv4.conv1._modules['1'].bias.requires_grad = False
    model.module.conv4.conv1._modules['1'].running_mean.requires_grad = False
    model.module.conv4.conv1._modules['1'].running_var.requires_grad = False
    model.module.conv4.conv1._modules['1'].num_batches_tracked.requires_grad = False
    model.module.conv4.conv2._modules['0'].weight.requires_grad = False
    model.module.conv4.conv2._modules['0'].bias.requires_grad = False
    model.module.conv4.conv2._modules['1'].weight.requires_grad = False
    model.module.conv4.conv2._modules['1'].bias.requires_grad = False
    model.module.conv4.conv2._modules['1'].running_mean.requires_grad = False
    model.module.conv4.conv2._modules['1'].running_var.requires_grad = False
    model.module.conv4.conv2._modules['1'].num_batches_tracked.requires_grad = False

    #center     56——69
    model.module.center.conv1._modules['0'].weight.requires_grad = False
    model.module.center.conv1._modules['0'].bias.requires_grad = False
    model.module.center.conv1._modules['1'].weight.requires_grad = False
    model.module.center.conv1._modules['1'].bias.requires_grad = False
    model.module.center.conv1._modules['1'].running_mean.requires_grad = False
    model.module.center.conv1._modules['1'].running_var.requires_grad = False
    model.module.center.conv1._modules['1'].num_batches_tracked.requires_grad = False
    model.module.center.conv2._modules['0'].weight.requires_grad = False
    model.module.center.conv2._modules['0'].bias.requires_grad = False
    model.module.center.conv2._modules['1'].weight.requires_grad = False
    model.module.center.conv2._modules['1'].bias.requires_grad = False
    model.module.center.conv2._modules['1'].running_mean.requires_grad = False
    model.module.center.conv2._modules['1'].running_var.requires_grad = False
    model.module.center.conv2._modules['1'].num_batches_tracked.requires_grad = False


    #解码器1     70——85
    model.module.up_concat4.conv.conv1._modules['0'].weight.requires_grad = False
    model.module.up_concat4.conv.conv1._modules['0'].bias.requires_grad = False
    model.module.up_concat4.conv.conv1._modules['1'].weight.requires_grad = False
    model.module.up_concat4.conv.conv1._modules['1'].bias.requires_grad = False
    model.module.up_concat4.conv.conv1._modules['1'].running_mean.requires_grad = False
    model.module.up_concat4.conv.conv1._modules['1'].running_var.requires_grad = False
    model.module.up_concat4.conv.conv1._modules['1'].num_batches_tracked.requires_grad = False
    model.module.up_concat4.conv.conv2._modules['0'].weight.requires_grad = False
    model.module.up_concat4.conv.conv2._modules['0'].bias.requires_grad = False
    model.module.up_concat4.conv.conv2._modules['1'].weight.requires_grad = False
    model.module.up_concat4.conv.conv2._modules['1'].bias.requires_grad = False
    model.module.up_concat4.conv.conv2._modules['1'].running_mean.requires_grad = False
    model.module.up_concat4.conv.conv2._modules['1'].running_var.requires_grad = False
    model.module.up_concat4.conv.conv2._modules['1'].num_batches_tracked.requires_grad = False
    model.module.up_concat4.up.weight.requires_grad = False
    model.module.up_concat4.up.bias.requires_grad = False

    #解码器2     86——101
    model.module.up_concat3.conv.conv1._modules['0'].weight.requires_grad = False
    model.module.up_concat3.conv.conv1._modules['0'].bias.requires_grad = False
    model.module.up_concat3.conv.conv1._modules['1'].weight.requires_grad = False
    model.module.up_concat3.conv.conv1._modules['1'].bias.requires_grad = False
    model.module.up_concat3.conv.conv1._modules['1'].running_mean.requires_grad = False
    model.module.up_concat3.conv.conv1._modules['1'].running_var.requires_grad = False
    model.module.up_concat3.conv.conv1._modules['1'].num_batches_tracked.requires_grad = False
    model.module.up_concat3.conv.conv2._modules['0'].weight.requires_grad = False
    model.module.up_concat3.conv.conv2._modules['0'].bias.requires_grad = False
    model.module.up_concat3.conv.conv2._modules['1'].weight.requires_grad = False
    model.module.up_concat3.conv.conv2._modules['1'].bias.requires_grad = False
    model.module.up_concat3.conv.conv2._modules['1'].running_mean.requires_grad = False
    model.module.up_concat3.conv.conv2._modules['1'].running_var.requires_grad = False
    model.module.up_concat3.conv.conv2._modules['1'].num_batches_tracked.requires_grad = False
    model.module.up_concat3.up.weight.requires_grad = False
    model.module.up_concat3.up.bias.requires_grad = False

    #解码器3     102——117
    model.module.up_concat2.conv.conv1._modules['0'].weight.requires_grad = False
    model.module.up_concat2.conv.conv1._modules['0'].bias.requires_grad = False
    model.module.up_concat2.conv.conv1._modules['1'].weight.requires_grad = False
    model.module.up_concat2.conv.conv1._modules['1'].bias.requires_grad = False
    model.module.up_concat2.conv.conv1._modules['1'].running_mean.requires_grad = False
    model.module.up_concat2.conv.conv1._modules['1'].running_var.requires_grad = False
    model.module.up_concat2.conv.conv1._modules['1'].num_batches_tracked.requires_grad = False
    model.module.up_concat2.conv.conv2._modules['0'].weight.requires_grad = False
    model.module.up_concat2.conv.conv2._modules['0'].bias.requires_grad = False
    model.module.up_concat2.conv.conv2._modules['1'].weight.requires_grad = False
    model.module.up_concat2.conv.conv2._modules['1'].bias.requires_grad = False
    model.module.up_concat2.conv.conv2._modules['1'].running_mean.requires_grad = False
    model.module.up_concat2.conv.conv2._modules['1'].running_var.requires_grad = False
    model.module.up_concat2.conv.conv2._modules['1'].num_batches_tracked.requires_grad = False
    model.module.up_concat2.up.weight.requires_grad = False
    model.module.up_concat2.up.bias.requires_grad = False

    #解码器3     118——133
    model.module.up_concat1.conv.conv1._modules['0'].weight.requires_grad = False
    model.module.up_concat1.conv.conv1._modules['0'].bias.requires_grad = False
    model.module.up_concat1.conv.conv1._modules['1'].weight.requires_grad = False
    model.module.up_concat1.conv.conv1._modules['1'].bias.requires_grad = False
    model.module.up_concat1.conv.conv1._modules['1'].running_mean.requires_grad = False
    model.module.up_concat1.conv.conv1._modules['1'].running_var.requires_grad = False
    model.module.up_concat1.conv.conv1._modules['1'].num_batches_tracked.requires_grad = False
    model.module.up_concat1.conv.conv2._modules['0'].weight.requires_grad = False
    model.module.up_concat1.conv.conv2._modules['0'].bias.requires_grad = False
    model.module.up_concat1.conv.conv2._modules['1'].weight.requires_grad = False
    model.module.up_concat1.conv.conv2._modules['1'].bias.requires_grad = False
    model.module.up_concat1.conv.conv2._modules['1'].running_mean.requires_grad = False
    model.module.up_concat1.conv.conv2._modules['1'].running_var.requires_grad = False
    model.module.up_concat1.conv.conv2._modules['1'].num_batches_tracked.requires_grad = False
    model.module.up_concat1.up.weight.requires_grad = False
    model.module.up_concat1.up.bias.requires_grad = False

    #分类器
    model.module.final_1.weight.requires_grad = False
    # model.module.final_1.bias.requires_grad = False

    return model