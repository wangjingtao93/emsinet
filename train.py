##### System library #####
import os
import argparse
from pickle import TRUE
import time
from datetime import datetime
import tqdm
import socket
import numpy as np
from PIL import Image
from torchvision.transforms import Resize
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

from data.slit_loader import Slit_loader
from utils.loss import loss_builder
import utils.utils as u
from models.net_builder import net_builder
from utils.config import DefaultConfig
import utils
from numpy import *


def val(args, model, dataloader, epoch):
    print('\n')
    print('Start Validation!')
    with torch.no_grad():  # 适用于推断阶段，不需要反向传播
        model.eval()  # 测试模式，自动把BN和DropOut固定住，用训练好的值
        tbar = tqdm.tqdm(dataloader, desc='\r')  # 添加一个进度提示信息，只需要封装任意的迭代器 tqdm(iterator)，desc进度条前缀

        total_Dice = {}  # total_Dice中包含2个类别的dice列表
        total_Dice1 = []
        total_Dice2 = []
        total_Dice['dice1'] = total_Dice1
        total_Dice['dice2'] = total_Dice2

        total_Acc = []
        loss_val = []

        for i, (data, labels) in enumerate(tbar):  # i=300/batch_size=3
            # tbar.update()
            if torch.cuda.is_available() and args.use_gpu:
                data = data.cuda()
                label = labels.cuda().long()  # val模式下labels返回一个元祖，[0]为tensor标签索引值nhw，[1]为cube中包含slice张数int值15

            # get RGB predict image

            # _,predicts = model(data)
            predicts = model(data)
            predict = torch.argmax(predicts, dim=1)  # predicts预测结果nchw,寻找通道维度上的最大值predict变为nhw
            # predict 是nhw，但是label是n，1，h,w
            batch_size = predict.size()[0]  # 即n

            Dice1, Dice2, Acc = u.eval_multi_seg_2D_all(predict, label,
                                                        args.num_classes)  # 返回Dice列表, True_label列表中包含2个类别的像素数量，acc为标量值
            for i in range(len(Dice1)):
                total_Dice['dice1'].append(Dice1[i])
                total_Dice['dice2'].append(Dice2[i])
                total_Acc.append(Acc[i])

            # len2 = len(total_Dice[2]) if len(total_Dice[2]) != 0 else 1
            # len3 = len(total_Dice[3]) if len(total_Dice[3]) != 0 else 1
            # len4 = len(total_Dice[4]) if len(total_Dice[4]) != 0 else 1

            dice1 = sum(total_Dice['dice1']) / len(total_Dice['dice1'])
            dice2 = sum(total_Dice['dice2']) / len(total_Dice['dice2'])

            mean_dice = (dice1 + dice2) / 2.0
            acc = sum(total_Acc) / len(total_Acc)
            tbar.set_description(
                'Mean_D: %3f, Dice1: %.3f, Dice2: %.3f, Acc: %.3f' % (
                    mean_dice, dice1, dice2, acc))
        print('Mean_Dice:', mean_dice)
        print('Dice1:', dice1)
        print('Dice2:', dice2)
        print('Acc:', acc)
        with open(f'{current_time}+dice.txt', 'a+') as f:
            f.write('Mean_Dice:' + str(mean_dice) + ',')
            f.write('Dice1:' + str(dice1) + ',')
            f.write('Dice2:' + str(dice2) + ',')

            f.write('Acc:' + str(acc) + ',')
            f.write('\n')

        return mean_dice, dice1, dice2, acc


def train(args, model, optimizer, criterion, dataloader_train, dataloader_val, writer, k_fold,current_time):
    step = 0
    best_dice = 0.0

    #*****************
    deep_supervision = False
    #********************

    for epoch in range(args.num_epochs):
        # 循环100个epoch
        # lr = u.adjust_learning_rate(args, optimizer, epoch)  # 随着epoch更新学习率
        lr = args.lr
        model.train()  # 网络调整为训练状态，打开dropout和BN
        tq = tqdm.tqdm(total=len(dataloader_train) * args.batch_size)  # 进度条总迭代次数 len*batchsize
        tq.set_description('fold %d,epoch %d, lr %f' % (int(k_fold), epoch, lr))  # 设置修改进度条的前缀内容
        loss_record = []
        train_loss = 0.0

        for i, (data, label) in enumerate(dataloader_train):  # i= 14*128/4= 448
            # print(data.shape,label.shape)
            if torch.cuda.is_available() and args.use_gpu:
                data = data.cuda()
                label = label.cuda().long()  # 索引值格式为long
            optimizer.zero_grad()  # 根据backward()函数的计算，当网络参量反馈时梯度是被积累的而不是被替换掉；因此在每一个batch时设置一遍zero_grad将梯度清零
            if deep_supervision:
                aux_out, main_out = model(data)  # 前向传播
            else:
                main_out = model(data)

            loss_cel = criterion[0](main_out, label[:, 0, :, :])  # criterion[0]为CE LOSS()
            ##
            # print(main_out.dtype,aux_out[0].dtype)

            if deep_supervision:
                loss_aux = torch.tensor([0]).float().cuda()
                for i in range(len(aux_out)):
                    torch_resize = Resize([label.shape[2]//(2**(3-i)),label.shape[3]//(2**(3-i))])  # 定义Resize类对象
                    label_ = torch_resize(label)[:,0,:,:]
                    # print(label_.shape, aux_out[i].shape)
                    loss_aux += 0.5*(4-i)*criterion[0](aux_out[i],label_)
            # print(main_out.shape)
            # print(label.shape)
            loss_el = criterion[1](main_out, label[:, 0, :, :])  # dice loss

            if deep_supervision:
                loss = loss_cel + loss_el + loss_aux
            else:
                loss = loss_cel + loss_el
            loss.backward()  # 反向传播回传损失，计算梯度（loss为一个零维的标量）
            optimizer.step()  # optimizer基于反向梯度更新网络参数空间，因此当调用optimizer.step()的时候应当是loss.backward()后
            tq.update(args.batch_size)  # 进度条每次更新batch_size个进度
            train_loss += loss.item()  # 使用.item()可以从标量中获取Python数字
            tq.set_postfix(loss='%.6f' % (train_loss / (i + 1)))  # 设置进度条后缀，显示之前所有loss的平均
            step += 1
            # if step%10==0:
            # #writer.add_scalar('Train/loss_step', loss, step)
            loss_record.append(loss.item())  # loss_record包含所有loss的列表

            # time.sleep(0.5)
        tq.close()
        loss_train_mean = np.mean(loss_record)  # 此次训练epoch的448个Loss的平均值
        writer.add_scalar('Train/loss_epoch_{}'.format(int(k_fold)), float(loss_train_mean),
                          epoch)  # 每一折的每个epoch记录一次平均loss和epoch到日志

        print('loss for train : %f' % loss_train_mean)

        if epoch % args.validation_step == 0:  # 每训练validation_step个epoch进行一次验证，此时为1
            with open(f'{current_time}+dice.txt', 'a+') as f:
                f.write('fold:' + str(k_fold) + ',')
                f.write('EPOCH:' + str(epoch) + ',')
                f.write('loss(train):' + str(loss_train_mean) + ',')
            # mean_Dice, Dice1, Dice2, Dice3, Dice4, Dice5, Acc = val(args, model, dataloader_val, epoch)  # 验证集结果

            # 这里表示了一些指标的输出。
            mean_Dice, Dice1, Dice2, Acc = val(args, model, dataloader_val, epoch)

            # writer.add_scalar('Valid/Mean_val_{}'.format(int(k_fold)), mean_Dice, epoch)
            # writer.add_scalar('Valid/Dice1_val_{}'.format(int(k_fold)), Dice1, epoch)
            # writer.add_scalar('Valid/Dice2_val_{}'.format(int(k_fold)), Dice2, epoch)

            # # writer.add_scalar('Valid/Dice3_val_{}'.format(int(k_fold)), Dice3, epoch)
            # # writer.add_scalar('Valid/Dice4_val_{}'.format(int(k_fold)), Dice4, epoch)
            # # writer.add_scalar('Valid/Dice5_val_{}'.format(int(k_fold)), Dice5, epoch)
            # writer.add_scalar('Valid/Acc_val_{}'.format(int(k_fold)), Acc, epoch)

            is_best = mean_Dice > best_dice  # is_best为bool，以验证集平均dice为指标判断是否为best
            best_dice = max(best_dice, mean_Dice)  # 更新当前best dice
            # checkpoint_dir_root = args.save_model_path
            # checkpoint_dir=os.path.join(checkpoint_dir_root,str(k_fold))

            if is_best:
                print('===> Saving models...')
                state = {
                    'net': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'epoch': epoch,  # 将epoch一并保存
                    'best_dice': best_dice
                }
                if not os.path.exists(f'./checkpoint/{current_time}/'):
                    os.mkdir(f'./checkpoint/{current_time}/')
                torch.save(state, f'./checkpoint/{current_time}/' + str(k_fold) + '_' + str(epoch) + args.best_model)


def main(mode='train', args=None, writer=None, k_fold=1, channel=3, current_time=''):
    # create dataset and dataloader
    dataset_path = args.data +'/' + args.dataset
    dataset_train = Slit_loader(dataset_path, scale=(args.crop_width, args.crop_height), k_fold_test=k_fold,
                                mode='train',inputChannel=channel)
    dataloader_train = DataLoader(
        dataset_train,  # 加载的数据集(Dataset对象),dataloader是一个可迭代的对象，可以像使用迭代器一样使用
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,  # 使用多进程加载的进程数，0代表不使用多进程
        pin_memory=True,  # 是否将数据保存在pin memory区，pin memory中的数据转到GPU会快一些
        drop_last=True  # dataset中的数据个数可能不是batch_size的整数倍，drop_last为True会将多出来不足一个batch的数据丢弃
    )

    dataset_val = Slit_loader(dataset_path, scale=(args.crop_width, args.crop_height), k_fold_test=k_fold, mode='val',
                              inputChannel=channel)
    dataloader_val = DataLoader(
        dataset_val,
        # this has to be 1
        batch_size=1,  # 只选择1块gpu，batchsize=1
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False
    )


    # set gpu
    #***********************************************************
    os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda  # 指定gpu
    #***********************************************************


    # bulid model
    model = net_builder(args.net_work, args.pretrained_model_path, args.pretrained)
    print('Model have been loaded!,you chose the ' + args.net_work + '!')
    if torch.cuda.is_available() and args.use_gpu:
        model = model.cuda()  # torch.nn.DataParallel是支持并行GPU使用的模型包装器，并将模型放到cuda上
        # model = torch.nn.DataParallel(model).cuda() 照顾涛哥的网络而将这个去掉
    # load trained model for test
    if False:
        if args.trained_model_path :  # 测试时加载训练好的模型
            print("=> loading trained model '{}'".format(args.trained_model_path))
            checkpoint = torch.load(
                args.trained_model_path)  # torch.load：使用pickle unpickle工具将pickle的对象文件反序列化为内存，包括参数、优化器、epoch等
            model.load_state_dict(checkpoint)  # torch.nn.Module.load_state_dict:使用反序列化状态字典加载model’s参数字典
            print('Done!')

    # optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum,
    #                            weight_decay=args.weight_decay)  # 初始化优化器，构造一个optimizer对象
    # optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)  # 初始化优化器，构造一个optimizer对象
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # set the loss criterion
    criterion = loss_builder(args.loss_type, args.multitask)

    if mode == 'train':
        train(args, model, optimizer, criterion, dataloader_train, dataloader_val, writer, k_fold,current_time)
    # if mode == 'test':
    #     test(args, model, dataloader_test)
    # if mode == 'train_test':
    #     train(args, model, optimizer, criterion, dataloader_train, dataloader_val, writer, k_fold)
    #     test(args, model, dataloader_test)


if __name__ == '__main__':
    seed = 8888  # 在训练开始时，参数的初始化是随机的，为了让每次的结果一致，我们需要设置随机种子
    u.setup_seed(seed)
    args = DefaultConfig()  # 配置设置
    modes = args.mode
    palette = [0, 1, 2, 3, 4, 5]
    if modes == 'train':

        # comments = os.getcwd().split('')[-1]  # 方法用于返回当前工作目录即网络名称，仅train.py所在最后一个目录U-Net
        current_time = datetime.now().strftime('%b%d_%H-%M')  # 程序开始运行的时间
        str_para = input("请输入你想加入的后缀：")
        current_time = current_time + ' ' + str_para
        log_dir = os.path.join(args.log_dirs,
                               "Fiuld" + '_' + current_time + '_' + socket.gethostname())  # 获取当前计算节点名称cu01
        writer = SummaryWriter(logdir=log_dir)  # 创建writer object，log会被存入指定文件夹，writer.add_scalar保存标量值到log

        for i in range(0, 1):
            main(mode='train', args=args, writer=writer, k_fold=int(i + 1), channel=1, current_time=current_time)
