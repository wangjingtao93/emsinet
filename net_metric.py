# -*- coding:utf-8 -*-
"""
作者：洪成健
日期：2022年06月13日
"""
from models.net_builder import net_builder
from utils.config import DefaultConfig
import torch
import tqdm
import os
import numpy as np
from data.slit_loader import Slit_loader
from torch.utils.data import DataLoader
import utils.utils
import utils.utils as u
import cv2 



def test(args, model, dataloader):
    print('\n')
    print('Start Validation!')
    with torch.no_grad():  # 适用于推断阶段，不需要反向传播
        model.eval()  # 测试模式，自动把BN和DropOut固定住，用训练好的值
        tbar = tqdm.tqdm(dataloader, desc='\r')  # 添加一个进度提示信息，只需要封装任意的迭代器 tqdm(iterator)，desc进度条前缀

        total_Dice = {}  # total_Dice中包含2个类别的dice列表
        total_Precision = {}
        total_recall = {}
        total_jaccard = {}
        total_Dice[1] = []
        total_Dice[2] = []
        total_Precision[2] = []
        total_Precision[1] = []
        total_jaccard[1] = []
        total_recall[1] = []
        total_recall[2] = []
        total_jaccard[2] = []
        total_Acc = []
        # k=0
        for i, (data, labels) in enumerate(tbar):  # i=300/batch_size=3
            # tbar.update()
            if torch.cuda.is_available() and args.use_gpu:
                data = data.cuda()
                label = labels.cuda().squeeze(1)
            if torch.sum(label) == 0:
                continue

            # get RGB predict image

            predicts= model(data)
            predict = torch.argmax(predicts, dim=1)  # predicts预测结果nchw,寻找通道维度上的最大值predict变为nhw
            # predict 是nhw，但是label是n，1，h,w

            # 下面这些程序不是很重要就是实现关键的图形的保存
            dir_name  = f'./result/img_{i}'
            if not os.path.exists(dir_name):
                os.mkdir(dir_name)
                cv2.imwrite(f'{dir_name}/img.png',np.uint8(data[0][0].cpu().numpy() * 255))  
                cv2.imwrite(f'{dir_name}/gt.png',np.uint8(label[0].cpu().numpy() * 127))
            cv2.imwrite(f'{dir_name}/pred_{args.net_work}.png',np.uint8(predict[0].cpu().numpy()*127))

            for j in range(1, args.num_classes):
                Dice_list, Precision_list, recall_list, jaccard_list = u.eval_seg(predict, label, forground=j)
                for m in range(len(Dice_list)):
                    total_Dice[j].append(Dice_list[m])
                    total_Precision[j].append(Precision_list[m])
                    total_recall[j].append(recall_list[m])
                    total_jaccard[j].append(jaccard_list[m])

        n = len(total_Dice[1])
        for k in range(n):
            dice_all = [sum(total_Dice[1]) / n, sum(total_Dice[2]) / n]
            precision_all = [sum(total_Precision[1]) / n, sum(total_Precision[2]) / n]
            recall_all = [sum(total_recall[1]) / n, sum(total_recall[2]) / n]
            jaccard_all = [sum(total_jaccard[1]) / n, sum(total_jaccard[2]) / n]

        print('Mean_Dice:', dice_all, sum(dice_all) / 2)
        print('precision:', precision_all, sum(precision_all) / 2)
        print('recall:', recall_all, sum(recall_all) / 2)
        print('jaccard:', jaccard_all, sum(jaccard_all) / 2)


def main(mode='test', args=None, inchannel= 3):
    # create dataset and dataloader
    dataset_path = args.data +'/'+ args.dataset

    dataset_test = Slit_loader(dataset_path, scale=(args.crop_width, args.crop_height), mode='test', inputChannel=inchannel)
    dataloader_test = DataLoader(
        dataset_test,
        # this has to be 1
        batch_size=1,  # 只选择1块gpu，batchsize=1
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False
    )

    # set gpu
    os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda # 指定gpu

    # bulid model
    model = net_builder(args.net_work, args.pretrained_model_path, args.pretrained)

    print('Model have been loaded!,you chose the ' + args.net_work + '!')
    if torch.cuda.is_available() and args.use_gpu:
        # model = torch.nn.DataParallel(model).cuda()  # torch.nn.DataParallel是支持并行GPU使用的模型包装器，并将模型放到cuda上
        model = model.cuda()
    # load trained model for test
    if args.trained_model_path and mode == 'test':  # 测试时加载训练好的模型
        print("=> loading trained model '{}'".format(args.trained_model_path))
        checkpoint = torch.load(
            args.trained_model_path)  # torch.load：使用pickle unpickle工具将pickle的对象文件反序列化为内存，包括参数、优化器、epoch等
        model.load_state_dict(checkpoint['net'])  # torch.nn.Module.load_state_dict:使用反序列化状态字典加载model’s参数字典
        print('Done!')

    if mode == 'test':
        test(args, model, dataloader_test)


if __name__ == '__main__':
    args = DefaultConfig()
    main('test', args, inchannel=1)

    # def test(args, model, dataloader, epoch):
