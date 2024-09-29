# -*- coding: utf-8 -*-
class DefaultConfig(object):
    # 数据集与日志
    data = './dataset'  # 数据存放的根目录
    dataset = 'BV1000crop'  # 数据库名字(需修改成自己的数据名字)
    log_dirs = './log'  # 存放tensorboard log的文件夹()
    save_model_path = '.\\checkpoints_all\\unet'

    # 网络训练设置
    net_work = 'unet'
    best_model = f'Fiuld_{net_work}.pkl'
    mode = 'train'
    num_epochs = 200
    batch_size = 8
    loss_type = 'mix_dice'
    multitask = True
    validation_step = 1

    in_channels = 3
    num_classes = 3  # 分割类别数，二类分割设置为1，多类分割设置成 类别数+加背景
    # crop_height = 512 #输入图像裁切
    # crop_width = 256
    crop_height = 512  # 输入图像缩放
    crop_width = 256

    # 优化器设置
    lr = 0.005
    lr_mode = 'poly'
    momentum = 0.9
    weight_decay = 1e-4  # L2正则化系数

    # 预训练模型
    pretrained = False
    pretrained_model_path = None

    # 交叉验证
    k_fold = 4
    test_fold = 1

    # 断点恢复（latest checkpoint路径）
    resume_model_path = None
    continu = False
    # 模型保存设置
    save_every_checkpoint = False

    # GPU使用
    cuda = '1'
    num_workers = 0
    use_gpu = True

    # 网络预测
    trained_model_path = 'checkpoint/May20_09-37 最新网络实验/1_94Fiuld_unet.pkl'  # test的时候模型文件的选择（当mode='test'的时候用）
    predict_fold = f'{net_work}_predict_mask'
