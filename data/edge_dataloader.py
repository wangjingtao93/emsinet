import torch
import glob
import os
import numpy as np
import albumentations as A
import cv2

# from utils.utils import one_hot_it
import random


# from utils.config import DefaultConfig
class Slit_loader(torch.utils.data.Dataset):
    def __init__(self, dataset_path, scale, k_fold_test=1, mode='train', inputChannel=3):
        super().__init__()
        self.mode = mode
        self.scale = scale
        self.channel = inputChannel
        if mode != 'test':
            self.img_path = dataset_path + '/train' + '/image'
            self.mask_path = dataset_path + '/train' + '/label'
        else:
            self.img_path = dataset_path + '/test' + '/image'
            self.mask_path = dataset_path + '/test' + '/label'
        self.img_file, self.mask_file = self.read_list(self.img_path, k_fold_test=k_fold_test)

    def __getitem__(self, i):
        # load image
        # print(self.mask_file[i],self.img_file[i])
        mask_ = cv2.imread(self.mask_file[i], 0)
        mask = np.zeros_like(mask_)
        
        mask[mask_ == 127] = 1
        mask[mask_ == 255] = 2
        mask_shape = mask.shape
        if self.channel == 3:
            img = cv2.imread(self.img_file[i])
        else:
            img = cv2.imread(self.img_file[i], 0)
        mask = cv2.resize(mask, self.scale, interpolation=cv2.INTER_NEAREST)
        mask_edge = cv2.Canny(mask*125,10,100)/255.0
        img = cv2.resize(img, self.scale, interpolation=cv2.INTER_CUBIC)
        if self.mode == 'train':
            img, mask = self.transform(img, mask)
        

        if img.shape[-1] != 3:
            img = img[None]
        mask = mask[None]
        mask_edge = mask_edge[None]

        img = torch.from_numpy(img) / 255.0  # 注意图片已经进行了归一化了

        if img.shape[-1] == 3:
            img = img.permute(2, 0, 1)
        mask = torch.from_numpy(mask)  # y=由于标签已经进行过归一化，故不需要除以255.0
        mask_edge = torch.from_numpy(mask_edge)
        # print(img.size(), mask.size())

        return img, mask, mask_edge
        # load label

    def transform(self, image, mask):

        image5 = A.Compose([
            A.HorizontalFlip(p=0.5),
            # A.VerticalFlip(p=0.5),
            A.GaussNoise(p=0.7),  # 将高斯噪声应用于输入图像。
            A.OneOf([
                A.MotionBlur(p=0.2),  # 使用随机大小的内核将运动模糊应用于输入图像。
                A.MedianBlur(blur_limit=3, p=0.1),  # 中值滤波
                A.Blur(blur_limit=3, p=0.1),  # 使用随机大小的内核模糊输入图像。
            ], p=0.7),
            # A.ElasticTransform(p=0.5, alpha=120*0.1, sigma=120*0.1 * 0.05, alpha_affine=120*0.1 * 0.03),
            # A.GridDistortion(p=0.5, num_steps=50),
            A.ShiftScaleRotate(shift_limit=0.0625 / 2, scale_limit=0.2 / 2, rotate_limit=15, p=0.5),
            # 随机应用仿射变换：平移，缩放和旋转输入
            A.CLAHE(p=0.3),
            A.RandomBrightnessContrast(p=0.2),  # 随机明亮对比度
        ])(image=image, mask=mask)
        return image5["image"], image5["mask"]

    def __len__(self):
        return len(self.img_file)

    def read_list(self, image_path, k_fold_test=1):
        fold = sorted(os.listdir(image_path))
        img_list = []
        label_list = []

        if self.mode == 'train':
            fold_r = list(fold)
            fold_r.remove('f' + str(k_fold_test))  # remove testdata

            print("train fold =========== {}".format(fold_r))
            # fold_r = ['f2']
            for item in fold_r:
                cube_path = os.path.join(image_path, item)
                for cube_num in glob.glob(cube_path + '/*'):
                    img_list += glob.glob(cube_num + '/*')
                # label_list += glob.glob(cube_path.replace("img",'mask')+'/*')
            label_list = [elm.replace('image', 'label') for elm in img_list]
            # label_list_TMP = [x.replace("img","mask") for x in img_list]
            # label_list = [x.replace("jpg","png") for x in label_list_TMP]

        elif self.mode == 'val':
            fold_s = fold[k_fold_test - 1]
            print("val fold =========== {}".format(fold_s))
            cube_path = os.path.join(image_path, fold_s)
            for cube_num in glob.glob(cube_path + '/*'):
                img_list += glob.glob(cube_num + '/*')
            # label_list += glob.glob(cube_path.replace("img",'mask')+'/*')
            label_list = [elm.replace('image', 'label') for elm in img_list]

            # label_list_TMP = [x.replace("img","mask") for x in img_li  st]
            # label_list = [x.replace("jpg","png") for x in label_list_TMP]

        elif self.mode == 'test':
            print("test!")
            cube_path = image_path
            for cube_num in glob.glob(cube_path + '/*'):
                img_list += glob.glob(cube_num + '/*')
            label_list =[elm.replace('image', 'label') for elm in img_list]

        assert len(img_list) == len(label_list), print(len(img_list), len(label_list))
        print('Total {} image is:{}'.format(self.mode, len(img_list)))

        return img_list, label_list


# class Slit_loader_test(torch.utils.data.Dataset):
#
#     def __init__(self, dataset_list,label_list, scale, mode='train'):
#         super().__init__()
#         self.mode = mode
#         self.scale = scale
#         self.image_lists, self.label_lists = dataset_list, label_list
#         # data augmentation
#         self.aug = iaa.Sequential([
#             iaa.Fliplr(0.5),  # 百分之五十的图片上下翻转
#             iaa.SomeOf((0, 2), [
#                 iaa.Affine(
#                     rotate=(-10, 10)),  # 旋转
#             ], random_order=True)
#         ])  # 对比度
#         # resize
#         self.resize_label = transforms.Resize(scale, Image.NEAREST)
#         self.resize_img = transforms.Resize(scale, Image.BILINEAR)
#         # resize
#         self.to_gray = transforms.Grayscale()
#         # normalization
#         self.to_tensor = transforms.ToTensor()  # 将numpy的ndarray或PIL.Image读的图片转换成形状为(C,H, W)的Tensor格式，
#
#     def __getitem__(self, index):
#         # load image
#         img = Image.open(self.image_lists[index]).convert('RGB')
#         img = self.resize_img(img)
#         img = np.array(img).astype(np.uint8)
#         labels = self.label_lists[index]
#         # load label
#         if self.mode != 'test':
#             label = Image.open(self.label_lists[index])
#             label = self.resize_label(label)
#             label = np.array(label).astype(np.uint8)
#             label[label == 0] = 1
#             label[label == 255] = 0
#             # augment image and label
#             if self.mode == 'train':
#                 seq_det = self.aug.to_deterministic()  # 得到一个确定的增强函数
#                 # 将图片转换为SegmentationMapOnImage类型
#                 segmap = ia.SegmentationMapsOnImage(label, shape=label.shape)
#                 # segmap2 = ia.SegmentationMapOnImage(label, shape=label.shape, nb_classes=4)
#                 # 将方法应用在原图像上
#                 img = seq_det.augment_image(img)
#                 # 将方法应用在分割标签上，并且转换成np类型
#                 label = seq_det.augment_segmentation_maps([segmap])[0]
#                 label = label.get_arr().astype(np.uint8)
#             label = label.reshape((1, label.shape[0], label.shape[1]))
#             label_img = torch.from_numpy(label.copy()).float()
#
#             if self.mode == 'val' or self.mode == 'test':
#                 assert len(os.listdir(os.path.dirname(self.image_lists[index]))) == len(
#                     os.listdir(os.path.dirname(labels)))
#                 img_num = len(os.listdir(os.path.dirname(labels)))
#                 labels = (label_img, img_num)  # val模式下labels返回一个元祖，[0]为tensor标签索引值h*w，[1]为cube中包含slice张数int值
#             else:
#                 labels = label_img  # train模式下labels返回tensor标签索引值
#
#         img = (img.transpose(2, 0, 1)-127.5) / 127.5
#         img = torch.from_numpy(img.copy()).float()
#         # test模式下labels返回label的路径列表
#         return img, labels
#
#     def __len__(self):
#         return len(self.image_lists)
'''
if __name__ == '__main__':
    from torch.utils.data import DataLoader

    data = Slit_loader('../../../Dataset/PED', (512, 256), mode='train')
    dataloader_train = DataLoader(
        data,
        batch_size=2,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        drop_last = True
    )
    for i, (img, label) in enumerate(dataloader_train):
        print(img.shape)
        #print(label.shape)  # train
        print(img)
        print(label[0].shape)  # val
        # print(label)  # test

        print(i)


'''
