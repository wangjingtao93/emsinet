from models.nets.UNet import UNet,UNet_Nested,UNet_Nested_dilated
# from models.nets.unet.unet_model import UNet
from models.nets.resnet_UNet import resnet34_UNet,resnet50_UNet
from models.nets.CPFNet import CPFNet
from models.nets.CE_Net import CE_Net
from models.nets.AttU_Net import AttU_Net
from models.nets.model_mannal import *
from models.nets.deeplabv3 import DeepLabv3_plus
from models.nets.pspnet import PSPNet
from models.nets.praNet import PraNet
from models.nets.double_net import Double_UNet

def net_builder(name,pretrained_model=None,pretrained=False):
    if name == 'resnet50_unet':
        net = resnet50_UNet(pretrained=pretrained)
    elif name == 'resnet34_unet':
        net = resnet34_UNet(pretrained=pretrained)
    # elif name == 'resnet34_unet_aux':
    #     net = resnet_UNet_aux(pretrained=pretrained)
        
    elif name == 'unet':
        net = UNet(in_channels = 1,n_classes=3,feature_scale=2)
    elif name == 'cpfnet':
        net = CPFNet(in_channels=1, out_planes=3)

    elif name == 'ce_net':
        net = CE_Net(in_channels=1, num_classes=3)

    elif name == 'attu_net':
        net = AttU_Net(in_channels=1, num_classes=3, feature_scale=2)

    elif name == 'deeplabv3':
        net = DeepLabv3_plus(nInputChannels=1, n_classes=3, os=16, pretrained=False, _print=True)
    elif name == 'pspnet':
        net = PSPNet(layers=50, bins=(1, 2, 3, 6), dropout=0.1, classes=3, zoom_factor=8, use_ppm=True, pretrained=False)

    elif name == 'model_me':
        net = UNet_DAFM(1,3,feature_scale =2)

    elif name == 'model_skip':
        net = UNet_nonlocal_skip(1,3,feature_scale=2)
    elif name == 'model_skip_ds':
        net = UNet_nonlocal_skip_deepsupervision(1,3,feature_scale=2)

    elif name == 'edge_net':
        net = Double_UNet(1,3,feature_scale =2)
    elif name == 'pranet':
        net = PraNet()
    # elif name == 'unet_new':
    #     net = UNet(n_channels=1, n_classes=3, bilinear=True)
    else:
        raise NameError("Unknow Model Name!")
    return net
