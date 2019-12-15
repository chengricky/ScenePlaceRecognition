"""
This file is designed to load networks for the unified network
"""
import torch
from Place365 import wideresnet
import torchvision.models.mobilenet as mobilenet
import torchvision.models.shufflenetv2 as shufflenet


def get_shufflenet(load_ckpt=True):
    """
    loading ShuffleNet V2 of trained on places365 as basenet
    """
    model_shuffle_net = shufflenet.shufflenet_v2_x1_0(num_classes=365)
    if load_ckpt:
        model_file = 'Place365/shufflenet_v2_x1_0_best.pth.tar'
        try:
            checkpoint = torch.load(model_file, map_location=lambda storage, loc: storage)
        except IOError:
            print('Fail to load checkpoint, because no such file:', model_file)
        else:
            state_dict = {str.replace(k, 'module.', ''): v for k, v in checkpoint['state_dict'].items()}
            model_shuffle_net.load_state_dict(state_dict)
    return model_shuffle_net


def get_mobilenet(load_ckpt=True):
    """
    loading MobileNet V2 of trained on places365 as basenet
    """
    model_mobile_net = mobilenet.MobileNetV2(num_classes=365)
    if load_ckpt:
        model_file = 'Place365/mobilenet_v2_best.pth.tar'
        try:
            checkpoint = torch.load(model_file, map_location=lambda storage, loc: storage)
        except IOError:
            print('Fail to load checkpoint, because no such file:', model_file)
        else:
            state_dict = {str.replace(k, 'module.', ''): v for k, v in checkpoint['state_dict'].items()}
            model_mobile_net.load_state_dict(state_dict)
    return model_mobile_net


def get_resnet(type, load_ckpt=True):
    """
    loading ResNet of trained on places365 as basenet
    """
    if type == 18:
        model_file = 'Place365/wideresnet18_places365.pth.tar'
        model_res_net = wideresnet.resnet18(num_classes=365)
    elif type == 34:
        model_file = 'Place365/resnet34-333f7ec4.pth'  # Pre-trained on ImageNet
        model_res_net = wideresnet.resnet34()
    elif type == 50:
        model_file = 'Place365/resnet50_places365.pth.tar'
        model_res_net = wideresnet.resnet50(num_classes=365)
    else:
        raise Exception('Unknown ResNet Type')
    if load_ckpt:
        try:
            checkpoint = torch.load(model_file, map_location=lambda storage, loc: storage)
        except IOError:
            print('Fail to load checkpoint, because no such file:', model_file)
        else:
            if type == 34:
                model_res_net.load_state_dict(checkpoint)
            else:
                state_dict = {str.replace(k, 'module.', ''): v for k, v in checkpoint['state_dict'].items()}
                model_res_net.load_state_dict(state_dict)
    return model_res_net
