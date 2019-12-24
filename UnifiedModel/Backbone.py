"""
This file is to define the NetVLAD network based on different backbones for the phase of NetVLAD training.
Ricky 2019.Dec.14
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import h5py
import numpy as np
from os.path import join, exists
from UnifiedModel import netvlad


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class L2Norm(nn.Module):
    def __init__(self, dim=1):
        super().__init__()
        self.dim = dim

    def forward(self, input):
        return F.normalize(input, p=2, dim=self.dim)


def get_alexnet(pre_trained):
    return models.alexnet(pretrained=pre_trained)


def baseAlexNet(pre_trained=True, num_train=1):
    encoder = get_alexnet(pre_trained)
    # capture only features and remove last ReLU and MaxPool
    layers = list(encoder.features.children())[:-2]
    if pre_trained:
        # if using pre-trained only train Conv-5
        for l in layers[:-1 * num_train]:
            for p in l.parameters():
                p.requires_grad = False
    return layers


def get_vgg16(pre_trained):
    encoder = models.vgg16(pretrained=False)
    if pre_trained is True:
        encoder.load_state_dict(torch.load('vgg16-397923af.pth'))
    return encoder


def baseVGG16(pre_trained=False, numTrain=5):
    encoder = get_vgg16(pre_trained)
    # capture only feature part and remove last ReLU and MaxPool
    layers = list(encoder.features.children())[:-2]

    if numTrain <= 0:
        for layer in layers:
            for p in layer.parameters():
                p.requires_grad = False
    else:
        # if using pre-trained then only train conv5_1, conv5_2, and conv5_3
        for layer in layers[:-1*numTrain]:
            for p in layer.parameters():
                p.requires_grad = False
    return layers


def get_resnet(type):
    """
    loading ResNet-18/34/50 of trained on places365 (ImageNet) as basenet
    """
    from Place365 import wideresnet
    if type == 18:
        model_file = 'Place365/wideresnet18_places365.pth.tar'
        model_res_net = wideresnet.resnet18(num_classes=365)
    elif type == 34:
        model_file = 'Place365/resnet34-333f7ec4.pth'   # Pre-trained on ImageNet
        model_res_net = wideresnet.resnet34()
    elif type == 50:
        model_file = 'Place365/resnet50_places365.pth.tar'
        model_res_net = wideresnet.resnet50(num_classes=365)
    else:
        raise Exception('Unknown ResNet Type')
    return model_res_net, model_file


def baseResNet(pre_trained=True, type=50, num_train=2):
    modelResNet, model_file = get_resnet(type)
    if pre_trained:
        checkpoint = torch.load(model_file, map_location=lambda storage, loc: storage)
        if type == 34:
            modelResNet.load_state_dict(checkpoint)
        else:
            state_dict = {str.replace(k, 'module.', ''): v for k, v in checkpoint['state_dict'].items()}  # 去掉module.字样
            modelResNet.load_state_dict(state_dict)

    # children()只包括了第一代儿子模块，get rid of the last two layers: AvePool & fc
    layers = list(modelResNet.children())[:-2]
    # 让最后1\2个block参与netVLAD训练
    for l in layers[:-1 * num_train]:
        for p in l.parameters():
            p.requires_grad = False
    return layers


def get_mobilenet(pre_trained=True):
    import torchvision.models.mobilenet as mobilenet
    model_mobile_net = mobilenet.MobileNetV2(num_classes=365)
    if pre_trained:
        model_file = 'Place365/mobilenet_v2_best.pth.tar'
        checkpoint = torch.load(model_file, map_location=lambda storage, loc: storage)
        state_dict = {str.replace(k, 'module.', ''): v for k, v in checkpoint['state_dict'].items()}
        model_mobile_net.load_state_dict(state_dict)
    return model_mobile_net


def baseMobileNet(pre_trained=True, num_train=5):
    modelMobileNet = get_mobilenet(pre_trained)
    layers = list(list(modelMobileNet.children())[0].children())[:-1]
    for l in layers[:-1 * num_train]:
        for p in l.parameters():
            p.requires_grad = False
    return layers


def get_shfflenet(pre_trained=True):
    import torchvision.models.shufflenetv2 as shufflenet
    model_shuffle_net = shufflenet.shufflenet_v2_x1_0(num_classes=365)
    if pre_trained:
        model_file = 'Place365/shufflenet_v2_x1_0_best.pth.tar'
        checkpoint = torch.load(model_file, map_location=lambda storage, loc: storage)
        state_dict = {str.replace(k, 'module.', ''): v for k, v in checkpoint['state_dict'].items()}
        model_shuffle_net.load_state_dict(state_dict)
    return model_shuffle_net


def baseShuffleNet(pre_trained=True, num_train=5):
    net = get_shfflenet(pre_trained)
    model_list = list(net.children())
    layers = []
    for m in model_list[:5]:
        layers.extend(list(m.children()))
    for layer in layers[:-1 * num_train]:
        for p in layer.parameters():
            p.requires_grad = False
    return layers


hook_features = []


def get_hook(module, input, output):
    hook_features.append(np.squeeze(output.data.cpu().numpy()))


def get_netavlad_model(opt, train_set, whole_test_set):
    """
    Return the NetVLAD network (with attention module and/or intermediate feature maps)
    """
    pretrained = not opt.fromscratch
    arch = opt.arch.lower()
    mode = opt.mode.lower()
    dataPath = opt.dataPath
    pooling = opt.pooling.lower()
    resume = opt.resume
    num_clusters = opt.num_clusters

    hook_dim = 0

    if arch == 'alexnet':
        encoder_dim = 256
        hook_layer = 6  # TODO: fake value, to be determine
        layers = baseAlexNet(pre_trained=pretrained, num_train=opt.numTrain)
    elif arch == 'vgg16':
        encoder_dim = 512
        # vgg16-conv3(pooling之前的ReLU层，0-base)
        hook_layer = 15
        hook_dim = 256
        layers = baseVGG16(pre_trained=pretrained, numTrain=opt.numTrain)
    elif arch == 'resnet18':
        encoder_dim = 512
        # the output of the second block
        hook_layer = 4
        layers = baseResNet(pre_trained=pretrained,type=18, num_train=opt.numTrain)
    elif arch == 'resnet34':
        encoder_dim = 512
        hook_layer = 2
        layers = baseResNet(pre_trained=pretrained,type=34, num_train=opt.numTrain)
    elif arch == 'resnet50':
        encoder_dim = 2048
        hook_layer = 2
        layers = baseResNet(pre_trained=pretrained,type=50, num_train=opt.numTrain)
    elif arch == 'mobilenet2':
        encoder_dim = 320
        hook_layer = 2
        layers = baseMobileNet(pre_trained=pretrained, num_train=opt.numTrain)
    elif arch == 'shufflenet2':
        encoder_dim = 464
        hook_layer = 2
        layers = baseShuffleNet(pre_trained=pretrained, num_train=opt.numTrain)
    else:
        raise Exception('Unknown architecture')
    if mode == 'cluster':
        layers.append(L2Norm())

    if opt.saveDecs:
        layers[hook_layer].register_forward_hook(get_hook)

    encoder = nn.Sequential(*layers)    # 参数数目不定时，使用*号作为可变参数列表，就可以在方法内对参数进行调用。
    model = nn.Module()
    model.add_module('encoder', encoder)


    # 初始化model中的pooling模块
    if mode != 'cluster':
        if pooling == 'netvlad':
            net_vlad = netvlad.NetVLAD(num_clusters=num_clusters, dim=encoder_dim)
            if not resume:
                if mode == 'train':
                    initcache = join(dataPath, 'centroids', arch + '_' + train_set.dataset + '_' + str(
                        num_clusters) + '_desc_cen.hdf5')
                else:
                    initcache = join(dataPath, 'centroids', arch + '_' + whole_test_set.dataset + '_' + str(
                        num_clusters) + '_desc_cen.hdf5')

                if not exists(initcache):
                    raise FileNotFoundError('Could not find clusters, please run with --mode=cluster before proceeding')

                with h5py.File(initcache, mode='r') as h5:
                    clsts = h5.get("centroids")[...]
                    traindescs = h5.get("descriptors")[...]
                    net_vlad.init_params(clsts, traindescs)
                    del clsts, traindescs

            model.add_module('pool', net_vlad)

        elif pooling == 'max':
            global_pool = nn.AdaptiveMaxPool2d((1, 1))
            model.add_module('pool', nn.Sequential(*[global_pool, Flatten(), L2Norm()]))

        elif pooling == 'avg':
            global_pool = nn.AdaptiveAvgPool2d((1, 1))
            model.add_module('pool', nn.Sequential(*[global_pool, Flatten(), L2Norm()]))

        else:
            raise ValueError('Unknown pooling type: ' + pooling)

    return model, encoder_dim, hook_dim

