import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

import h5py
from os.path import join, exists

from NetAVLAD import netvlad
from NetAVLAD import attention as delfModel

import numpy as np


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class L2Norm(nn.Module):
    def __init__(self, dim=1):
        super().__init__()
        self.dim = dim

    def forward(self, input):
        return F.normalize(input, p=2, dim=self.dim)


def get_alexnet(pretrained):
    return models.alexnet(pretrained=pretrained)


def baseAlexNet(pretrained=True, numTrain=1):
    encoder = get_alexnet(pretrained)
    # capture only features and remove last relu and maxpool
    layers = list(encoder.features.children())[:-2]
    if pretrained:
        # if using pretrained only train conv5
        for l in layers[:-1*numTrain]:
            for p in l.parameters():
                p.requires_grad = False
    return layers


def get_vgg16(pretrained):
    encoder = models.vgg16(pretrained=False)
    if pretrained is True:
        encoder.load_state_dict(torch.load('vgg16-397923af.pth'))
    return encoder


def baseVGG16(pretrained=False, numTrain=5):
    encoder = get_vgg16(pretrained)
    # capture only feature part and remove last relu and maxpool
    layers = list(encoder.features.children())[:-2]
    # print(len(layers))
    # print(layers)
    # input()
    if pretrained:
        if numTrain<=0:
            for l in layers:
                for p in l.parameters():
                    p.requires_grad = False
        else:
            # if using pretrained then only train conv5_1, conv5_2, and conv5_3
            for l in layers[:-1*numTrain]:
                for p in l.parameters():
                    p.requires_grad = False
    return layers


def get_resnet(type):
    """
    loading resnet18 of trained on places365 as basenet
    """
    from Place365 import wideresnet
    if type == 18:
        model_file = 'Place365/wideresnet18_places365.pth.tar'
        model_res_net = wideresnet.resnet18(num_classes=365)
    elif type == 34:
        model_file = 'Place365/resnet34-333f7ec4.pth'# Pretrained on ImageNet
        model_res_net = wideresnet.resnet34()
    elif type == 50:
        model_file = 'Place365/resnet50_places365.pth.tar'
        model_res_net = wideresnet.resnet50(num_classes=365)
    else:
        raise Exception('Unknown ResNet Type')
    return model_res_net, model_file


def baseResNet(type=50, numTrain=2):
    modelResNet, model_file = get_resnet(type)
    # load object saved with torch.save() from a file, with funtion specifiying how to remap storage locations in the parameter list
    checkpoint = torch.load(model_file, map_location=lambda storage, loc: storage)
    if type == 34:
        modelResNet.load_state_dict(checkpoint)
    else:
        state_dict = {str.replace(k, 'module.', ''): v for k, v in checkpoint['state_dict'].items()}  # 去掉module.字样
        modelResNet.load_state_dict(state_dict)
    layers = list(modelResNet.children())[:-2]  # children()只包括了第一代儿子模块，get rid of the last two layers: avepool & fc
    # print(layers)
    # input()
    # 让最后1\2个block参与netVLAD训练
    for l in layers[:-1*numTrain]:
        for p in l.parameters():
            p.requires_grad = False
    return layers


def get_mobilenet():
    import torchvision.models.mobilenet as mobilenet
    model_file = 'Place365/mobilenet_v2_best.pth.tar'
    model_mobile_net = mobilenet.MobileNetV2(num_classes=365)
    checkpoint = torch.load(model_file, map_location=lambda storage, loc: storage)
    state_dict = {str.replace(k, 'module.', ''): v for k, v in checkpoint['state_dict'].items()}  # 去掉module.字样
    model_mobile_net.load_state_dict(state_dict)
    return model_mobile_net


def baseMobileNet(type=2, numTrain=5):
    modelMobileNet = get_mobilenet()
    layers = list(list(modelMobileNet.children())[0].children())[:-1]
    for l in layers[:-1*numTrain]:
        for p in l.parameters():
            p.requires_grad = False
    return layers

def get_shufflenet():
    import torchvision.models.shufflenetv2 as shufflenet
    model_file = 'Place365/shufflenet_v2_best.pth.tar'
    model_shuffle_net = shufflenet.shufflenet_v2_x1_0(num_classes=365)
    checkpoint = torch.load(model_file, map_location=lambda storage, loc: storage)
    state_dict = {str.replace(k, 'module.', ''): v for k, v in checkpoint['state_dict'].items()}  # 去掉module.字样
    model_shuffle_net.load_state_dict(state_dict)
    return model_shuffle_net

hook_features = []


def get_hook(module, input, output):
    hook_features.append(np.squeeze(output.data.cpu().numpy()))


def get_netavlad_model(opt, train_set, whole_test_set, middleAttention):

    pretrained = not opt.fromscratch
    arch = opt.arch.lower()
    mode = opt.mode.lower()
    dataPath = opt.dataPath
    pooling = opt.pooling.lower()
    resume = opt.resume
    num_clusters = opt.num_clusters
    remain = opt.remain
    vladv2 = opt.vladv2

    hook_dim = 0

    if arch == 'alexnet':
        encoder_dim = 256
        hook_layer = 6 # TODO fake value, to be determine
        layers = baseAlexNet(pretrained=pretrained, numTrain=opt.numTrain)
    elif arch == 'vgg16':
        encoder_dim = 512
        hook_layer = 15 # vgg16-conv3(pooling之前的relu层，0-base)
        hook_dim = 256
        layers = baseVGG16(pretrained=pretrained, numTrain=opt.numTrain)
    elif arch == 'resnet18':
        encoder_dim = 512
        hook_layer = 4 # the output of the second block
        layers = baseResNet(type=18, numTrain=opt.numTrain)
    elif arch == 'resnet34':
        encoder_dim = 512
        hook_layer = 2
        layers = baseResNet(type=34, numTrain=opt.numTrain)
    elif arch == 'resnet50':
        encoder_dim = 2048
        hook_layer = 2
        layers = baseResNet(type=50, numTrain=opt.numTrain)
    elif arch == 'mobilenet':
        encoder_dim = 320
        hook_layer = 2
        layers = baseMobileNet(type=2, numTrain=opt.numTrain)
    else:
        raise Exception('Unknown architecture')
    if mode == 'cluster':  # and opt.vladv2 == False #TODO add v1 v2 switching as flag
        layers.append(L2Norm())

    if opt.saveDecs:
        layers[hook_layer].register_forward_hook(get_hook)

    encoder = nn.Sequential(*layers)    # 参数数目不定时，使用*号作为可变参数列表，就可以在方法内对参数进行调用。
    model = nn.Module()
    model.add_module('encoder', encoder)

    if opt.withAttention:
        delf = delfModel.DELF(numc_featmap=encoder_dim, remain=remain)
        delf.init()
        model.add_module('attention', delf)
        if middleAttention:
            delf_mid = delfModel.DELF(numc_featmap=hook_dim, remain=remain)
            delf_mid.init()
            model.add_module('attention_mid', delf_mid)

    # 初始化model中的pooling模块
    if mode != 'cluster':
        if pooling == 'netvlad':
            net_vlad = netvlad.NetVLAD(num_clusters=num_clusters, dim=encoder_dim, vladv2=vladv2)
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

