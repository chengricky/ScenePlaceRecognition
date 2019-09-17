import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

import h5py
from os.path import join, exists

from NetAVLAD import netvlad
from NetAVLAD import attention as delfModel


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class L2Norm(nn.Module):
    def __init__(self, dim=1):
        super().__init__()
        self.dim = dim

    def forward(self, input):
        return F.normalize(input, p=2, dim=self.dim)

def baseAlexNet(pretrained=True, numTrain=1):
    encoder = models.alexnet(pretrained=pretrained)
    # capture only features and remove last relu and maxpool
    layers = list(encoder.features.children())[:-2]
    if pretrained:
        # if using pretrained only train conv5
        for l in layers[:-1*numTrain]:
            for p in l.parameters():
                p.requires_grad = False
    return layers

def baseVGG16(pretrained=False, numTrain=5):
    encoder = models.vgg16(pretrained=False)
    if pretrained is True:
        encoder.load_state_dict(torch.load('vgg16-397923af.pth'))
    # capture only feature part and remove last relu and maxpool
    layers = list(encoder.features.children())[:-2]
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

def baseResNet(type=50, numTrain=2):
    # loading resnet18 of trained on places365 as basenet
    from Place365 import wideresnet
    if type == 18:
        model_file = 'Place365/wideresnet18_places365.pth.tar'
        modelResNet = wideresnet.resnet18(num_classes=365)
    elif type == 34:
        model_file = 'Place365/resnet34-333f7ec4.pth'# Pretrained on ImageNet
        modelResNet = wideresnet.resnet34()
    elif type == 50:
        model_file = 'Place365/resnet50_places365.pth.tar'
        modelResNet = wideresnet.resnet50(num_classes=365)
    else:
        raise Exception('Unknown ResNet Type')
    # load object saved with torch.save() from a file, with funtion specifiying how to remap storage locations in the parameter list
    checkpoint = torch.load(model_file, map_location=lambda storage, loc: storage)  # gpu->cpu, why?!
    if type == 34:
        modelResNet.load_state_dict(checkpoint)
    else:
        state_dict = {str.replace(k, 'module.', ''): v for k, v in checkpoint['state_dict'].items()}  # 去掉module.字样
        modelResNet.load_state_dict(state_dict)
    layers = list(modelResNet.children())[:-2]  # children()只包括了第一代儿子模块，get rid of the last two layers: avepool & fc
    # 让最后1\2个block参与netVLAD训练
    for l in layers[:-1*numTrain]:
        for p in l.parameters():
            p.requires_grad = False
    return layers


def get_netavlad_model(opt, train_set, whole_test_set, middleAttention):

    pretrained = not opt.fromscratch
    arch = opt.arch.lower()
    mode = opt.mode.lower()
    numTrain = opt.numTrain
    withAttention = opt.withAttention
    dataPath = opt.dataPath
    pooling = opt.pooling.lower()
    resume = opt.resume
    num_clusters = opt.num_clusters
    remain = opt.remain
    vladv2 = opt.vladv2

    hook_dim = 0

    if arch == 'alexnet':
        encoder_dim = 256
        hook_layer = 6 #TODO fake value, to be determine
        layers = baseAlexNet(pretrained=pretrained, numTrain=numTrain)
    elif arch == 'vgg16':
        encoder_dim = 512
        hook_layer = 15 # vgg16-conv3
        hook_dim = 256
        layers = baseVGG16(pretrained=pretrained, numTrain=numTrain)
    elif arch == 'resnet18':
        encoder_dim = 512
        hook_layer = 2
        layers = baseResNet(type=18, numTrain=numTrain)
    elif arch == 'resnet34':
        encoder_dim = 512
        hook_layer = 2
        layers = baseResNet(type=34, numTrain=numTrain)
    elif arch == 'resnet50':
        encoder_dim = 2048
        hook_layer = 2
        layers = baseResNet(type=50, numTrain=numTrain)
    else:
        raise Exception('Unknown architecture')
    if mode == 'cluster':  # and opt.vladv2 == False #TODO add v1 v2 switching as flag
        layers.append(L2Norm())

    encoder = nn.Sequential(*layers)
    model = nn.Module()
    model.add_module('encoder', encoder)

    if withAttention:
        delf = delfModel.DELF(numc_featmap=encoder_dim, remain=remain)
        delf.init()
        model.add_module('attention', delf)
        if middleAttention:
            delf_mid = delfModel.DELF(numc_featmap=hook_dim, remain=remain)
            delf_mid.init()
            model.add_module('attention_mid', delf_mid)


    ### 初始化model中的pooling模块
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

