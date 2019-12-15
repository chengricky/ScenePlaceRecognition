"""
This file builds the unified network model for both scene classification and place recognition.
Ricky 2019.Dec.10
"""

import torch
import torch.nn as nn
from os.path import isfile
from Place365 import loadNetwork
from UnifiedModel import netvlad


class L2Norm(nn.Module):
    def __init__(self, dim=1):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        return nn.functional.normalize(x, p=2, dim=self.dim)


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class UnifiedSceneNetwork(nn.Module):
    def __init__(self, arch, num_trained, num_clusters, branch1_enable=0, branch2_enable=1,
                 mode='test', pooling='netvlad', cls_ckpt=True, wpca=True, red_dim=512):
        """
        Args:
            arch : str
                The type of backbone network.
            num_trained : int
                The number of trained layers at the end of backbone network.
            branch1_enable : int
                The branch enable switch. scene classification
            branch2_enable : int
                The branch enable switch. scene description
            mode : str
                The mode of network working status, used to determine forward() and gradient.
            pooling : str
                The type of pooling module in the branch of scene description.
        """
        super().__init__()
        if branch2_enable and branch1_enable and mode == 'train':
            raise RuntimeError('The two branches can not be trained in same time.')
        elif branch1_enable == 0 and branch2_enable == 0:
            raise RuntimeError('One of the branch must be activate')
        self.arch = arch.lower()
        self.mode = mode
        self.pooling = pooling
        self.encoder_dim = 0
        self.cls_ckpt = cls_ckpt
        self.branch1_enable = branch1_enable
        self.branch2_enable = branch2_enable
        self.num_clusters = num_clusters
        self.num_branch = num_trained
        self.wpca = wpca
        self.red_dim = red_dim
        self.base_net, self.branch_1, self.branch_2, self.classifier = self.decompose_network()
        self.pool = self.get_pooling()
        self.set_gradient()

    def get_network(self):
        """
        Return the scene classification branch with checkpoint in default
        """
        if self.arch == 'resnet18':
            scene_model = loadNetwork.get_resnet(18, load_ckpt=self.cls_ckpt)
        elif self.arch == 'mobilenet_v2':
            scene_model = loadNetwork.get_mobilenet(load_ckpt=self.cls_ckpt)
        elif self.arch == 'shufflenet_v2':
            scene_model = loadNetwork.get_shufflenet(load_ckpt=self.cls_ckpt)
        else:
            raise RuntimeError('Unrecognized Network.')
        return scene_model

    def decompose_network(self):
        """
        Return basenet, branch, and classifier part of network.
        The cut-point between branch and classifier is stationary,
        and the cut-point between base_net and branch is determined by the parameter.
        """
        scene_model = self.get_network()
        model_list = list(scene_model.children())
        if self.arch == 'resnet18':
            base_net = nn.Sequential(*model_list[:-1*self.num_branch - 3])
            branch1 = branch2 = nn.Sequential(*model_list[-1*self.num_branch:-3])
            classifier = nn.Sequential(*model_list[-3:])
            self.encoder_dim = 512

        elif self.arch == 'mobilenet_v2':
            feature = list(model_list[0].children())
            base_net = nn.Sequential(*feature[:-1 * self.num_branch - 1])
            branch1 = nn.Sequential(*feature[-1 * self.num_branch - 1:])
            branch2 = nn.Sequential(*feature[-1 * self.num_branch - 1:-1])
            classifier = nn.Sequential(*model_list[1:])
            self.encoder_dim = 320

        elif self.arch == 'shufflenet_v2':
            feature = [m.children() for m in model_list[:5]]
            base_net = nn.Sequential(*feature[:-1 * self.num_branch])
            branch1 = branch2 = nn.Sequential(*feature[-1 * self.num_branch:])
            classifier = nn.Sequential(*model_list[5:])
            self.encoder_dim = 464

        else:
            raise RuntimeError('Unrecognized Network Architecture.')

        return base_net, branch1, branch2, classifier

    def get_pooling(self):
        """
        The pooling module in the branch of scene description (place recognition).
        """
        if self.mode == 'cluster':
            return L2Norm()
        elif self.pooling == 'netvlad':
            return netvlad.NetVLAD(num_clusters=self.num_clusters, dim=self.encoder_dim)
        elif self.pooling == 'max':
            global_pool = nn.AdaptiveMaxPool2d((1, 1))
            return nn.Sequential(*[global_pool, Flatten(), L2Norm()])
        elif self.pooling == 'avg':
            global_pool = nn.AdaptiveAvgPool2d((1, 1))
            return nn.Sequential(*[global_pool, Flatten(), L2Norm()])

    def set_gradient(self):
        if self.branch2_enable and self.mode == 'train':
            for p in self.base_net.parameters():
                p.requires_grad = False

    def forward(self, x):
        x = self.base_net(x)
        if self.branch1_enable:
            x1 = self.branch_1(x)
            x1 = x1.mean([2, 3])
            x1 = self.classifier(x1)
        if self.branch2_enable:
            x2 = self.branch_2(x)
            x2 = self.pool(x2)
            if self.wpca:
                x2 = self.wpca(x2.unsqueeze(-1).unsqueeze(-1))
                x2 = x2.squeeze(-1).squeeze(-1)

        return x1, x2

    def load_branch2_params(self, resume_ckpt, wpca_ckpt=''):
        """
        读入netVLAD分支的预先训练结果
        """
        if isfile(resume_ckpt):
            print("=> loading checkpoint '{}'".format(resume_ckpt))
            checkpoint = torch.load(resume_ckpt, map_location=lambda storage, loc: storage)
            state_dict = {k: v for k, v in checkpoint['state_dict'].items()}
            if self.arch == 'mobilenet_v2':
                num_base = list(self.base_net.children()).__len__()
                for k in list(state_dict.keys()):
                    if 'encoder' in k and int(k.split('.')[1]) < num_base:
                        del state_dict[k]
                for k in list(state_dict.keys()):
                    if 'encoder' in k:
                        key_sublist = k.split('.')
                        key_sublist[1] = str(int(key_sublist[1])-num_base)
                        new_k = '.'.join(key_sublist[1:])
                        state_dict[new_k] = state_dict.pop(k)
            self.branch_2.load_state_dict(state_dict, strict=False)
            self.pool.load_state_dict(state_dict, strict=False)

            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(resume_ckpt, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(resume_ckpt))

        if self.wpca:
            self.wpca = torch.load(wpca_ckpt)