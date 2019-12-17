'''
Test unified network with scene classification and place recognition
'''

from __future__ import print_function
import argparse
import random
from os.path import join, exists
from os import mkdir
import torch
from torch.utils.data import DataLoader
from torch import nn
import numpy as np
from UnifiedModel import SceneModel
import warnings
import csv

parser = argparse.ArgumentParser(description='ScenePlaceRecognitionTest')
parser.add_argument('--cacheBatchSize', type=int, default=96, help='Batch size for caching and testing')
parser.add_argument('--nocuda', action='store_true', help='Dont use cuda')
parser.add_argument('--threads', type=int, default=4, help='Number of threads for each data loader to use')
parser.add_argument('--seed', type=int, default=123, help='Random seed to use.')
parser.add_argument('--dataPath', type=str, default='data/', help='Path for centroid data.')
parser.add_argument('--cachePath', type=str, default='/tmp/', help='Path to save cache to.')
parser.add_argument('--resume', type=str, default='checkpoints_mbnet_pitts30_n11',
                    help='Path to load checkpoint from, for resuming training or testing.')
parser.add_argument('--ckpt', type=str, default='best',
                    help='Resume from latest or best checkpoint.', choices=['latest', 'best'])
parser.add_argument('--dataset', type=str, default='Pittsburgh', help='Dataset to use',
                    choices=['westlake', 'yuquan', 'Pittsburgh'])
parser.add_argument('-a', '--arch', type=str, default='mobilenet_v2', help='the backbone network',
                    choices=['resnet18', 'mobilenet_v2', 'shufflenet_v2'])
parser.add_argument('--netVLADtrainNum', type=int, default=11, help='Number of trained blocks in the backbone.')
parser.add_argument('--savePath', type=str, default='', help='where to save descriptors and classes.')
parser.add_argument('--reducedDim', type=int, default=512, help='the compressed dimension number.')
parser.add_argument('--randNum', type=int, default=10000, help='how many samples are used in PCA.')


def test(data_set, reduced_dim=4096, whiten=True):
    data_loader = DataLoader(dataset=data_set, num_workers=opt.threads,
                             batch_size=opt.cacheBatchSize, shuffle=False, pin_memory=cuda)
    if unified_model.pooling.lower() == 'netvlad':
        pool_size = unified_model.num_clusters * unified_model.encoder_dim
    feat = torch.zeros([len(data_set), pool_size])

    with torch.no_grad():  # 不会反向传播，提高inference速度
        print('====> Extracting Features')
        for iteration, (input_image, indices) in enumerate(data_loader, 1):
            input_image = input_image.to(device)
            # forward inference - place add scene information
            logit, dsc = unified_model.forward(input_image)
            feat[indices.detach().numpy(), :] = dsc.detach().cpu()

            if iteration % 50 == 0 or len(data_loader) <= 10:
                print("==> Batch ({}/{})".format(iteration, len(data_loader)), flush=True)

            del input_image, dsc
    del data_loader

    feat = feat.t()     # 按列排列向量
    feat_mean = torch.mean(feat, 1).unsqueeze(-1)
    feat = feat - feat_mean.expand_as(feat)

    print('===>Start SVD decomposition.')
    U, S, V = torch.svd(feat)
    U = U[:, :reduced_dim]
    S = S[:reduced_dim]
    if whiten:
        U = torch.mm(U, torch.diag(torch.div(1, torch.sqrt(S+1e-9))))
    Utmu = torch.mm(U.t(), feat_mean)

    conv = nn.Conv2d(pool_size, reduced_dim, kernel_size=(1, 1), bias=True, stride=1, padding=0)
    # weight参数的顺序 [out,in,filter,filter]
    conv.weight = nn.Parameter(U.t().unsqueeze(-1).unsqueeze(-1))
    conv.bias = nn.Parameter(-Utmu.squeeze(-1))

    torch.save(conv, 'pca'+str(reduced_dim)+'.pth')


if __name__ == "__main__":
    # ignore warnings -- UserWarning: Loky-backed parallel loops cannot be called in a multiprocessing, setting n_jobs=1
    warnings.filterwarnings("ignore")

    opt = parser.parse_args()

    # designate device
    cuda = not opt.nocuda
    if cuda and not torch.cuda.is_available():
        raise Exception("No GPU found, please run with --nocuda")

    device = torch.device("cuda" if cuda else "cpu")

    random.seed(opt.seed)
    np.random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    if cuda:
        torch.cuda.manual_seed(opt.seed)

    # designate dataset
    if opt.dataset.lower() == 'pittsburgh':
        from DataSet import pittsburgh as dataset
    else:
        raise Exception('Unknown dataset')
    print('===> Loading dataset(s)')
    random_train_set = dataset.get_random_training_set(rand_num=opt.randNum)
    # build Unified network architecture
    print('===> Building model')
    unified_model = SceneModel.UnifiedSceneNetwork(opt.arch, opt.netVLADtrainNum, num_clusters=64,
                                                   branch1_enable=1, branch2_enable=1, wpca=False)

    # load the paramters of the UnifiedModel branch
    if opt.ckpt.lower() == 'latest':
        resume_ckpt = join(opt.resume, 'checkpoint.pth.tar')
    elif opt.ckpt.lower() == 'best':
        resume_ckpt = join(opt.resume, 'model_best.pth.tar')
    unified_model.load_branch2_params(resume_ckpt)

    # execute test procedures
    print('===> Running evaluation step')
    unified_model = unified_model.to(device)
    unified_model.eval()
    test(random_train_set, reduced_dim=opt.reducedDim)
