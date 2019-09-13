# Train the NetVLAD network
from __future__ import print_function
import argparse
from math import ceil
import random, shutil, json
from os.path import join, exists
from os import makedirs, remove

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

from datetime import datetime

import h5py
import faiss

from tensorboardX import SummaryWriter
import numpy as np
from NetAVLAD import model as netavlad
from DataSet import loadDataset
import loadCkpt

import warnings
import arguments

parser = argparse.ArgumentParser(description='PlaceRecognitionTrain')
parser.add_argument('--mode', type=str, default='train', help='Mode', choices=['train', 'cluster', 'test'])
parser.add_argument('--batchSize', type=int, default=4,
                    help='Number of triplets (query, pos, negs). Each triplet consists of 12 images.')
parser.add_argument('--cacheBatchSize', type=int, default=64, help='Batch size for caching and testing')
parser.add_argument('--cacheRefreshRate', type=int, default=1000,
                    help='How often to refresh cache, in number of queries. 0 for off')
parser.add_argument('--nEpochs', type=int, default=30, help='number of epochs to train for')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--nGPU', type=int, default=1, help='number of GPU to use.')
parser.add_argument('--optim', type=str, default='SGD', help='optimizer to use', choices=['SGD', 'ADAM'])
parser.add_argument('--lr', type=float, default=0.0001, help='Learning Rate.')
parser.add_argument('--lrStep', type=float, default=5, help='Decay LR ever N steps.')
parser.add_argument('--lrGamma', type=float, default=0.5, help='Multiply LR by Gamma for decaying.')
parser.add_argument('--weightDecay', type=float, default=0.001, help='Weight decay for SGD.')
parser.add_argument('--momentum', type=float, default=0.9, help='Momentum for SGD.')
parser.add_argument('--threads', type=int, default=8, help='Number of threads for each data loader to use')
parser.add_argument('--seed', type=int, default=123, help='Random seed to use.')
parser.add_argument('--dataPath', type=str, default='data/', help='Path for centroid data.')
parser.add_argument('--runsPath', type=str, default='runs/', help='Path to save runs to.')
parser.add_argument('--savePath', type=str, default='checkpoints_res18_pitts30/',
                    help='Path to save checkpoints to in logdir. Default=checkpoints/')
parser.add_argument('--cachePath', type=str, default='tmp/', help='Path to save cache to.')
parser.add_argument('--resume', type=str, default='',
                    help='Path to load checkpoint from, for resuming training or testing.')
parser.add_argument('--ckpt', type=str, default='latest',
                    help='Resume from latest or best checkpoint.', choices=['latest', 'best'])
parser.add_argument('--evalEvery', type=int, default=1,
                    help='Do a validation set run, and save, every N epochs.')
parser.add_argument('--patience', type=int, default=20, help='Patience for early stopping. 0 is off.')
parser.add_argument('--dataset', type=str, default='tokyo247',
                    help='Dataset to use', choices=['pittsburgh', 'tokyo247', 'highway', 'GB'])
parser.add_argument('--arch', type=str, default='resnet18',
                    help='basenetwork to use', choices=['vgg16', 'alexnet', 'resnet18', 'resnet34', 'resnet50'])
parser.add_argument('--pooling', type=str, default='netvlad', help='type of pooling to use',
                    choices=['netvlad', 'max', 'avg'])
parser.add_argument('--num_clusters', type=int, default=64, help='Number of NetVlad clusters. Default=64')
parser.add_argument('--margin', type=float, default=0.1, help='Margin for triplet loss. Default=0.1')
parser.add_argument('--split', type=str, default='val', help='Data split to use for testing. Default is val',
                    choices=['test', 'test250k', 'train', 'val'])
parser.add_argument('--fromscratch', action='store_true', help='Train from scratch rather than using pretrained models')
parser.add_argument('--numTrain', type=int, default=2, help='the number of trained layers of basenet')
parser.add_argument('--withAttention', action='store_true', help='Whether with the attention module.')
parser.add_argument('--remain', type=float, default=0.5, help='The remained ratio of feature map.')
parser.add_argument('--vladv2', action='store_true', help='whether to use VLADv2.')
parser.add_argument('--reduction', action='store_true', help='whether to perform PCA dimension reduction.')



def get_clusters(cluster_set):
    nDescriptors = 50000
    nPerImage = 100
    nIm = ceil(nDescriptors / nPerImage)

    sampler = SubsetRandomSampler(np.random.choice(len(cluster_set), nIm, replace=False))
    data_loader = DataLoader(dataset=cluster_set,
                             num_workers=opt.threads, batch_size=opt.cacheBatchSize, shuffle=False,
                             pin_memory=True,
                             sampler=sampler)

    if not exists(join(opt.dataPath, 'centroids')):
        makedirs(join(opt.dataPath, 'centroids'))

    initcache = join(opt.dataPath, 'centroids',
                     opt.arch + '_' + cluster_set.dataset + '_' + str(opt.num_clusters) + '_desc_cen.hdf5')
    with h5py.File(initcache, mode='w') as h5:
        with torch.no_grad():
            model.eval()
            print('====> Extracting Descriptors')
            dbFeat = h5.create_dataset("descriptors",
                                       [nDescriptors, encoder_dim],
                                       dtype=np.float32)  # float32

            for iteration, (input, indices) in enumerate(data_loader, 1):
                input = input.to(device)
                image_descriptors = model.encoder(input).view(input.size(0), encoder_dim, -1).permute(0, 2, 1)

                batchix = (iteration - 1) * opt.cacheBatchSize * nPerImage
                for ix in range(image_descriptors.size(0)):
                    # sample different location for each image in batch
                    sample = np.random.choice(image_descriptors.size(1), nPerImage, replace=False)
                    startix = batchix + ix * nPerImage
                    dbFeat[startix:startix + nPerImage, :] = image_descriptors[ix, sample, :].detach().cpu().numpy()

                if iteration % 50 == 0 or len(data_loader) <= 10:
                    print("==> Batch ({}/{})".format(iteration,
                                                     ceil(nIm / opt.cacheBatchSize)), flush=True)
                del input, image_descriptors

        print('====> Clustering..')
        niter = 100
        kmeans = faiss.Kmeans(encoder_dim, opt.num_clusters, niter=niter, verbose=False)
        kmeans.train(dbFeat[...])

        print('====> Storing centroids', kmeans.centroids.shape)
        h5.create_dataset('centroids', data=kmeans.centroids)
        print('====> Done!')


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    model_out_path = join(opt.savePath, filename)
    torch.save(state, model_out_path)
    if is_best:
        shutil.copyfile(model_out_path, join(opt.savePath, 'model_best.pth.tar'))


if __name__ == "__main__":
    # ignore warnings -- UserWarning: Loky-backed parallel loops cannot be called in a multiprocessing, setting n_jobs=1
    warnings.filterwarnings("ignore")

    ## read arguments from command or json file
    opt = parser.parse_args()
    restore_var = ['lr', 'lrStep', 'lrGamma', 'weightDecay', 'momentum', 'nGPU',
                   'runsPath', 'savePath', 'arch', 'num_clusters', 'pooling', 'optim',
                   'margin', 'seed', 'patience']
    if opt.resume:
        optLoaded = arguments.readArguments(opt, parser, restore_var)

    print(opt)

    ## desinate the device (CUDA) to train
    if not torch.cuda.is_available():
        raise Exception("No GPU found, program terminated")
    device = torch.device("cuda")
    random.seed(opt.seed)
    np.random.random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed(opt.seed)

    print('===> Loading dataset(s)')
    train_set, whole_train_set, whole_training_data_loader, whole_test_set, dataset = \
        loadDataset.loadDataSet(opt.mode.lower(), opt.split.lower(), opt.dataset.lower(),
                                opt.threads, opt.cacheBatchSize, opt.margin)

    ## 构造网络模型
    print('===> Building model')
    model, encoder_dim, hook_dim = netavlad.getNetAVLADModel(pretrained=not opt.fromscratch, arch=opt.arch.lower(),
                                                             mode=opt.mode.lower(), numTrain=opt.numTrain,
                                                             withAttention=opt.withAttention, dataPath=opt.dataPath,
                                                             pooling=opt.pooling.lower(), resume=opt.resume,
                                                             num_clusters=opt.num_clusters, train_set=train_set,
                                                             whole_test_set=whole_test_set, remain=opt.remain,
                                                             vladv2=opt.vladv2, middleAttention=False)

    isParallel = False
    if opt.nGPU > 1 and torch.cuda.device_count() > 1:
        # torch.distributed.init_process_group(backend="nccl", init_method='file:///mnt/lustre/chengruiqi/sharedfile', rank=0, world_size=4)
        print('Available GPU num = ', torch.cuda.device_count())

        model.encoder = nn.parallel.DataParallel(model.encoder)
        if opt.withAttention:
            model.attention = nn.parallel.DataParallel(model.attention)
        if opt.mode.lower() != 'cluster':
            model.pool = nn.parallel.DataParallel(model.pool)
            # model = nn.parallel.DataParallel(model)
        # else:
        #     model.encoder = nn.parallel.DataParallel(model.encoder)
        isParallel = True

    ## 读入预先训练结果
    if opt.resume:
        model, start_epoch, best_metric = loadCkpt.loadckpt(opt.ckpt.lower(), opt.resume, opt.start_epoch,
                                                            opt.mode.lower(), optLoaded, opt.nGPU, device, model)
    if not opt.resume:
        model = model.to(device)

    ## 定义优化器和损失函数
    if opt.mode.lower() == 'train':
        if opt.optim.upper() == 'ADAM':
            optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                                   lr=opt.lr, weight_decay=opt.weightDecay)  # , betas=(0,0.9))

        elif opt.optim.upper() == 'SGD':
            optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()),
                                  lr=opt.lr, momentum=opt.momentum, weight_decay=opt.weightDecay)
            # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=opt.lrStep, gamma=opt.lrGamma)
            scheduler = optim.lr_scheduler.ExponentialLR(optimizer, opt.lrGamma)

        else:
            raise ValueError('Unknown optimizer: ' + opt.optim)

        # original paper/code doesn't sqrt() the distances, we do, so sqrt() the margin, I think :D
        criterion = nn.TripletMarginLoss(margin=opt.margin ** 0.5, p=2, size_average=False).to(device)  # reduction='sum'

    ## 执行test/cluster/train操作
    if opt.mode.lower() == 'test':
        print('===> Running evaluation step')
        epoch = 1
        recalls = test(whole_test_set, epoch, write_tboard=False)

    elif opt.mode.lower() == 'cluster':
        print('===> Calculating descriptors and clusters')
        get_clusters(whole_train_set)

    elif opt.mode.lower() == 'train':
        print('===> Training model')
        writer = SummaryWriter(
            logdir=join(opt.runsPath, datetime.now().strftime('%b%d_%H-%M-%S') + '_' + opt.arch + '_' + opt.pooling))

        # write checkpoints in logdir
        logdir = writer.file_writer.get_logdir()
        # opt.savePath = join(logdir, opt.savePath)
        if not opt.resume:
            makedirs(opt.savePath)

        with open(join(opt.savePath, 'flags.json'), 'w') as f:
            f.write(json.dumps(
                {k: v for k, v in vars(opt).items()}
            ))
        print('===> Saving state to:', logdir)

        not_improved = 0
        best_score = 0
        for epoch in range(opt.start_epoch + 1, opt.nEpochs + 1):
            if opt.optim.upper() == 'SGD':
                scheduler.step(epoch)
            train(epoch)
            if (epoch % opt.evalEvery) == 0:
                recalls = test(whole_test_set, epoch, write_tboard=True)
                is_best = recalls[5] > best_score
                if is_best:
                    not_improved = 0
                    best_score = recalls[5]
                else:
                    not_improved += 1

                save_checkpoint({
                    'epoch': epoch,
                    'state_dict': model.state_dict(),
                    'recalls': recalls,
                    'best_score': best_score,
                    'optimizer': optimizer.state_dict(),
                    'parallel': isParallel,
                }, is_best)

                if opt.patience > 0 and not_improved > (opt.patience / opt.evalEvery):
                    print('Performance did not improve for', opt.patience, 'epochs. Stopping.')
                    break

        print("=> Best Recall@5: {:.4f}".format(best_score), flush=True)
        writer.close()
