# Train the NetVLAD network
from __future__ import print_function

from math import ceil
import random, shutil, json
from os.path import join, exists
from os import makedirs

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
import TestScript
import TrainScript


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


class RunningVariables:
    def __init__(self):
        self.opt = 0
        self.train_set = 0
        self.whole_train_set = 0
        self.whole_training_data_loader = 0
        self.whole_test_set = 0
        self.dataset = 0
        self.model = 0
        self.encoder_dim = 0
        self.hook_dim = 0
        self.device = 0

    def set_opt(self, opt_):
        self.opt = opt_

    def set_dataset(self, train_set_, whole_train_set_, whole_training_data_loader_, whole_test_set_, dataset_):
        self.train_set = train_set_
        self.whole_train_set = whole_train_set_
        self.whole_training_data_loader = whole_training_data_loader_
        self.whole_test_set = whole_test_set_
        self.dataset = dataset_

    def set_model(self, model_, encoder_dim_, hook_dim_):
        self.model = model_
        self.encoder_dim = encoder_dim_
        self.hook_dim = hook_dim_

    def set_device(self, device_):
        self.device = device_


if __name__ == "__main__":
    # ignore warnings -- UserWarning: Loky-backed parallel loops cannot be called in a multiprocessing, setting n_jobs=1
    warnings.filterwarnings("ignore")

    rv = RunningVariables()

    # get arguments from the json file or the command
    opt = arguments.get_args()
    print(opt)
    rv.set_opt(opt)

    ## desinate the device (CUDA) to train
    if not torch.cuda.is_available():
        raise Exception("No GPU found, program terminated")
    device = torch.device("cuda")
    random.seed(opt.seed)
    np.random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed(opt.seed)
    rv.set_device(device)

    print('===> Loading dataset(s)')
    train_set, whole_train_set, whole_training_data_loader, whole_test_set, dataset = \
        loadDataset.loadDataSet(opt.mode.lower(), opt.split.lower(), opt.dataset.lower(),
                                opt.threads, opt.cacheBatchSize, opt.margin)
    rv.set_dataset(train_set, whole_train_set, whole_training_data_loader, whole_test_set, dataset)

    ## 构造网络模型
    print('===> Building model')
    model, encoder_dim, hook_dim = netavlad.get_netavlad_model(opt=opt, train_set=train_set,
                                                               whole_test_set=whole_test_set,
                                                               middleAttention=False)

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
                                                            opt.mode.lower(), opt, opt.nGPU, device, model)
    if not opt.resume:
        model = model.to(device)

    rv.set_model(model, encoder_dim, hook_dim)

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
        criterion = nn.TripletMarginLoss(margin=opt.margin ** 0.5, p=2, size_average=False).to(
            device)  # reduction='sum'

    ## 执行test/cluster/train操作
    if opt.mode.lower() == 'test':
        print('===> Running evaluation step')
        epoch = 1
        recalls = TestScript.test(rv, opt, epoch, write_tboard=False)

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
            TrainScript.train(rv, writer, opt, epoch)
            if (epoch % opt.evalEvery) == 0:
                recalls = TestScript.test(rv, opt, epoch, write_tboard=True)
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
