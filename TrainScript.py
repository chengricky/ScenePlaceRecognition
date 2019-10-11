import torch
from torch.utils.data.dataset import Subset
from math import ceil
import numpy as np
from os.path import join
import h5py
from os import remove
from torch.utils.data import DataLoader
import time


def train(rv, writer, opt, epoch):
    epoch_loss = 0
    startIter = 1  # keep track of batch iter across subsets for logging

    if opt.cacheRefreshRate > 0:
        subsetN = ceil(len(rv.train_set) / opt.cacheRefreshRate)
        # TODO randomise the arange before splitting?
        subsetIdx = np.array_split(np.arange(len(rv.train_set)), subsetN)
    else:
        subsetN = 1
        subsetIdx = [np.arange(len(rv.train_set))]

    nBatches = (len(rv.train_set) + opt.batchSize - 1) // opt.batchSize

    for subIter in range(subsetN):
        print('====> Building Cache, subIter =', subIter)
        start_time = time.time()
        rv.model.eval()
        rv.train_set.cache = join(opt.cachePath, rv.train_set.whichSet + '_feat_cache.hdf5')
        with h5py.File(rv.train_set.cache, mode='w') as h5:
            pool_size = rv.encoder_dim
            if opt.pooling.lower() == 'netvlad':
                pool_size *= opt.num_clusters
            h5feat = h5.create_dataset("features", [len(rv.whole_train_set), pool_size], dtype=np.float32)
            with torch.no_grad():
                for iteration, (input, indices) in enumerate(rv.whole_training_data_loader, 1):
                    image_encoding = rv.model.encoder(input.to(rv.device))
                    if opt.withAttention:
                        image_encoding = rv.model.attention(image_encoding)
                        vlad_encoding = rv.model.pool(image_encoding)
                        del image_encoding
                    else:
                        vlad_encoding = rv.model.pool(image_encoding)
                        del image_encoding
                    h5feat[indices.detach().numpy(), :] = vlad_encoding.detach().cpu().numpy()
                    del input, vlad_encoding

        elapsed = time.time() - start_time
        # used to store Tensors on the GPU
        print('Allocated:', torch.cuda.memory_allocated())
        # used on the CPU by pytorch (shown in nvidia-smi)
        print('Cached:', torch.cuda.memory_cached())
        torch.cuda.empty_cache()
        print('====> Building Cache, Time =', int(elapsed), "s")

        print('====> Loading Sub Training Dataset, subIter =', subIter)
        start_time = time.time()
        sub_train_set = Subset(dataset=rv.train_set, indices=subsetIdx[subIter])
        training_data_loader = DataLoader(dataset=sub_train_set, num_workers=opt.threads,
                                          batch_size=opt.batchSize, shuffle=True,
                                          collate_fn=rv.dataset.collate_fn, pin_memory=True,
                                          drop_last=True)
        rv.model.train()
        for iteration, (query, positives, negatives,
                        negCounts, indices) in enumerate(training_data_loader, startIter):
            # some reshaping to put query, pos, negs in a single (N, 3, H, W) tensor
            # where N = batchSize * (nQuery + nPos + nNeg)
            if query is None:
                continue  # in case we get an empty batch

            B, C, H, W = query.shape
            nNeg = torch.sum(negCounts)
            input = torch.cat([query, positives, negatives])
            del query, positives, negatives
            image_encoding = rv.model.encoder(input.to(rv.device))
            del input
            if opt.withAttention:
                image_encoding = rv.model.attention(image_encoding)
                vlad_encoding = rv.model.pool(image_encoding)
                del image_encoding
            else:
                vlad_encoding = rv.model.pool(image_encoding)
                del image_encoding

            vladQ, vladP, vladN = torch.split(vlad_encoding, [B, B, nNeg])
            del vlad_encoding

            rv.optimizer.zero_grad()
            # calculate loss for each Query, Positive, Negative triplet
            # due to potential difference in number of negatives have to
            # do it per query, per negative
            loss = 0
            for i, negCount in enumerate(negCounts):
                for n in range(negCount):
                    negIx = (torch.sum(negCounts[:i]) + n).item()
                    loss += rv.criterion(vladQ[i:i + 1], vladP[i:i + 1], vladN[negIx:negIx + 1])
            del vladQ, vladP, vladN

            loss /= nNeg.float().to(rv.device)  # normalise by actual number of negatives
            loss.backward()
            rv.optimizer.step()

            batch_loss = loss.item()
            epoch_loss += batch_loss

            if iteration % 50 == 0 or nBatches <= 10:
                print("==> Epoch[{}]({}/{}): Loss: {:.4f}".format(epoch, iteration,
                                                                  nBatches, batch_loss), flush=True)
                writer.add_scalar('Train/Loss', batch_loss,
                                  ((epoch - 1) * nBatches) + iteration)
                writer.add_scalar('Train/nNeg', nNeg,
                                  ((epoch - 1) * nBatches) + iteration)
                print('Allocated:', torch.cuda.memory_allocated())
                print('Cached:', torch.cuda.memory_cached())

        startIter += len(training_data_loader)
        del training_data_loader, loss
        rv.optimizer.zero_grad()
        torch.cuda.empty_cache()
        remove(rv.train_set.cache)  # delete HDF5 cache

        elapsed = time.time() - start_time
        print('====> Training Time =', int(elapsed), "s")

    avg_loss = epoch_loss / nBatches

    print("===> Epoch {} Complete: Avg. Loss: {:.4f}".format(epoch, avg_loss),
          flush=True)
    writer.add_scalar('Train/AvgLoss', avg_loss, epoch)