import torch
from torch.utils.data.dataset import Subset
from math import ceil
import numpy as np
from os.path import join
import h5py


def train(model, opt, encoder_dim, epoch, train_set):
    epoch_loss = 0
    startIter = 1  # keep track of batch iter across subsets for logging

    if opt.cacheRefreshRate > 0:
        subsetN = ceil(len(train_set) / opt.cacheRefreshRate)
        # TODO randomise the arange before splitting?
        subsetIdx = np.array_split(np.arange(len(train_set)), subsetN)
    else:
        subsetN = 1
        subsetIdx = [np.arange(len(train_set))]

    nBatches = (len(train_set) + opt.batchSize - 1) // opt.batchSize

    for subIter in range(subsetN):
        print('====> Building Cache, subIter =', subIter)
        model.eval()
        train_set.cache = join(opt.cachePath, train_set.whichSet + '_feat_cache.hdf5')
        with h5py.File(train_set.cache, mode='w') as h5:
            pool_size = encoder_dim
            if opt.pooling.lower() == 'netvlad':
                # if not opt.reduction:
                pool_size *= opt.num_clusters
                # else:
                #     pool_size = 4096
            h5feat = h5.create_dataset("features",
                                       [len(whole_train_set), pool_size],
                                       dtype=np.float32)
            with torch.no_grad():
                for iteration, (input, indices) in enumerate(whole_training_data_loader, 1):
                    input = input.to(device)
                    image_encoding = model.encoder(input)
                    if opt.withAttention:
                        image_encoding = model.attention(image_encoding)
                        vlad_encoding = model.pool(image_encoding)
                        del image_encoding
                    else:
                        vlad_encoding = model.pool(image_encoding)
                        del image_encoding
                    h5feat[indices.detach().numpy(), :] = vlad_encoding.detach().cpu().numpy()
                    del input, vlad_encoding

        sub_train_set = Subset(dataset=train_set, indices=subsetIdx[subIter])

        training_data_loader = DataLoader(dataset=sub_train_set, num_workers=opt.threads,
                                          batch_size=opt.batchSize, shuffle=True,
                                          collate_fn=dataset.collate_fn, pin_memory=True)

        print('Allocated:', torch.cuda.memory_allocated())
        print('Cached:', torch.cuda.memory_cached())

        model.train()
        for iteration, (query, positives, negatives,
                        negCounts, indices) in enumerate(training_data_loader, startIter):
            # print('Start Iteration', iteration)
            # some reshaping to put query, pos, negs in a single (N, 3, H, W) tensor
            # where N = batchSize * (nQuery + nPos + nNeg)
            if query is None:
                # print('EMPTY QUERY', iteration)
                continue  # in case we get an empty batch

            B, C, H, W = query.shape
            nNeg = torch.sum(negCounts)
            input = torch.cat([query, positives, negatives])

            input = input.to(device)
            image_encoding = model.encoder(input)
            if opt.withAttention:
                image_encoding = model.attention(image_encoding)
                vlad_encoding = model.pool(image_encoding)
                del image_encoding
            else:
                vlad_encoding = model.pool(image_encoding)
                del image_encoding

            vladQ, vladP, vladN = torch.split(vlad_encoding, [B, B, nNeg])

            optimizer.zero_grad()

            # calculate loss for each Query, Positive, Negative triplet
            # due to potential difference in number of negatives have to
            # do it per query, per negative
            loss = 0
            for i, negCount in enumerate(negCounts):
                for n in range(negCount):
                    negIx = (torch.sum(negCounts[:i]) + n).item()
                    loss += criterion(vladQ[i:i + 1], vladP[i:i + 1], vladN[negIx:negIx + 1])

            loss /= nNeg.float().to(device)  # normalise by actual number of negatives
            loss.backward()
            optimizer.step()
            del input, vlad_encoding, vladQ, vladP, vladN
            del query, positives, negatives

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
        optimizer.zero_grad()
        torch.cuda.empty_cache()
        remove(train_set.cache)  # delete HDF5 cache

    avg_loss = epoch_loss / nBatches

    print("===> Epoch {} Complete: Avg. Loss: {:.4f}".format(epoch, avg_loss),
          flush=True)
    writer.add_scalar('Train/AvgLoss', avg_loss, epoch)