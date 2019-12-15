import torch
import torchvision.transforms as transforms
import torch.utils.data as data

from os.path import join, exists, dirname, abspath
from scipy.io import loadmat
import numpy as np
from random import randint, random
from collections import namedtuple
from PIL import Image

from sklearn.neighbors import NearestNeighbors
import h5py

from DataSet.data_augment import *
from torchvision.transforms import ColorJitter
import yaml
import faiss

# root_dir = '/home/ruiqi/netVLADdatasets/Pittsburgh/'
root_dir = '/data/Pittsburgh/'

if not exists(root_dir):
    raise FileNotFoundError('root_dir is hardcoded, please adjust to point to Pittsburth dataset')

struct_dir = join(root_dir, 'datasets/')
queries_dir = join(root_dir, 'queries_real')


def data_aug(img, configs):
    # 数据增强
    jitter = ColorJitter(configs['brightness'], configs['contrast'], configs['saturation'], configs['hue'])
    name2func = {
        # 'blur': lambda img_in: gaussian_blur(img_in, configs['blur_range']),
        # Randomly change the brightness, contrast and saturation of an image.
        'jitter': lambda img_in: jitter(img_in), # np.asarray(jitter(img_in)),
        # 'noise': lambda img_in: add_noise(img_in),
        'none': lambda img_in: img_in,
        # 'sp_gaussian_noise': lambda img_in: additive_gaussian_noise(img_in, configs['sp_gaussian_range']),
        # 'sp_speckle_noise': lambda img_in: additive_speckle_noise(img_in, configs['sp_speckle_prob_range']),
        # 'sp_additive_shade': lambda img_in: additive_shade(img_in, configs['sp_nb_ellipse'],
        #                                                    configs['sp_transparency_range'],
        #                                                    configs['sp_kernel_size_range']),
        'motion_blur': lambda img_in: motion_blur(img_in, configs['sp_max_kernel_size']),
        # 'resize_blur': lambda img_in: resize_blur(img_in, configs['resize_blur_min_ratio'])
        'random_rotate_img': lambda img_in: random_rotate_img(img_in, configs["min_angle"], configs["max_angle"])
    }

    # ['random_rotate_img','jitter','motion_blur','none']
    if configs['augment_num'] < 0:
        return img
    elif len(configs['augment_classes']) > configs['augment_num']:
        augment_classes = np.random.choice(configs['augment_classes'], configs['augment_num'],
                                           False, p=configs['augment_classes_weight'])
    elif len(configs['augment_classes']) <= configs['augment_num']:
        augment_classes = configs["augment_classes"]
    else:
        return img

    for ac in augment_classes:
        img = name2func[ac](img)
    return img


def input_transform():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def get_random_training_set(rand_num=10000):
    structFile = join(struct_dir, 'pitts30k_train.mat')
    return RandomDatasetFromStruct(structFile, input_transform=input_transform(), rand_num=rand_num)


def get_whole_training_set(onlyDB=False):
    structFile = join(struct_dir, 'pitts30k_train.mat')
    return WholeDatasetFromStruct(structFile, input_transform=input_transform(), onlyDB=onlyDB)


def get_250k_whole_training_set(onlyDB=False):
    structFile = join(struct_dir, 'pitts250k_train.mat')
    return WholeDatasetFromStruct(structFile, input_transform=input_transform(), onlyDB=onlyDB)


def get_whole_val_set():
    structFile = join(struct_dir, 'pitts30k_val.mat')
    return WholeDatasetFromStruct(structFile, input_transform=input_transform())


def get_250k_whole_val_set():
    structFile = join(struct_dir, 'pitts250k_val.mat')
    return WholeDatasetFromStruct(structFile, input_transform=input_transform())


def get_whole_test_set():
    structFile = join(struct_dir, 'pitts30k_test.mat')
    return WholeDatasetFromStruct(structFile, input_transform=input_transform())


def get_250k_whole_test_set():
    structFile = join(struct_dir, 'pitts250k_test.mat')
    return WholeDatasetFromStruct(structFile, input_transform=input_transform())


def get_training_query_set(margin=0.1):
    structFile = join(struct_dir, 'pitts30k_train.mat')
    return QueryDatasetFromStruct(structFile, input_transform=input_transform(), margin=margin)


def get_250k_training_query_set(margin=0.1):
    structFile = join(struct_dir, 'pitts250k_train.mat')
    return QueryDatasetFromStruct(structFile, input_transform=input_transform(), margin=margin)


def get_val_query_set():
    structFile = join(struct_dir, 'pitts30k_val.mat')
    return QueryDatasetFromStruct(structFile, input_transform=input_transform())


def get_250k_val_query_set():
    structFile = join(struct_dir, 'pitts250k_val.mat')
    return QueryDatasetFromStruct(structFile, input_transform=input_transform())


dbStruct = namedtuple('dbStruct', ['whichSet', 'dataset', 'dbImage', 'utmDb', 'qImage', 'utmQ',
                                   'numDb', 'numQ', 'posDistThr', 'posDistSqThr', 'nonTrivPosDistSqThr'])


def parse_dbStruct(path):
    mat = loadmat(path)
    matStruct = mat['dbStruct'].item()

    if '250k' in path.split('/')[-1]:
        dataset = 'pitts250k'
    else:
        dataset = 'pitts30k'

    whichSet = matStruct[0].item()

    dbImage = [f[0].item() for f in matStruct[1]]
    utmDb = matStruct[2].T

    qImage = [f[0].item() for f in matStruct[3]]
    utmQ = matStruct[4].T

    numDb = matStruct[5].item()
    numQ = matStruct[6].item()

    posDistThr = matStruct[7].item()
    posDistSqThr = matStruct[8].item()
    nonTrivPosDistSqThr = matStruct[9].item()

    return dbStruct(whichSet, dataset, dbImage, utmDb, qImage,
                    utmQ, numDb, numQ, posDistThr,
                    posDistSqThr, nonTrivPosDistSqThr)


class RandomDatasetFromStruct(data.Dataset):
    def __init__(self, structFile, input_transform=None, rand_num=10000):
        super().__init__()

        self.input_transform = input_transform

        self.dbStruct = parse_dbStruct(structFile)
        self.images = [join(root_dir, dbIm) for dbIm in self.dbStruct.dbImage]
        self.images += [join(queries_dir, qIm) for qIm in self.dbStruct.qImage]
        num = self.dbStruct.numDb + self.dbStruct.numQ
        idx = np.random.random_integers(0, num-1, rand_num)
        self.images = np.array(self.images)[idx]

        self.whichSet = self.dbStruct.whichSet
        self.dataset = self.dbStruct.dataset

    def __getitem__(self, index):
        img = Image.open(self.images[index])

        if self.input_transform:
            img = self.input_transform(img)

        return img, index

    def __len__(self):
        return len(self.images)


class WholeDatasetFromStruct(data.Dataset):
    def __init__(self, structFile, input_transform=None, onlyDB=False):
        super().__init__()

        self.input_transform = input_transform
        self.is_train = "train.mat" in structFile

        self.dbStruct = parse_dbStruct(structFile)
        self.images = [join(root_dir, dbIm) for dbIm in self.dbStruct.dbImage]
        if not onlyDB:
            self.images += [join(queries_dir, qIm) for qIm in self.dbStruct.qImage]

        self.whichSet = self.dbStruct.whichSet
        self.dataset = self.dbStruct.dataset

        self.positives = None
        self.distances = None

        self.augment_config = None
        with open(join(dirname(dirname(abspath(__file__))), 'DataSet/data_augment.yaml'), 'r') as f:
            self.augment_config = yaml.load(f)

    def __getitem__(self, index):
        img = Image.open(self.images[index])

        if self.is_train:
            img = data_aug(img, self.augment_config)

        if self.input_transform:
            img = self.input_transform(img)

        return img, index

    def __len__(self):
        return len(self.images)

    def get_positives(self):
        # positives for evaluation are those within trivial threshold range
        # fit NN to find them, search by radius
        if self.positives is None:
            knn = NearestNeighbors(n_jobs=-1)
            knn.fit(self.dbStruct.utmDb)

            self.distances, self.positives = knn.radius_neighbors(self.dbStruct.utmQ,
                                                                  radius=self.dbStruct.posDistThr)

        return self.positives


def collate_fn(batch):
    """Creates mini-batch tensors from the list of tuples (query, positive, negatives).
    
    Args:
        data: list of tuple (query, positive, negatives). 
            - query: torch tensor of shape (3, h, w).
            - positive: torch tensor of shape (3, h, w).
            - negative: torch tensor of shape (n, 3, h, w).
    Returns:
        query: torch tensor of shape (batch_size, 3, h, w).
        positive: torch tensor of shape (batch_size, 3, h, w).
        negatives: torch tensor of shape (batch_size, n, 3, h, w).
    """

    batch = list(filter (lambda x:x is not None, batch))
    if len(batch) == 0: return None, None, None, None, None

    query, positive, negatives, indices = zip(*batch)

    query = data.dataloader.default_collate(query)
    positive = data.dataloader.default_collate(positive)
    negCounts = data.dataloader.default_collate([x.shape[0] for x in negatives])
    negatives = torch.cat(negatives, 0)
    import itertools
    indices = list(itertools.chain(*indices))

    return query, positive, negatives, negCounts, indices


class QueryDatasetFromStruct(data.Dataset):
    def __init__(self, structFile, nNegSample=1000, nNeg=10, margin=0.1, input_transform=None):
        super().__init__()

        self.is_train = "train.mat" in structFile
        self.input_transform = input_transform
        self.margin = margin

        self.dbStruct = parse_dbStruct(structFile)
        self.whichSet = self.dbStruct.whichSet
        self.dataset = self.dbStruct.dataset
        # number of negatives to randomly sample
        self.nNegSample = nNegSample
        # number of negatives used for training
        self.nNeg = nNeg

        # use a single GPU
        # self.res = faiss.StandardGpuResources()

        # potential positives are those within nontrivial threshold range
        # fit NN to find them, search by radius
        knn = NearestNeighbors(n_jobs=-1)
        knn.fit(self.dbStruct.utmDb)
        self.nontrivial_positives = knn.radius_neighbors(self.dbStruct.utmQ,
                                                         radius=self.dbStruct.nonTrivPosDistSqThr**0.5,
                                                         return_distance=False)

        # radius returns unsorted, sort once now so we dont have to later
        for i, posi in enumerate(self.nontrivial_positives):
            self.nontrivial_positives[i] = np.sort(posi)
        # its possible some queries don't have any non trivial potential positives
        # lets filter those out
        self.queries = np.where(np.array([len(x) for x in self.nontrivial_positives])>0)[0]

        # potential negatives are those outside of posDistThr range
        potential_positives = knn.radius_neighbors(self.dbStruct.utmQ,
                                                   radius=self.dbStruct.posDistThr,
                                                   return_distance=False)

        self.potential_negatives = []
        for pos in potential_positives:
            self.potential_negatives.append(np.setdiff1d(np.arange(self.dbStruct.numDb),
                pos, assume_unique=True))

        self.cache = None # filepath of HDF5 containing feature vectors for images

        self.negCache = [np.empty((0,)) for _ in range(self.dbStruct.numQ)]

        self.augment_config = None
        with open(join(dirname(dirname(abspath(__file__))), 'DataSet/data_augment.yaml'), 'r') as f:
            self.augment_config = yaml.load(f)

        self.pool_size = 0
        # self.gpu_index_flat = None
        self.index_flat = None

    def __getitem__(self, index):
        index = self.queries[index] # re-map index to match dataset
        with h5py.File(self.cache, mode='r') as h5: 
            h5feat = h5.get("features")

            qOffset = self.dbStruct.numDb 
            qFeat = h5feat[index+qOffset]

            posFeat = h5feat[self.nontrivial_positives[index].tolist()]

            if self.pool_size == 0:
                # netVLAD dimension
                pool_size = posFeat.shape[1]
                # build a flat (CPU) index
                self.index_flat = faiss.IndexFlatL2(pool_size)
                # make it into a gpu index
                # self.gpu_index_flat = faiss.index_cpu_to_gpu(self.res, 0, index_flat)
            else:
                self.index_flat.reset()
            # add vectors to the index
            self.index_flat.add(posFeat)
            # search for the nearest +ive
            dPos, posNN = self.index_flat.search(qFeat.reshape(1, -1).astype('float32'), 1)

            # knn = NearestNeighbors(n_jobs=-1)
            # knn.fit(posFeat)
            # dPos, posNN = knn.kneighbors(qFeat.reshape(1,-1), 1)
            dPos = dPos.item()
            posIndex = self.nontrivial_positives[index][posNN[0]].item()

            negSample = np.random.choice(self.potential_negatives[index], self.nNegSample)
            negSample = np.unique(np.concatenate([self.negCache[index], negSample]))

            negFeat = h5feat[negSample.tolist()]
            self.index_flat.reset()
            self.index_flat.add(negFeat)
            # to quote netVLAD paper code: 10x is hacky but fine
            dNeg, negNN = self.index_flat.search(qFeat.reshape(1, -1).astype('float32'), self.nNeg*10)


            # knn.fit(negFeat)
            # dNeg, negNN = knn.kneighbors(qFeat.reshape(1,-1),
            #         self.nNeg*10) # to quote netvlad paper code: 10x is hacky but fine
            dNeg = dNeg.reshape(-1)
            negNN = negNN.reshape(-1)

            # try to find negatives that are within margin, if there aren't any return none
            violatingNeg = dNeg < dPos + self.margin**0.5
     
            if np.sum(violatingNeg) < 1:
                #if none are violating then skip this query
                return None

            negNN = negNN[violatingNeg][:self.nNeg]
            negIndices = negSample[negNN].astype(np.int32)
            self.negCache[index] = negIndices

        query = Image.open(join(queries_dir, self.dbStruct.qImage[index]))
        positive = Image.open(join(root_dir, self.dbStruct.dbImage[posIndex]))

        if self.is_train:
            query = data_aug(query, self.augment_config)
            positive = data_aug(positive, self.augment_config)

        if self.input_transform:
            query = self.input_transform(query)
            positive = self.input_transform(positive)

        negatives = []
        for negIndex in negIndices:
            negative = Image.open(join(root_dir, self.dbStruct.dbImage[negIndex]))
            if self.is_train:
                negative = data_aug(negative, self.augment_config)
            if self.input_transform:
                negative = self.input_transform(negative)
            negatives.append(negative)

        negatives = torch.stack(negatives, 0)

        return query, positive, negatives, [index, posIndex]+negIndices.tolist()

    def __len__(self):
        return len(self.queries)
