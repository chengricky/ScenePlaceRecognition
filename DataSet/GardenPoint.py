"""
The dataset file of West Lake is only used for TEST, so the positive selection is not implemented
Ricky 2019.Dec.10
"""
import torchvision.transforms as transforms
import torch.utils.data as data
from os.path import join, exists
import os
import numpy as np
from PIL import Image

root_dir = '/localresearch/VisualLocalization/Dataset/dataset GardensPointWalking/'
if not exists(root_dir):
    raise FileNotFoundError('root_dir is hardcoded, please adjust to point to West Lake dataset')

# the list of database folder (images)
dbFolder = join(root_dir, 'day_right')
qFolder = join(root_dir, 'day_left')


def input_transform():
    return transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


def get_whole_val_set(onlyDB=False):
    """
    database + query
    """
    return DatasetFromStruct(dbFolder, qFolder, input_transform=input_transform(), onlyDB=onlyDB)


class DatasetFromStruct(data.Dataset):
    def __init__(self, dbFolder, qFolder, input_transform=None, onlyDB=False):
        super().__init__()

# DATABASE 图像
        self.input_transform = input_transform
        listImg = os.listdir(dbFolder)
        listImg.sort()
        self.images = []
        self.images.extend([join(dbFolder, dbIm) for dbIm in listImg])
        self.numDb = len(self.images)
# QUERY 图像
        if not onlyDB:
            listImg = os.listdir(qFolder)
            listImg.sort()
            self.images.extend([join(qFolder, qIm) for qIm in listImg])
        self.numQ = len(self.images)-self.numDb

        self.positive = None

    def __getitem__(self, index):
        img = Image.open(self.images[index])
        if self.input_transform:
            img = self.input_transform(img)

        return img, index

    def __len__(self):
        return len(self.images)

    # def get_positives(self):
    #     # positives for evaluation are those within trivial threshold range
    #     # fit NN to find them, search by radius
    #     if self.positives is None:
    #         self.positives = np.ndarray(())
    #         knn = NearestNeighbors(n_jobs=-1)
    #         knn.fit(self.dbStruct.utmDb)
    #
    #         self.distances, self.positives = knn.radius_neighbors(self.dbStruct.utmQ,
    #                                                               radius=self.dbStruct.posDistThr)
    #
    #     return self.positives


