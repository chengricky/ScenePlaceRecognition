import torch
import torchvision.transforms as transforms
import torch.utils.data as data

from os.path import join, exists
import os
from scipy.io import loadmat
import numpy as np
from random import randint, random
from collections import namedtuple
from PIL import Image
import cv2

from sklearn.neighbors import NearestNeighbors
import h5py

root_dir = '/data/SenseTimeLocalizationData_GB/1-8_Alldata/3F/Frames/'
if not exists(root_dir):
    raise FileNotFoundError('root_dir is hardcoded, please adjust to point to GB dataset')

# the list of database folder (images)
dbFolders = ["3071", "3011", "3012", "3013", "3014", "3021", "3022", "3031", "3032",
              "3033", "3034", "3035", "3036", "3041", "3042", "3043", "3044", "3051",
              "3052", "3061", "3062", "3072", "3081", "3082", "3091", "3092", "3101",
              "3102", "3511", "3512", "3513", "3514", "3531", "3532", "3533", "3534",
              "3535", "3536", "3811", "3812", "3813", "3814", "3815", "3816", "3821",
              "3822", "3911", "3912", "3913", "3914", "3915", "3916"]
# dbFolders = ["3011", "3012", "3021", "3022","3031", "3032", "3041", "3042", "3051", "3052",
#              "3061", "3062", "3071", "3072", "3081", "3082", "3091", "3092", "3101", "3102"]
#dbFolders = ["3011", "3012", "3013", "3014"]
# dbFolders = ["3071", "3011", "3012", "3013", "3014", "3021", "3022", "3031", "3032",
#               "3033", "3034", "3035", "3036", "3041", "3042", "3043", "3044", "3051",
#               "3052", "3061", "3062", "3072", "3081", "3082", "3091", "3092", "3101",
#               "3102"]
#qFolders = "/home/SENSETIME/chengruiqi/PlaceRecognition/SenseLocalizationRicky/config/locate_test_config_no34_s.yaml"
#qFolders = "/home/SENSETIME/chengruiqi/PlaceRecognition/SenseLocalizationRicky/config/locate_test_config_301_s.yaml"
qFolders = "/home/SENSETIME/chengruiqi/PlaceRecognition/SenseLocalizationRicky/config/locate_gbtest_3f_config.yaml"


def input_transform():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])



# DB is too big to load in the memory
def get_query_set(onlyDB=False):
    return QueryDatasetFromStruct(dbFolders, qFolders,
                                  input_transform=input_transform(),
                                  onlyDB=onlyDB)

def get_sub_set(numDb, onlyDB=False):
    return SubDatasetFromStruct(dbFolders[numDb], qFolders,
                                  input_transform=input_transform(),
                                  onlyDB=onlyDB)

class SubDatasetFromStruct(data.Dataset):
    def __init__(self, dbFolder, qFolders, input_transform=None, onlyDB=False):
        super().__init__()

        self.input_transform = input_transform
        listImg = os.listdir(root_dir + dbFolder)
        listImg.sort()
        listImgs = listImg[:len(listImg)-1:] #get rid of the dir
        #listImgs = listImg[::3] # set step=3
        self.images = []
        self.images.extend([join(root_dir+dbFolder, dbIm) for dbIm in listImgs])
        self.numDb = len(self.images)

        if not onlyDB:
            fsSetting = cv2.FileStorage(qFolders, cv2.FileStorage_READ)
            fs = fsSetting.getNode('origin')
            for idx in range(0, fs.size()):
                imgNum = int(fs.at(idx).getNode('imNum').real())
                path = fs.at(idx).getNode('path').string()
                step = (imgNum+1)//10
                for i in range(1, imgNum+1, step):
                    imgName = '%06d.jpg' % i
                    self.images.append(path+imgName)
        self.numQ = len(self.images) - self.numDb

    def __getitem__(self, index):
        img = Image.open(self.images[index])
        #img = img.resize((640,360))
        img = img.resize((224, 224))

        if self.input_transform:
            img = self.input_transform(img)

        return img, index

    def __len__(self):
        return len(self.images)

class QueryDatasetFromStruct(data.Dataset):
    def __init__(self, dbFolders, qFolders, input_transform=None, onlyDB=False):
        super().__init__()

        self.input_transform = input_transform

        self.images = []

        if not onlyDB:
            fsSetting = cv2.FileStorage(qFolders, cv2.FileStorage_READ)
            fs = fsSetting.getNode('origin')
            for idx in range(0, fs.size()):
                imgNum = int(fs.at(idx).getNode('imNum').real())
                path = fs.at(idx).getNode('path').string()
                step = (imgNum+1)//10
                for i in range(1, imgNum, step)[0:10]:
                    imgName = '%06d.jpg' % i
                    self.images.append(path+imgName)

        self.numQ = len(self.images)

    def __getitem__(self, index):
        img = Image.open(self.images[index])
        #img = img.resize((640, 360))
        img = img.resize((224, 224))

        if self.input_transform:
            img = self.input_transform(img)


        return img, index

    def __len__(self):
        return len(self.images)

