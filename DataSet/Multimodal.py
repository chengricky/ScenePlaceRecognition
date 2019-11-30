import torchvision.transforms as transforms
import torch.utils.data as data

from os.path import join, exists
import os
from PIL import Image

from collections import namedtuple

root_dir = '/localresearch/PreciseLocalization/Dataset/VLdataset_MST/T3-Lib/'
if not exists(root_dir):
    raise FileNotFoundError('root_dir is hardcoded, please adjust to point to Yuquan Multimodal dataset')

# the list of database folder (images)
dbFolder = root_dir + 'reference'
qFolder = root_dir + 'query'

dbStruct = namedtuple('dbStruct', ['dataset', 'numDb', 'numQ'])

def input_transform():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


# DB + query
def get_whole_val_set(onlyDB=False):
    return DatasetFromStruct(dbFolder, qFolder, input_transform=input_transform(), onlyDB=onlyDB)


def get_whole_test_set(onlyDB=False):
    return DatasetFromStruct(dbFolder, qFolder, input_transform=input_transform(), onlyDB=onlyDB)


class DatasetFromStruct(data.Dataset):
    def __init__(self, dbFolder, qFolder, input_transform=None, onlyDB=False):
        super().__init__()
        self.dataset = 'pitts30k'
        self.input_transform = input_transform
# DATABASE 图像
        list_img = os.listdir(dbFolder)
        list_img.sort()
        # get rid of the dir
        list_img = list_img[:len(list_img)-1]
        list_rgb = [ele for ele in list_img if "color" in ele]
        list_ir = [ele for ele in list_img if "IR" in ele]
        self.rgb_images = [join(dbFolder, dbIm) for dbIm in list_rgb]
        self.ir_images = [join(dbFolder, dbIm) for dbIm in list_ir]
        numDb = len(self.rgb_images)
# QUERY 图像
        list_img = os.listdir(qFolder)
        list_img.sort()
        # get rid of the dir
        list_img = list_img[:len(list_img) - 1]
        list_rgb = [ele for ele in list_img if "color" in ele]
        list_ir = [ele for ele in list_img if "IR" in ele]
        self.rgb_images.extend([join(qFolder, qIm) for qIm in list_rgb])
        self.ir_images.extend([join(qFolder, qIm) for qIm in list_ir])
        numQ = len(self.rgb_images)-numDb

        self.dbStruct = dbStruct(self.dataset, numDb, numQ)

    def __getitem__(self, index):
        rgb_img = Image.open(self.rgb_images[index])
        ir_img = Image.open(self.ir_images[index]).convert(mode="RGB")
        if self.input_transform:
            rgb_img = self.input_transform(rgb_img)
            ir_img = self.input_transform(ir_img)

        return rgb_img, ir_img, index

    def __len__(self):
        return len(self.rgb_images)

