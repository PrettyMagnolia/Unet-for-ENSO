import math
import os
import numpy as np
from sklearn import preprocessing
import netCDF4 as nc
from netCDF4 import Dataset
from net2_utils import *
from torchvision import transforms
from torch.utils.data import DataLoader, random_split
import torch
# my code
import torch.nn.functional as F

transform = transforms.Compose([
    transforms.ToTensor()
])


def logTransform(c, img):
    # 3通道RGB
    h, w, d = img.shape[0], img.shape[1], img.shape[2]
    new_img = np.zeros((h, w, d))
    for i in range(h):
        for j in range(w):
            for k in range(d):
                new_img[i, j, k] = c * (math.log(1.0 + img[i, j, k]))
    return new_img


# 获取（384,420）数据
class Unet_Dataset:
    def __init__(self, path):
        self.path = path
        # self.name1 = os.listdir(os.path.join(path, 'ASRMEdata'))
        # self.name2 = os.listdir(os.path.join(path, 'Modeldata'))
        # self.name3 = os.listdir(os.path.join(path, 'obs_data'))

        # 读取image第一个图层
        image_1 = np.load('./dataset/input_observe.npy')
        self.image_1 = np.nan_to_num(image_1, nan=0)
        # 读取image第二个图层
        image_2 = np.load('./dataset/input_pred.npy')
        self.image_2 = np.nan_to_num(image_2, nan=0)
        # 读取mask
        mask = np.load('./dataset/output.npy')
        self.mask = np.nan_to_num(mask, nan=0)

    def __len__(self):
        # return len(self.name1)
        return self.image_1.shape[0]

    def __getitem__(self, index):
        # 标签数据
        # name1 = self.name1[index]
        # name2 = self.name2[index]
        # name3 = self.name3[index]
        # asrmedata_path = os.path.join(self.path, 'ASRMEdata', name1)  # 拼接地址
        # Modeldata_path = os.path.join(self.path, 'Modeldata', name2)  # 源数据地址
        # obsdata_path = os.path.join(self.path, 'obs_data', name3)  # 源数据地址
        #
        # Modeldata = Dataset(Modeldata_path)
        # mod_data = Normalization(Modeldata['SIC'][:])
        # ASRMEdata = Dataset(asrmedata_path)
        # SICdata = Normalization(ASRMEdata['var'][:])
        # obsdata = Dataset(obsdata_path)
        # obs_data = Normalization(obsdata['var'][:])

        img1 = self.image_1[index]
        img2 = self.image_2[index]
        msk = self.mask[index]

        img1_ = F.interpolate(torch.from_numpy(img1).unsqueeze(0).unsqueeze(0), scale_factor=0.25, mode='nearest')
        img2_ = F.interpolate(torch.from_numpy(img2).unsqueeze(0).unsqueeze(0), scale_factor=0.25, mode='nearest')
        msk_ = F.interpolate(torch.from_numpy(msk).unsqueeze(0).unsqueeze(0), scale_factor=0.25, mode='nearest')

        # return transform(img1), transform(img2), transform(msk)
        return img1_.squeeze(0), img2_.squeeze(0), msk_.squeeze(0)


def Normalization(X):
    min_max_scaler = preprocessing.MinMaxScaler()
    X_minMax = min_max_scaler.fit_transform(X)
    return X_minMax


def load_data(train_data_path, batch_size):
    train_set = Unet_Dataset(train_data_path)
    # 验证集分配
    train_loader, val_loader = val_set_alloc(train_set, batch_size)
    return train_loader, val_loader


# 验证集分配
def val_set_alloc(dataset, batch_size):
    n_val = int(len(dataset) * 0.2)
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0))
    train_loader = DataLoader(train_set, batch_size, shuffle=False)
    val_loader = DataLoader(val_set, batch_size, shuffle=False)

    return train_loader, val_loader
