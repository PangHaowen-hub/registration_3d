import os, glob
import torch, sys
from torch.utils.data import Dataset
from .data_utils import pkload
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
import os
import SimpleITK as sitk
import torchio
import numpy as np


class JHUBrainDataset(Dataset):
    def __init__(self, data_path, transforms):
        self.paths = data_path
        self.transforms = transforms

    def one_hot(self, img, C):
        out = np.zeros((C, img.shape[1], img.shape[2], img.shape[3]))
        for i in range(C):
            out[i, ...] = img == i
        return out

    def __getitem__(self, index):
        path = self.paths[index]
        x, y = pkload(path)
        # print(x.shape)
        # print(x.shape)
        # print(np.unique(y))
        # print(x.shape, y.shape)#(240, 240, 155) (240, 240, 155)
        # transforms work with nhwtc
        x, y = x[None, ...], y[None, ...]
        # print(x.shape, y.shape)#(1, 240, 240, 155) (1, 240, 240, 155)
        x, y = self.transforms([x, y])
        # y = self.one_hot(y, 2)
        # print(y.shape)
        # sys.exit(0)
        x = np.ascontiguousarray(x)  # [Bsize,channelsHeight,,Width,Depth]
        y = np.ascontiguousarray(y)
        # plt.figure()
        # plt.subplot(1, 2, 1)
        # plt.imshow(x[0, :, :, 8], cmap='gray')
        # plt.subplot(1, 2, 2)
        # plt.imshow(y[0, :, :, 8], cmap='gray')
        # plt.show()
        # sys.exit(0)
        # y = np.squeeze(y, axis=0)
        x, y = torch.from_numpy(x), torch.from_numpy(y)
        return x, y

    def __len__(self):
        return len(self.paths)


class JHUBrainInferDataset(Dataset):
    def __init__(self, data_path, transforms):
        self.paths = data_path
        self.transforms = transforms

    def one_hot(self, img, C):
        out = np.zeros((C, img.shape[1], img.shape[2], img.shape[3]))
        for i in range(C):
            out[i, ...] = img == i
        return out

    def __getitem__(self, index):
        path = self.paths[index]
        x, y, x_seg, y_seg = pkload(path)
        # print(x.shape)
        # print(x.shape)
        # print(np.unique(y))
        # print(x.shape, y.shape)#(240, 240, 155) (240, 240, 155)
        # transforms work with nhwtc
        x, y = x[None, ...], y[None, ...]
        x_seg, y_seg = x_seg[None, ...], y_seg[None, ...]
        # print(x.shape, y.shape)#(1, 240, 240, 155) (1, 240, 240, 155)
        x, x_seg = self.transforms([x, x_seg])
        y, y_seg = self.transforms([y, y_seg])
        # y = self.one_hot(y, 2)
        # print(y.shape)
        # sys.exit(0)
        x = np.ascontiguousarray(x)  # [Bsize,channelsHeight,,Width,Depth]
        y = np.ascontiguousarray(y)
        x_seg = np.ascontiguousarray(x_seg)  # [Bsize,channelsHeight,,Width,Depth]
        y_seg = np.ascontiguousarray(y_seg)
        # plt.figure()
        # plt.subplot(1, 2, 1)
        # plt.imshow(x[0, :, :, 8], cmap='gray')
        # plt.subplot(1, 2, 2)
        # plt.imshow(y[0, :, :, 8], cmap='gray')
        # plt.show()
        # sys.exit(0)
        # y = np.squeeze(y, axis=0)
        x, y, x_seg, y_seg = torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(x_seg), torch.from_numpy(y_seg)
        return x, y, x_seg, y_seg

    def __len__(self):
        return len(self.paths)


def get_listdir(path):
    tmp_list = []
    for file in os.listdir(path):
        if os.path.splitext(file)[1] == '.gz':
            file_path = os.path.join(path, file)
            tmp_list.append(file_path)
    return tmp_list


class MyDataset(Dataset):
    def __init__(self, fixed_path, moving_path):
        self.fixed_list = get_listdir(fixed_path)
        self.fixed_list.sort()
        self.moving_list = get_listdir(moving_path)
        self.moving_list.sort()

    def __getitem__(self, idx):
        fixed_name = self.fixed_list[idx]
        moving_name = self.moving_list[idx]
        fixed_img = sitk.ReadImage(fixed_name)
        fixed_arr = sitk.GetArrayFromImage(fixed_img)
        moving_img = sitk.ReadImage(moving_name)
        moving_arr = sitk.GetArrayFromImage(moving_img)
        fixed_arr, moving_arr = torch.from_numpy(fixed_arr), torch.from_numpy(moving_arr)

        return fixed_arr.unsqueeze(0), moving_arr.unsqueeze(0)

    def __len__(self):
        return len(self.fixed_list)
