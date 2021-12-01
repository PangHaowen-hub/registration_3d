import torch
import tqdm
from torch.utils.data import DataLoader
import numpy as np
import torch.utils.data as data
import os
from torchio.data import UniformSampler
from torchio.transforms import ZNormalization, CropOrPad, Compose, Resample, Resize
import torchio


def get_listdir(path):
    tmp_list = []
    for file in os.listdir(path):
        if os.path.splitext(file)[1] == '.gz':
            file_path = os.path.join(path, file)
            tmp_list.append(file_path)
    return tmp_list


class dataset(data.Dataset):
    def __init__(self, imgs_path):
        self.img_list = get_listdir(imgs_path)
        self.img_list.sort()
        self.subjects = []
        for image_path in self.img_list:
            subject = torchio.Subject(
                source=torchio.ScalarImage(image_path)
            )
            self.subjects.append(subject)

        self.test_set = torchio.SubjectsDataset(self.subjects)

    def get_shape(self, i):
        return self.subjects[i].shape


if __name__ == '__main__':
    batch_size = 1
    data_path = r'G:\CT2CECT\registration\data\cect_a_preprocess'
    save_path = r'G:\CT2CECT\registration\data\cect_a_preprocess_patch'
    dataset = dataset(data_path)
    patch_overlap = 128, 128, 128
    patch_size = 256

    for subj in tqdm.tqdm(dataset.test_set):
        grid_sampler = torchio.inference.GridSampler(subj, patch_size, patch_overlap)
        patch_loader = torch.utils.data.DataLoader(grid_sampler, batch_size)
        aggregator = torchio.inference.GridAggregator(grid_sampler, 'average')

        for j, patches_batch in enumerate(patch_loader):
            input_tensor = patches_batch['source'][torchio.DATA].float()
            locations = patches_batch[torchio.LOCATION]  # patch的位置信息
            _, fullflname = os.path.split(subj['source']['path'])
            affine = subj['source']['affine']
            output_arr = np.squeeze(input_tensor.numpy(), 0)
            output_image = torchio.ScalarImage(tensor=output_arr, affine=affine)
            temp = os.path.join(save_path, fullflname[:3] + str(j).rjust(3, '0') + '.nii.gz')
            output_image.save(temp)
