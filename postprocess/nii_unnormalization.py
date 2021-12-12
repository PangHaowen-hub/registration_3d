import SimpleITK as sitk
import numpy as np
import os
import tqdm
import torchio as tio


def get_listdir(path):  # 获取目录下所有gz格式文件的地址，返回地址list
    tmp_list = []
    for file in os.listdir(path):
        if os.path.splitext(file)[1] == '.gz':
            file_path = os.path.join(path, file)
            tmp_list.append(file_path)
    return tmp_list


def unnormalization(img_path, moved_path, save_path):
    sitk_img = sitk.ReadImage(img_path)
    img_arr = sitk.GetArrayFromImage(sitk_img)

    sitk_moved = sitk.ReadImage(moved_path)
    moved_arr = sitk.GetArrayFromImage(sitk_moved)

    moved_arr = np.expand_dims(moved_arr, axis=0)
    normalization_transform = tio.transforms.RescaleIntensity(out_min_max=(np.min(img_arr), np.max(img_arr)))
    new_arr = normalization_transform(moved_arr)
    new_arr = np.squeeze(new_arr, 0)
    new_img = sitk.GetImageFromArray(new_arr)
    new_img.SetDirection(sitk_moved.GetDirection())
    new_img.SetOrigin(sitk_moved.GetOrigin())
    new_img.SetSpacing(sitk_moved.GetSpacing())
    _, fullflname = os.path.split(img_path)
    sitk.WriteImage(new_img, os.path.join(save_path, fullflname))


if __name__ == '__main__':
    img_path = r'G:\CT2CECT\registration\data\cect_a_resample'
    moved_path = r'G:\CT2CECT\registration\data\voxelmorph\postprocessing_unnorm'
    save_path = r'G:\CT2CECT\registration\data\voxelmorph\postprocessing_padding'
    img_list = get_listdir(img_path)
    img_list.sort()
    moved_list = get_listdir(moved_path)
    moved_list.sort()

    for i in tqdm.trange(len(img_list)):
        unnormalization(img_list[i], moved_list[i], save_path)

