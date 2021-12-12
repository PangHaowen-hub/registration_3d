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


def unpadding(img_path, moved_path, save_path):
    sitk_img = sitk.ReadImage(img_path)
    sitk_moved = sitk.ReadImage(moved_path)
    new_arr = sitk.GetArrayFromImage(sitk_moved)
    new_arr = np.expand_dims(new_arr, axis=0)
    shape = sitk_img.GetSize()
    pad_transform = tio.transforms.CropOrPad((shape[2], shape[0], shape[1]), padding_mode=-1024)  # TODO:修改Padding值
    new_arr = pad_transform(new_arr)
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
        unpadding(img_list[i], moved_list[i], save_path)
