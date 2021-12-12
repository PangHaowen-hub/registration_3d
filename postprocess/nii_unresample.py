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


def unresample(img_path, moved_path, save_path):
    sitk_img = sitk.ReadImage(img_path)
    sitk_moved = sitk.ReadImage(moved_path)

    resample = sitk.ResampleImageFilter()  # 设置一个Filter
    resample.SetInterpolator(sitk.sitkLinear)  # 设置插值方式
    resample.SetDefaultPixelValue(-1024)  # 默认像素值
    resample.SetOutputSpacing(sitk_img.GetSpacing())
    resample.SetOutputOrigin(sitk_moved.GetOrigin())
    resample.SetOutputDirection(sitk_moved.GetDirection())
    resample.SetSize(sitk_img.GetSize())
    new = resample.Execute(sitk_moved)
    _, fullflname = os.path.split(img_path)
    sitk.WriteImage(new, os.path.join(save_path, fullflname))


if __name__ == '__main__':
    img_path = r'G:\CT2CECT\registration\data\cect_a'
    moved_path = r'G:\CT2CECT\registration\data\voxelmorph\postprocessing_padding'
    save_path = r'G:\CT2CECT\registration\data\voxelmorph\postprocessing_resample'
    img_list = get_listdir(img_path)
    img_list.sort()
    moved_list = get_listdir(moved_path)
    moved_list.sort()

    for i in tqdm.trange(len(img_list)):
        unresample(img_list[i], moved_list[i], save_path)
