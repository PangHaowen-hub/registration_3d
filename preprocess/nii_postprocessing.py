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


def resample_padding(img_path, moved_path, save_path):
    sitk_img = sitk.ReadImage(img_path)
    sitk_moved = sitk.ReadImage(moved_path)
    array_moved = sitk.GetArrayFromImage(sitk_moved)

    resample = sitk.ResampleImageFilter()  # 设置一个Filter
    resample.SetInterpolator(sitk.sitkLinear)  # 设置插值方式
    resample.SetDefaultPixelValue(-1024)  # 默认像素值
    newspacing = sitk_img.GetSpacing()
    resample.SetOutputSpacing(newspacing)
    resample.SetOutputOrigin(sitk_moved.GetOrigin())
    resample.SetOutputDirection(sitk_moved.GetDirection())

    img_shape = sitk_moved.GetSize()
    img_spacing = sitk_moved.GetSpacing()

    new_size = np.asarray(img_shape) * np.asarray(img_spacing) / np.asarray(newspacing)
    new_size = new_size.astype(int).tolist()
    resample.SetSize(new_size)
    new = resample.Execute(sitk_moved)

    # new_arr = sitk.GetArrayFromImage(new)
    # new_arr = np.expand_dims(new_arr, axis=0)
    # pad_transform = tio.transforms.CropOrPad(sitk_img.GetSize(), padding_mode=-1024)  # TODO:修改Padding值
    # new_arr = pad_transform(new_arr)
    # new_arr = np.squeeze(new_arr, 0)
    #
    # new_img = sitk.GetImageFromArray(new_arr)
    # new_img.SetDirection(sitk_img.GetDirection())
    # new_img.SetOrigin(sitk_img.GetOrigin())
    # new_img.SetSpacing(sitk_img.GetSpacing())
    _, fullflname = os.path.split(img_path)
    sitk.WriteImage(new, os.path.join(save_path, fullflname))


# TODO:没写完
if __name__ == '__main__':
    img_path = r'G:\CT2CECT\registration\data\cect_a'
    moved_path = r'G:\CT2CECT\registration\data\voxelmorph\img'
    save_path = r'G:\CT2CECT\registration\data\voxelmorph\postprocessing'
    img_list = get_listdir(img_path)
    img_list.sort()
    moved_list = get_listdir(moved_path)
    moved_list.sort()

    for i in tqdm.trange(len(img_list)):
        resample_padding(img_list[i], moved_list[i], save_path)
