import SimpleITK as sitk
import numpy as np
import os
import tqdm


def get_listdir(path):  # 获取目录下所有gz格式文件的地址，返回地址list
    tmp_list = []
    for file in os.listdir(path):
        if os.path.splitext(file)[1] == '.gz':
            file_path = os.path.join(path, file)
            tmp_list.append(file_path)
    return tmp_list


def resample(img_path, save_path):
    mask_sitk_img = sitk.ReadImage(img_path)
    img_shape = mask_sitk_img.GetSize()
    img_spacing = mask_sitk_img.GetSpacing()
    resample = sitk.ResampleImageFilter()  # 设置一个Filter
    resample.SetInterpolator(sitk.sitkLinear)  # 设置插值方式
    resample.SetDefaultPixelValue(-1024)  # 默认像素值
    newspacing = [1.5, 1.5, 1.5]
    resample.SetOutputSpacing(newspacing)
    resample.SetOutputOrigin(mask_sitk_img.GetOrigin())
    resample.SetOutputDirection(mask_sitk_img.GetDirection())
    new_size = np.asarray(img_shape) * np.asarray(img_spacing) / np.asarray(newspacing)
    new_size = new_size.astype(int).tolist()
    resample.SetSize(new_size)
    new = resample.Execute(mask_sitk_img)
    _, fullflname = os.path.split(img_path)
    sitk.WriteImage(new, os.path.join(save_path, fullflname))


if __name__ == '__main__':
    img_path = r'G:\CT2CECT\registration\data\ncct'
    save_path = r'G:\CT2CECT\registration\data\ncct_preprocess_15'
    img_list = get_listdir(img_path)
    img_list.sort()
    for i in tqdm.tqdm(img_list):
        resample(i, save_path)
