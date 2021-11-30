from torch.utils.data import Dataset
import os
import SimpleITK as sitk
import torchio


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

        return fixed_arr, moving_arr

    def __len__(self):
        return len(self.fixed_list)


class test_dataset(Dataset):
    def __init__(self, fixed_path, moving_path):
        self.fixed_list = get_listdir(fixed_path)
        self.fixed_list.sort()
        self.moving_list = get_listdir(moving_path)
        self.moving_list.sort()
        self.subjects = []
        for fixed_path, moving_path in zip(self.fixed_list, self.moving_list):
            subject = torchio.Subject(
                fixed=torchio.ScalarImage(fixed_path),
                moving=torchio.ScalarImage(moving_path),
            )
            self.subjects.append(subject)

        self.test_set = torchio.SubjectsDataset(self.subjects)
