import os
import torch
import tqdm
import numpy as np
import voxelmorph as vxm
import SimpleITK as sitk


def get_listdir(path):
    tmp_list = []
    for file in os.listdir(path):
        if os.path.splitext(file)[1] == '.gz':
            file_path = os.path.join(path, file)
            tmp_list.append(file_path)
    return tmp_list


model_path = r'./models/0200.pt'

fixed_path = r'G:\CT2CECT\registration\data\cect_a_preprocess'
moving_path = r'G:\CT2CECT\registration\data\ncct_preprocess'
save_path = r'G:\CT2CECT\registration\data\voxelmorph'
fixed_list = get_listdir(fixed_path)
moving_list = get_listdir(moving_path)

device = 'cuda'
model = vxm.networks.VxmDense.load(model_path, device)
model.to(device)
model.eval()

for i in tqdm.trange(len(fixed_list)):
    fixed_img = sitk.ReadImage(fixed_list[i])
    fixed_arr = sitk.GetArrayFromImage(fixed_img)
    moving_img = sitk.ReadImage(moving_list[i])
    moving_arr = sitk.GetArrayFromImage(moving_img)
    fixed_arr = np.expand_dims(np.expand_dims(fixed_arr, 0), 0)
    moving_arr = np.expand_dims(np.expand_dims(moving_arr, 0), 0)

    input_moving = torch.from_numpy(moving_arr).to(device).float()
    input_fixed = torch.from_numpy(fixed_arr).to(device).float()

    # predict
    moved, warp = model(input_moving, input_fixed, registration=True)
    moved = moved.detach().cpu().numpy().squeeze()
    warp = warp.detach().cpu().numpy().squeeze()
    warp = np.transpose(warp, (1, 2, 3, 0))
    _, fullflname = os.path.split(moving_list[i])

    moved_img = sitk.GetImageFromArray(moved)
    moved_img.SetDirection(moving_img.GetDirection())
    moved_img.SetSpacing(moving_img.GetSpacing())
    moved_img.SetOrigin(moving_img.GetOrigin())
    sitk.WriteImage(moved_img, os.path.join(save_path, fullflname))

    warp_img = sitk.GetImageFromArray(warp)
    warp_img.SetDirection(moving_img.GetDirection())
    warp_img.SetSpacing(moving_img.GetSpacing())
    warp_img.SetOrigin(moving_img.GetOrigin())
    sitk.WriteImage(warp_img, os.path.join(save_path, 'warp_' + fullflname))
