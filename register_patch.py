import numpy as np
import voxelmorph as vxm
import SimpleITK as sitk
from my_dataset import MyDataset, test_dataset
import torch
from torch.utils.data import DataLoader
import os
import tqdm
import torchio

if __name__ == '__main__':

    model_path = r'./models/0060.pt'
    fixed_path = r'G:\CT2CECT\registration\fixed_resample_norm'
    moving_path = r'G:\CT2CECT\registration\moving_a_resample_norm'
    dataset = test_dataset(fixed_path, moving_path)
    save_path = r'G:\CT2CECT\registration\moved_a'

    device = 'cuda'
    model = vxm.networks.VxmDense.load(model_path, device)
    model.to(device)
    model.eval()
    batch_size = 1
    patch_overlap = 128, 128, 128
    patch_size = 256

    for i, subj in enumerate(dataset.test_set):
        grid_sampler_fixed = torchio.inference.GridSampler(subj, patch_size, patch_overlap)  # 从图像中提取patch
        patch_loader_fixed = torch.utils.data.DataLoader(grid_sampler_fixed, batch_size)
        aggregator_fixed = torchio.inference.GridAggregator(grid_sampler_fixed, 'average')  # 用于聚合patch推理结果

        # grid_sampler_moving = torchio.inference.GridSampler(subj['moving'], patch_size, patch_overlap)  # 从图像中提取patch
        # patch_loader_moving = torch.utils.data.DataLoader(grid_sampler_moving, batch_size)
        # aggregator_moving = torchio.inference.GridAggregator(grid_sampler_moving, 'average')  # 用于聚合patch推理结果
        with torch.no_grad():
            for patches in tqdm.tqdm(patch_loader_fixed):
                fixed_tensor = patches['fixed'][torchio.DATA].to(device).float()
                moving_tensor = patches['moving'][torchio.DATA].to(device).float()

                moved, warp = model(fixed_tensor, moving_tensor, registration=True)

                locations = patches[torchio.LOCATION]  # patch的位置信息
                aggregator_fixed.add_batch(moved, locations)

        output_tensor = aggregator_fixed.get_output_tensor()  # 获取聚合后volume
        affine = subj['fixed']['affine']
        output_image = torchio.ScalarImage(tensor=output_tensor.numpy(), affine=affine)
        _, fullflname = os.path.split(subj['fixed']['path'])
        output_image.save(os.path.join(save_path, fullflname))

        # new_mask = sitk.GetImageFromArray(output_image.data.argmax(dim=0).int().permute(2, 1, 0))
        # new_mask.SetSpacing(output_image.spacing)
        # new_mask.SetDirection(output_image.direction)
        # new_mask.SetOrigin(output_image.origin)
        # sitk.WriteImage(new_mask, os.path.join(save_path, fullflname))
