import glob
import os, losses, utils
from torch.utils.data import DataLoader
# from data import datasets, trans
import numpy as np
import torch
from torchvision import transforms
from natsort import natsorted
from models import VxmDense_1, VxmDense_2, VxmDense_huge, VxmDense_l
import SimpleITK as sitk
import os
import torch
import tqdm


def get_listdir(path):
    tmp_list = []
    for file in os.listdir(path):
        if os.path.splitext(file)[1] == '.gz':
            file_path = os.path.join(path, file)
            tmp_list.append(file_path)
    return tmp_list


def main():
    model_idx = -1
    img_size = (256, 256, 256)
    weights = [1, 0.02]
    model_folder = 'vxm_2_mse_{}_diffusion_{}/'.format(weights[0], weights[1])
    model_dir = 'experiments/' + model_folder
    model = VxmDense_l(img_size)
    best_model = torch.load(model_dir + natsorted(os.listdir(model_dir))[model_idx])['state_dict']
    print('Best model: {}'.format(natsorted(os.listdir(model_dir))[model_idx]))
    model.load_state_dict(best_model)
    model.cuda()
    reg_model = utils.register_model(img_size, 'nearest')
    reg_model.cuda()
    fixed_path = r'G:\CT2CECT\registration\data\cect_a_preprocess'
    moving_path = r'G:\CT2CECT\registration\data\ncct_preprocess'
    save_path = r'G:\CT2CECT\registration\data\voxelmorph'
    fixed_list = get_listdir(fixed_path)
    moving_list = get_listdir(moving_path)
    device = 'cuda'
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

        x_in = torch.cat((input_moving, input_fixed), dim=1)
        moved, warp = model(x_in)

        # predict
        # moved, warp = model(input_moving, input_fixed, registration=True)
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


    # fixed_path = r'G:\CT2CECT\registration\data\cect_a_preprocess'
    # moving_path = r'G:\CT2CECT\registration\data\ncct_preprocess'
    # model_idx = -1
    # img_size = (256, 256, 256)
    # weights = [1, 0.02]
    # model_folder = 'vxm_2_mse_{}_diffusion_{}/'.format(weights[0], weights[1])
    # model_dir = 'experiments/' + model_folder
    # dict = utils.process_label()
    # if os.path.exists('experiments/' + model_folder[:-1] + '.csv'):
        # os.remove('experiments/' + model_folder[:-1] + '.csv')
    # csv_writter(model_folder[:-1], 'experiments/' + model_folder[:-1])
    # line = ''
    # for i in range(46):
        # line = line + ',' + dict[i]
    # csv_writter(line, 'experiments/' + model_folder[:-1])
    # model = VxmDense_l(img_size)
    # best_model = torch.load(model_dir + natsorted(os.listdir(model_dir))[model_idx])['state_dict']
    # print('Best model: {}'.format(natsorted(os.listdir(model_dir))[model_idx]))
    # model.load_state_dict(best_model)
    # model.cuda()
    # reg_model = utils.register_model(img_size, 'nearest')
    # reg_model.cuda()
    # test_composed = transforms.Compose([trans.Seg_norm(),
    #                                     trans.NumpyType((np.float32, np.int16)),
    #                                     ])
    # test_set = TestDataset(fixed_path=fixed_path, moving_path=moving_path)
    # train_set = datasets.JHUBrainDataset(glob.glob(train_dir + '*.pkl'), transforms=train_composed)
    # val_set = datasets.JHUBrainInferDataset(glob.glob(val_dir + '*.pkl'), transforms=val_composed)
    # test_loader = DataLoader(test_set, batch_size=1, shuffle=True, num_workers=4, pin_memory=True)
    # val_loader = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=4, pin_memory=True, drop_last=True)

    # test_set = datasets.JHUBrainInferDataset(glob.glob(test_dir + '*.pkl'), transforms=test_composed)
    # test_loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=1, pin_memory=True, drop_last=True)
    # eval_dsc_def = utils.AverageMeter()
    # eval_dsc_raw = utils.AverageMeter()
    # eval_det = utils.AverageMeter()


    # with torch.no_grad():
        # stdy_idx = 0
        # for data in test_loader:
            # model.eval()
            # data = [t.cuda() for t in data]
            # x = data[0]
            # y = data[1]

            # x_in = torch.cat((x, y), dim=1)
            # x_def, flow = model(x_in)
            # out = torch.squeeze(x_def).cpu().numpy()
            # sitk_img = sitk.GetImageFromArray(out)
            # sitk.WriteImage(sitk_img, os.path.join(save_path, fullflname))
            # def_out = reg_model([x_seg.cuda().float(), flow.cuda()])
            # tar = y.detach().cpu().numpy()[0, 0, :, :, :]
            # jac_det = utils.jacobian_determinant_vxm(flow.detach().cpu().numpy()[0, :, :, :, :])
            # line = utils.dice_val_substruct(def_out.long(), y_seg.long(), stdy_idx)
            # line = line  # +','+str(np.sum(jac_det <= 0)/np.prod(tar.shape))
            # csv_writter(line, 'experiments/' + model_folder[:-1])
            # eval_det.update(np.sum(jac_det <= 0) / np.prod(tar.shape), x.size(0))

            # dsc_trans = utils.dice_val(def_out.long(), y_seg.long(), 46)
            # dsc_raw = utils.dice_val(x_seg.long(), y_seg.long(), 46)
            # print('Trans dsc: {:.4f}, Raw dsc: {:.4f}'.format(dsc_trans.item(), dsc_raw.item()))
            # eval_dsc_def.update(dsc_trans.item(), x.size(0))
            # eval_dsc_raw.update(dsc_raw.item(), x.size(0))
            # stdy_idx += 1

            # flip moving and fixed images
            # y_in = torch.cat((y, x), dim=1)
            # y_def, flow = model(y_in)
            # def_out = reg_model([y_seg.cuda().float(), flow.cuda()])
            # tar = x.detach().cpu().numpy()[0, 0, :, :, :]

            # jac_det = utils.jacobian_determinant_vxm(flow.detach().cpu().numpy()[0, :, :, :, :])
            # line = utils.dice_val_substruct(def_out.long(), x_seg.long(), stdy_idx)
            # line = line  # + ',' + str(np.sum(jac_det < 0) / np.prod(tar.shape))
            # out = def_out.detach().cpu().numpy()[0, 0, :, :, :]
            # print('det < 0: {}'.format(np.sum(jac_det <= 0) / np.prod(tar.shape)))
            # csv_writter(line, 'experiments/' + model_folder[:-1])
            # eval_det.update(np.sum(jac_det <= 0) / np.prod(tar.shape), x.size(0))

            # dsc_trans = utils.dice_val(def_out.long(), x_seg.long(), 46)
            # dsc_raw = utils.dice_val(y_seg.long(), x_seg.long(), 46)
            # print('Trans dsc: {:.4f}, Raw dsc: {:.4f}'.format(dsc_trans.item(), dsc_raw.item()))
            # eval_dsc_def.update(dsc_trans.item(), x.size(0))
            # eval_dsc_raw.update(dsc_raw.item(), x.size(0))
            # stdy_idx += 1

        # print('Deformed DSC: {:.3f} +- {:.3f}, Affine DSC: {:.3f} +- {:.3f}'.format(eval_dsc_def.avg,
        #                                                                             eval_dsc_def.std,
        #                                                                             eval_dsc_raw.avg,
        #                                                                             eval_dsc_raw.std))
        # print('deformed det: {}, std: {}'.format(eval_det.avg, eval_det.std))


# def csv_writter(line, name):
#     with open(name + '.csv', 'a') as file:
#         file.write(line)
#         file.write('\n')


if __name__ == '__main__':
    '''
    GPU configuration
    '''
    GPU_iden = 0
    GPU_num = torch.cuda.device_count()
    print('Number of GPU: ' + str(GPU_num))
    for GPU_idx in range(GPU_num):
        GPU_name = torch.cuda.get_device_name(GPU_idx)
        print('     GPU #' + str(GPU_idx) + ': ' + GPU_name)
    torch.cuda.set_device(GPU_iden)
    GPU_avai = torch.cuda.is_available()
    print('Currently using: ' + torch.cuda.get_device_name(GPU_iden))
    print('If the GPU is available? ' + str(GPU_avai))
    main()






