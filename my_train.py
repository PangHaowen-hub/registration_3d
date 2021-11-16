import os
import random
import argparse
import time
import numpy as np
import torch
from torch.utils.data import DataLoader
from my_dataset import MyDataset
import voxelmorph as vxm
from visdom import Visdom

# parse the commandline
parser = argparse.ArgumentParser()
parser.add_argument('--model-dir', default='models', help='model output directory')
# training parameters
parser.add_argument('--batch-size', type=int, default=1, help='batch size (default: 1)')
parser.add_argument('--epochs', type=int, default=1500,
                    help='number of training epochs (default: 1500)')
parser.add_argument('--load-model', default='./models/0040.pt', help='optional model file to initialize with')
# TODO:optional model file to initialize with
parser.add_argument('--initial-epoch', type=int, default=40,
                    help='initial epoch number (default: 0)')
# TODO:initial epoch number
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate (default: 1e-4)')
# network architecture parameters
parser.add_argument('--int-steps', type=int, default=7,
                    help='number of integration steps (default: 7)')
parser.add_argument('--int-downsize', type=int, default=2,
                    help='flow downsample factor for integration (default: 2)')
# loss hyperparameters
parser.add_argument('--image-loss', default='mse',
                    help='image reconstruction loss - can be mse or ncc (default: mse)')
parser.add_argument('--lambda', type=float, dest='weight', default=0.01,
                    help='weight of deformation loss (default: 0.01)')
args = parser.parse_args()

# fixed_path = r'./data/fixed_resample_norm_patch'
# moving_path = r'./data/moving_a_resample_norm_patch'

fixed_path = r'G:\CT2CECT\registration\fixed_resample_norm_patch'
moving_path = r'G:\CT2CECT\registration\moving_a_resample_norm_patch'

train_dataset = MyDataset(fixed_path, moving_path)
train_dataloader = DataLoader(train_dataset,
                              batch_size=args.batch_size,
                              shuffle=True,
                              pin_memory=True,
                              drop_last=True)

# extract shape from sampled input
inshape = train_dataset[0][0].shape

# prepare model folder
model_dir = args.model_dir
os.makedirs(model_dir, exist_ok=True)

device = 'cuda'

# unet architecture
enc_nf = [8, 16, 16]
dec_nf = [16, 16, 16, 8]

if args.load_model:
    # load initial model (if specified)
    model = vxm.networks.VxmDense.load(args.load_model, device)
else:
    # otherwise configure new model
    model = vxm.networks.VxmDense(
        inshape=inshape,
        nb_unet_features=[enc_nf, dec_nf],
        int_steps=args.int_steps,
        int_downsize=args.int_downsize
    )

# prepare the model for training and send to device
model.to(device)
model.train()

# set optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

# prepare image loss
if args.image_loss == 'ncc':
    image_loss_func = vxm.losses.NCC().loss
elif args.image_loss == 'mse':
    image_loss_func = vxm.losses.MSE().loss
else:
    raise ValueError('Image loss should be "mse" or "ncc", but found "%s"' % args.image_loss)

losses = [image_loss_func]
weights = [1]

# prepare deformation loss
losses += [vxm.losses.Grad('l2', loss_mult=args.int_downsize).loss]
weights += [args.weight]

# 实例化窗口
wind = Visdom()
# 初始化窗口参数
wind.line([[0., 0., 0.]], [0.], win='train', opts=dict(title='loss', legend=['loss', 'image_loss', 'deformation_loss']))

for epoch in range(args.initial_epoch, args.epochs):

    # save model checkpoint
    if epoch % 10 == 0:
        model.save(os.path.join(model_dir, '%04d.pt' % epoch))

    epoch_step_time = []
    epoch_total_loss = []
    image_loss_list = []
    deformation_loss_list = []

    for i, batch in enumerate(train_dataloader):
        step_start_time = time.time()

        fixed, moving = batch
        fixed = fixed.to(device).float().unsqueeze(1)
        moving = moving.to(device).float().unsqueeze(1)

        moved, warp = model(moving, fixed)

        image_loss = losses[0](fixed, moved) * weights[0]
        deformation_loss = losses[1](warp) * weights[1]
        loss = image_loss + deformation_loss

        image_loss_list.append(image_loss.item())
        deformation_loss_list.append(deformation_loss.item())
        epoch_total_loss.append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # get compute time
        epoch_step_time.append(time.time() - step_start_time)

        wind.line([[np.mean(epoch_total_loss), np.mean(image_loss_list), np.mean(deformation_loss_list)]],
                  [epoch + i / len(train_dataloader)], win='train', update='append')
        epoch_info = 'Epoch %d/%d' % (epoch + 1, args.epochs)
        time_info = '%.4f sec/step' % np.mean(epoch_step_time)
        total_loss_info = 'loss: %.4f' % np.mean(epoch_total_loss)
        loss_info = '(image_loss: %.8f, deformation_loss: %.8f)' % (
            np.mean(image_loss_list), np.mean(deformation_loss_list))
        print(' - '.join((epoch_info, time_info, total_loss_info, loss_info)), flush=True)

# final model save
model.save(os.path.join(model_dir, '%04d.pt' % args.epochs))
