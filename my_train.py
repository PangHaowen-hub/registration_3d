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
parser.add_argument('--steps-per-epoch', type=int, default=100,
                    help='frequency of model saves (default: 100)')
parser.add_argument('--load-model', help='optional model file to initialize with')
parser.add_argument('--initial-epoch', type=int, default=0,
                    help='initial epoch number (default: 0)')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate (default: 1e-4)')
# network architecture parameters
parser.add_argument('--enc', type=int, nargs='+',
                    help='list of unet encoder filters (default: 16 32 32 32)')
parser.add_argument('--dec', type=int, nargs='+',
                    help='list of unet decorder filters (default: 32 32 32 32 32 16 16)')
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
enc_nf = args.enc if args.enc else [8, 16, 16]
dec_nf = args.dec if args.dec else [16, 16, 16, 8]

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
wind.line([0.], [0.], win='train', opts=dict(title='loss', legend=['loss']))

for epoch in range(args.initial_epoch, args.epochs):

    # save model checkpoint
    if epoch % 20 == 0:
        model.save(os.path.join(model_dir, '%04d.pt' % epoch))

    epoch_loss = []
    epoch_total_loss = []
    epoch_step_time = []

    for i, batch in enumerate(train_dataloader):
        step_start_time = time.time()

        # generate inputs (and true outputs) and convert them to tensors
        inputs, y_true = batch
        inputs = inputs.to(device).float().unsqueeze(1)
        y_true = y_true.to(device).float().unsqueeze(1)

        # run inputs through the model to produce a warped image and flow field
        y_pred = model(inputs, y_true)

        # calculate total loss
        loss = 0
        loss_list = []
        for n, loss_function in enumerate(losses):
            curr_loss = loss_function(y_true, y_pred[0]) * weights[n]
            loss_list.append(curr_loss.item())
            loss += curr_loss

        epoch_loss.append(loss_list)
        epoch_total_loss.append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # get compute time
        epoch_step_time.append(time.time() - step_start_time)

    wind.line([np.mean(epoch_total_loss)], [epoch + 1], win='train', update='append')
    epoch_info = 'Epoch %d/%d' % (epoch + 1, args.epochs)
    time_info = '%.4f sec/step' % np.mean(epoch_step_time)
    losses_info = ', '.join(['%.4e' % f for f in np.mean(epoch_loss, axis=0)])
    loss_info = 'loss: %.4e  (%s)' % (np.mean(epoch_total_loss), losses_info)
    print(' - '.join((epoch_info, time_info, loss_info)), flush=True)

# final model save
model.save(os.path.join(model_dir, '%04d.pt' % args.epochs))
