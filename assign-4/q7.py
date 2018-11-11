"""
q7 Tutorial 4
"""
import os
import argparse
import numpy as np
from sys import platform as sys_pf
if sys_pf == 'darwin':
    import matplotlib
    # matplotlib.use("MacOSX")
    matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# Torch Modules 
import torch
from torch import nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torchvision.utils import save_image

# Library files
from q7_func import *

# Args
parser = argparse.ArgumentParser(description='Q4 Tutorial 4')
parser.add_argument('--disable-cuda', action='store_true',
                    help='Disable CUDA')
args = parser.parse_args()
args.device = None
if not args.disable_cuda and torch.cuda.is_available():
    args.device = torch.device('cuda')
else:
    args.device = torch.device('cpu')

# torch.manual_seed(1)    # reproducible
# Hyper Parameters
EPOCHS = 10               # train the training data n times, to save time, we just train 1 epoch
BATCH_SIZE = 1024
INPUT_SIZE = 28         # image width, height
LR = 0.01               # learning rate
DOWNLOAD_MNIST = not os.path.exists('/tmp/mnist/')   # set to True if haven't download the data

# Mnist digital dataset
train_data = dsets.MNIST(
    root='/tmp/mnist/',
    train=True,                         # this is training data
    transform=transforms.ToTensor(),    # Converts a PIL.Image or numpy.ndarray to
                                        # torch.FloatTensor of shape (C x H x
                                        # W) and normalize in the range [0.0,
                                        # 1.0]
    download=DOWNLOAD_MNIST,            # download it if you don't have it
)
test_data = dsets.MNIST(
    root='/tmp/mnist/',
    train=False,
    transform=transforms.ToTensor())

# plot one example
print(f"Train data size {train_data.train_data.size()}")     # (60000, 28, 28)
print(f"Train data labels nos {train_data.train_labels.size()}")   # (60000)
plt.imshow(train_data.train_data[0].numpy(), cmap='gray')
plt.title('%i' % train_data.train_labels[0])
# plt.show()
plt.close()

# Data Loader for easy mini-batch return in training
train_loader = torch.utils.data.DataLoader(
    dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)

# convert test data into Variable, pick 2000 samples to speed up testing
test_x = test_data.test_data.type(
    torch.FloatTensor)[:2000] / 255.   # shape (2000, 28, 28) value in range(0,1)
test_y = test_data.test_labels.numpy().squeeze()[:2000]    # covert to numpy array

make_dirs("logs/q7")
make_dirs("checkpoints/q7")

upsamp_ll = ["deconv","unpool-deconv"]

for upsamp in upsamp_ll:
    print(f"upsamp {upsamp}")
    do_autoencoder(train_loader, upsamp = upsamp, device = args.device, learning_rate = LR, num_epochs = EPOCHS )
