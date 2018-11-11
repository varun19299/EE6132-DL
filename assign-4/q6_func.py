'''
Functions for q6
'''
import numpy as np
import os
import shutil
import cv2
from sys import platform as sys_pf
if sys_pf == 'darwin':
    import matplotlib
    # matplotlib.use("MacOSX")
    matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Torch Modules
import torch
from torch import nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torchvision.utils import save_image

def make_dirs(path):
    if not os.path.exists(path):
        os.mkdir(path)

class autoencoder(nn.Module):
    def __init__(self, deep = True):
        super(autoencoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(28 * 28, 64),
            nn.ReLU(True),
            nn.Linear(64,8),
            nn.ReLU(True))
        self.decoder = nn.Sequential(
            nn.Linear(8,64),
            nn.ReLU(True), 
            nn.Linear(64, 28 * 28), 
            nn.Tanh())

    def forward(self, x):
        latent = self.encoder(x)
        x = self.decoder(latent)
        return x, latent

def to_img(x):
    x = 0.5 * (x + 1)
    x = x.clamp(0, 1)
    x = x.view(x.size(0), 1, 28, 28)
    return x

def add_noise(img, noise = 0.3 ):
    '''
    Add (not in place) random noise
    '''
    noise = torch.randn(img.size()) * noise
    noisy_img = img + noise
    return noisy_img

def do_inference(model, 
    digit , noise = 0.3,
    non_digit = './dummy_data/panda.png' ,
    path = './logs/q6/deep_autoencoder_hidden_64/' ,
    device = torch.device('cpu')):

    # Reconstruction of digit
    img = digit
    noisy = add_noise(img, noise)
    output, _ = model(noisy)
    pic = (output.data).reshape(28,28)

    plt.subplot(1, 3, 1);
    plt.imshow(img.reshape(28,28) * 255,
                cmap = plt.cm.gray, interpolation='nearest',
                clim=(0, 255));
    plt.title('Original Digit', fontsize = 14);

    plt.subplot(1,3,2)
    plt.imshow(noisy.reshape(28,28) * 255,
                cmap = plt.cm.gray, interpolation='nearest',
                clim=(0, 255));
    plt.title('Noisy Digit', fontsize = 14);

    plt.subplot(1,3,3)
    plt.imshow(pic.reshape(28,28) * 255,
                cmap = plt.cm.gray, interpolation='nearest',
                clim=(0, 255));
    plt.title('Reconstruction of Noisy Digit', fontsize = 14);

    path_d = path+"inference-digit.png"
    print(f"Saving inference at {path_d}")
    plt.savefig(path_d)

    # Reconstruction of digit
    img = cv2.imread(non_digit,0)
    img = cv2.resize(img, (28, 28))
    img = torch.Tensor(img)
    img = img.view(784)
    output, _ = model(add_noise(img, noise))
    pic = (output.data).reshape(28,28)

    plt.subplot(1,2,1)
    plt.imshow(img.reshape(28,28) * 255,
                cmap = plt.cm.gray, interpolation='nearest',
                clim=(0, 255));
    plt.title('Original Image', fontsize = 14);

    plt.subplot(1,2,2)
    plt.imshow(pic.reshape(28,28) * 255,
                cmap = plt.cm.gray, interpolation='nearest',
                clim=(0, 255));
    plt.title('Reconstruction of a Non-Digit', fontsize = 14);

    path_n = path+"inference-non.png"
    print(f"Saving inference at {path_n}")
    plt.savefig(path_n)
    # Filter visualisation

    sub = 1
    for m in model.modules():
        if isinstance(m, nn.Linear):
            print(f"Model layer {m}")
            print(m.weight.shape)
            im = m.weight
            im = im.detach().numpy()
            im = (im- im.min())/(im.max()) * 255
            plt.imshow(im,
                cmap = plt.cm.gray, interpolation='nearest',
                clim=(0, 255));
            plt.title('Filter visualisation', fontsize = 14);
            sub +=1
            path_w = path+f"inference-weights-layer-{sub}.png"
            print(f"Saving inference at {path_w}")
            plt.savefig(path_w)
            plt.close()


def save_checkpoint(state, is_best = True, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')

def do_autoencoder(train_loader, device = torch.device('cpu'), noise = 0.3, learning_rate = 0.01, num_epochs = 100 ):
    
    loss_ll = []
    # Labels
    label = "deep"

    print(f"\n\nAutoencoder {label}\n-------\n")

    path = f'./logs/q6/{label}_autoencoder_inference_noise_{noise}/'
    if not os.path.exists(path):
        os.mkdir(path)
    
    model = autoencoder().to(device)

    path = f'./checkpoints/q6/{label}_autoencoder_inference_noise_{noise}.pth'

    optimizer = torch.optim.Adam(
        model.parameters(), lr = learning_rate, weight_decay=1e-5)
    criterion = nn.MSELoss()

    # Load epoch, optimiser from checkpoint if available
    if os.path.isfile(path):
        print(f"=> loading checkpoint {path}")
        checkpoint = torch.load(path)
        epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print(f"=> loaded checkpoint {path} (epoch {checkpoint['epoch']})")
    else:
        print("=> no checkpoint found at '{}'".format(path))
        epoch = 0

    while epoch < num_epochs:
        for data in train_loader:
            img, _ = data
            img = img.view(img.size(0), -1).to(device)
        
            # ===================forward=====================
            output, _ = model(img)
            loss = criterion(output, img)
    
            # ===================backward====================
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        # ===================log========================
        print(f'epoch [{epoch + 1,}/{num_epochs}], loss:{loss.data.item() :.4f}')

        loss_ll.append(loss.data.item())

        if epoch % 5 == 0:
            # Run eval
            pic = to_img(output.to(device).data)
            save_image(pic[:16], f'./logs/q6/{label}_autoencoder_inference_noise_{noise}/image_{epoch}.png')

        # Increment
        epoch+=1

    # Plot convergence
    path =f'./logs/q6/convergence-noise-{noise}.png'
    plt.plot(np.arange(len(loss_ll))+1, loss_ll)
    plt.title(f"Loss convergence noise {noise} q6")
    plt.xlabel("Epochs")
    plt.savefig(path)
    plt.close()

    path = './checkpoints/q6/'
    print(f"Saving checkpoint at {path}")
    if not os.path.exists(path):
        os.mkdir(path)

    save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer' : optimizer.state_dict(),
        }, 
        is_best = True,
        filename =  f'./checkpoints/q6/{label}_autoencoder_inference_noise_{noise}.pth')

    # Perform inference
    for data in train_loader:
        img, _ = data 
        img = img.view(img.size(0), -1)
        img = img.to(torch.device("cpu"))
        break 
    path = f'./logs/q6/{label}_autoencoder_inference_noise_{noise}/'
    do_inference(model.to(torch.device("cpu")), noise = noise, digit = img[0], path = path, device = device)