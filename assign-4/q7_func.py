
'''
Functions for q7
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
    def __init__(self, upsamp = True):
        super(autoencoder, self).__init__()

        if upsamp =="unpool":
            pass
        elif upsamp == "deconv":
            self.encoder = nn.Sequential(
                nn.Conv2d(1, 16, 3, stride=3, padding=1),  # b, 16, 10, 10
                nn.ReLU(True),
                nn.MaxPool2d(2, stride=2),  # b, 16, 5, 5
                nn.Conv2d(16, 16, 3, stride=2, padding=1),  # b, 8, 3, 3
                nn.ReLU(True),
                nn.MaxPool2d(2, stride=1)  # b, 8, 2, 2
            )
            self.decoder = nn.Sequential(
                nn.ConvTranspose2d(16, 16, 3, stride=2),  # b, 16, 5, 5
                nn.ReLU(True),
                nn.ConvTranspose2d(16, 16, 5, stride=3, padding=1),  # b, 8, 15, 15
                nn.ReLU(True),
                nn.ConvTranspose2d(16, 1, 2, stride=2, padding=1),  # b, 1, 28, 28
                nn.Tanh()
            )
        elif upsamp=="unpool-deconv":
            self.encoder = nn.Sequential(
                nn.Conv2d(1, 16, 3, stride=3, padding=1),  # b, 16, 10, 10
                nn.ReLU(True),
                nn.MaxPool2d(2, stride=2),  # b, 16, 5, 5
                nn.Conv2d(16, 16, 3, stride=2, padding=1),  # b, 8, 3, 3
                nn.ReLU(True),
                nn.MaxPool2d(2, stride=1)  # b, 8, 2, 2
            )
            self.decoder = nn.Sequential(
                nn.MaxUnpool2d(2, stride=1),  # b, 8, 2, 2,
                nn.ConvTranspose2d(16, 16, 3, stride=2, padding=1),  # b, 16, 10, 10
                nn.ReLU(True),
                nn.MaxUnpool2d(2, stride=2),  # b, 16, 5, 5
                nn.ConvTranspose2d(16, 1, 3, stride=3, padding=1),  # b, 8, 3, 3
                nn.Tanh()
            )            
       
    def forward(self, x):
        latent = self.encoder(x)
        x = self.decoder(latent)
        return x, latent

def to_img(x):
    x = 0.5 * (x + 1)
    x = x.clamp(0, 1)
    x = x.view(x.size(0), 1, 28, 28)
    return x

def add_upsamp(img, upsamp = 0.3 ):
    '''
    Add (not in place) random upsamp
    '''
    upsamp = torch.randn(img.size()) * upsamp
    noisy_img = img + upsamp
    return noisy_img

def do_inference(model, 
    digit , 
    non_digit = './dummy_data/panda.png' ,
    path = './logs/q7/deep_autoencoder_hidden_64/' ,
    device = torch.device('cpu')):

    plt.subplot(1, 2, 1);
    # Reconstruction of digit
    img = digit
    output, _ = model(img)
    pic = (output.data[0]).reshape(28,28)

    plt.imshow(img[0].reshape(28,28) * 255,
                cmap = plt.cm.gray, interpolation='nearest',
                clim=(0, 255));
    plt.title('Original Digit', fontsize = 14);

    plt.subplot(1,2,2)
    plt.imshow(pic.reshape(28,28) * 255,
                cmap = plt.cm.gray, interpolation='nearest',
                clim=(0, 255));
    plt.title('Reconstruction of a Digit', fontsize = 14);

    path_d = path+"inference-digit.png"
    print(f"Saving inference at {path_d}")
    plt.savefig(path_d)

    # Reconstruction of digit
    img = cv2.imread(non_digit,0)
    img = cv2.resize(img, (28, 28))/255
    img = torch.Tensor(img).view(1,1,28,28)
    output, _ = model(img)
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


def save_checkpoint(state, filename='checkpoint.pth.tar'):
    torch.save(state, filename)

def do_autoencoder(train_loader, device = torch.device('cpu'), upsamp = 0.3, learning_rate = 0.01, num_epochs = 100 ):
    '''
    upsamp one of [unpool, unpool-dencov, deconv, rev]
    '''
    loss_ll = []
    # Labels
    label = "deep"

    print(f"\n\nAutoencoder {label}\n-------\n")

    path = f'./logs/q7/{label}_autoencoder_upsamp_{upsamp}/'
    if not os.path.exists(path):
        os.mkdir(path)
    
    model = autoencoder(upsamp = upsamp).to(device)

    path = f'./checkpoints/q7/{label}_autoencoder_upsamp_{upsamp}.pth'

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
            img = img.to(device)
        
            # ===================forward=====================
            output, _ = model(img)
            loss = criterion(output, img)
            # all_params = torch.cat([x.view(-1) for x in model.parameters()])
            # l1_regularization = upsamp * torch.norm(all_params, 1)
            # loss = loss + l1_regularization
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
            save_image(pic[:16], f'./logs/q7/{label}_autoencoder_upsamp_{upsamp}/image_{epoch}.png')

        # Increment
        epoch+=1

    # Plot convergence
    path =f'./logs/q7/convergence-upsamp-{upsamp}.png'
    plt.plot(np.arange(len(loss_ll))+1, loss_ll)
    plt.title(f"Loss convergence upsamp {upsamp} q7")
    plt.xlabel("Epochs")
    plt.savefig(path)
    plt.close()

    path = './checkpoints/q7/'
    print(f"Saving checkpoint at {path}")
    if not os.path.exists(path):
        os.mkdir(path)

    save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer' : optimizer.state_dict(),
        }, 
        filename =  f'./checkpoints/q7/{label}_autoencoder_upsamp_{upsamp}.pth')

    # Perform inference
    for data in train_loader:
        img, _ = data 
        img = img.to(torch.device("cpu"))
        break 
    
    path = f'./logs/q7/{label}_autoencoder_upsamp_{upsamp}/'
    do_inference(model.to(torch.device("cpu")), digit = img, path = path, device = device)