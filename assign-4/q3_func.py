'''
Functions for q3
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

class autoencoder(nn.Module):
    def __init__(self, deep = True):
        super(autoencoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(28 * 28, 1000),
            nn.ReLU(True),
            nn.Linear(1000,500),
            nn.ReLU(True), 
            nn.Linear(500,250), 
            nn.ReLU(True), 
            nn.Linear(250,30))
        self.decoder = nn.Sequential(
            nn.Linear(30, 250),
            nn.ReLU(True),
            nn.Linear(250,500),
            nn.ReLU(True),
            nn.Linear(500,1000),
            nn.ReLU(True), 
            nn.Linear(1000, 28 * 28), 
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

def to_img_infer(x):
    x = 0.5 * (x + 1)
    x = x.clamp(0, 1)
    x = x.view(28, 28)
    return x

def do_inference(model, 
    digit , 
    non_digit = './dummy_data/panda.png' ,
    path = './logs/q3/deep_autoencoder_hidden_64/' ,
    device = torch.device('cpu')):

    plt.subplot(1, 2, 1);
    # Reconstruction of digit
    img = digit
    print(img.shape)
    output, _ = model(img)
    pic = to_img_infer(output).detach().numpy()

    plt.imshow(img.reshape(28,28) * 255,
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
    img = cv2.resize(img, (28, 28))/ 255.0
    img = torch.Tensor(img)
    img = img.view(784)
    print(img.shape)
    output, _ = model(img)
    pic = (output.data).reshape(28,28)
    pic = to_img_infer(output).detach().numpy()

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

def do_autoencoder(train_loader, device = torch.device('cpu'), sparsity = 0.1, learning_rate = 0.01, num_epochs = 100 ):
    
    loss_ll=[]
    # Labels
    label = "deep"

    print(f"\n\nAutoencoder {label}\n-------\n")

    path = f'./logs/q3/{label}_autoencoder_sparsity_{sparsity}/'
    if not os.path.exists(path):
        os.mkdir(path)
    
    model = autoencoder().to(device)

    path = f'./checkpoints/q3/{label}_autoencoder_sparsity_{sparsity}.pth'

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
            output, latent = model(img)
            loss = criterion(output, img)
            # all_params = torch.cat([x.view(-1) for x in model.encoder.parameters()])
            l1_regularization = sparsity * torch.norm(latent, 1)
            loss = loss + l1_regularization
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
            save_image(pic[:16], f'./logs/q3/{label}_autoencoder_sparsity_{sparsity}/image_{epoch}.png')

        # Increment
        epoch+=1

    # Plot convergence
    path =f'./logs/q3/convergence-sparsity-{sparsity}.png'
    plt.plot(np.arange(len(loss_ll))+1, loss_ll)
    plt.title("Loss convergence q3")
    plt.xlabel("Epochs")
    plt.savefig(path)
    plt.close()

    path = './checkpoints/q3/'
    print(f"Saving checkpoint at {path}")
    if not os.path.exists(path):
        os.mkdir(path)

    save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer' : optimizer.state_dict(),
        }, 
        is_best = True,
        filename =  f'./checkpoints/q3/{label}_autoencoder_sparsity_{sparsity}.pth')

    # Perform inference
    for data in train_loader:
        img, _ = data 
        img = img.view(img.size(0), -1)
        img = img.to(torch.device("cpu"))
        break 
    path = f'./logs/q3/{label}_autoencoder_sparsity_{sparsity}/'
    do_inference(model.to(torch.device("cpu")), digit = img[0], path = path, device = device)