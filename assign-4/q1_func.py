'''
Functions for q1 
''' 
import numpy as np
import os
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

def do_PCA(train_data, supress = False, save_path = "./logs/q1/pca.png"):
    # PCA 
    pca = PCA(n_components = 30)
    train_x = train_data.train_data.type(torch.FloatTensor)/255.0
    train_x = train_x.reshape(-1,784)

    print("\n\nPCA \n--------\n")
    print(f"Reshaped Train Data Shape {train_x.shape}")
    print(f"Applying PCA ...")
    lower_dimensional_data = pca.fit_transform(train_x)
    print(f"Number of reduced components {pca.n_components_}")
    print(f"Estimated noise variance {pca.noise_variance_}")

    # Explained Variance
    explained_variance = np.sum(pca.explained_variance_ratio_) 
    print(f"Explained variance ratio {explained_variance}")

    approx_pca = pca.inverse_transform(lower_dimensional_data)

    # Original Image (784 components)
    plt.subplot(1, 2, 1);
    plt.imshow(train_x[5].reshape(28,28) * 255,
                cmap = plt.cm.gray, interpolation='nearest',
                clim=(0, 255));
    plt.xlabel('784 Components', fontsize = 12)
    plt.title('Original Image', fontsize = 14);

    # 30 principal components
    plt.subplot(1, 2, 2);
    plt.imshow(approx_pca[5].reshape(28, 28) * 255,
                cmap = plt.cm.gray, interpolation='nearest',
                clim=(0, 255));
    plt.xlabel('30 Components', fontsize = 12)
    plt.title(f'{explained_variance*100 : .2f}% of Explained Variance', fontsize = 14)

    if save_path:
        plt.savefig(save_path)

    if supress:
        plt.close()
    else:
        plt.show()

class autoencoder(nn.Module):
    def __init__(self, deep = True):
        super(autoencoder, self).__init__()

        if deep:
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
        else:
            self.encoder = nn.Sequential(
                nn.Linear(28 * 28, 1000),
                nn.Linear(1000,500),
                nn.Linear(500,250),
                nn.Linear(250,30))
            self.decoder = nn.Sequential(
                nn.Linear(30, 250),
                nn.Linear(250,500),
                nn.Linear(500,1000),
                nn.Linear(1000, 784))

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

def to_img(x, deep = True):
    if deep:
        x = 0.5 * (x + 1)
        x = x.clamp(0, 1)
    x = x.view(x.size(0), 1, 28, 28)
    return x

def do_autoencoder(train_loader, device = torch.device('cpu') , deep = True, learning_rate = 0.01, num_epochs = 100 ):
    loss_ll=[]
    # Labels
    if deep:
        label = "deep"
    else:
        label = "linear"

    print(f"\n\nAutoencoder {label}\n-------\n")

    if not os.path.exists(f'./logs/q1/{label}_autoencoder/'):
        os.mkdir(f'./logs/q1/{label}_autoencoder/')
    
    model = autoencoder(deep = deep).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr = learning_rate, weight_decay=1e-5)

    for epoch in range(num_epochs):
        for data in train_loader:
            img, _ = data
            img = img.view(img.size(0), -1).to(device)
        
            # ===================forward=====================
            output = model(img)
            loss = criterion(output, img)
            # ===================backward====================
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        # ===================log========================
        print(f'epoch [{epoch + 1,}/{num_epochs}], loss:{loss.data.item() :.4f}')
        loss_ll.append(loss.data.item())

        if epoch % 10 == 0:
            pic = to_img(output.to(device).data, deep = deep)
            save_image(pic[:16], f'./logs/q1/{label}_autoencoder/image_{epoch}.png')

    # Plot convergence
    path =f'./logs/q1/convergence-{label}_autoencoder.png'
    plt.plot(np.arange(len(loss_ll))+1, loss_ll)
    plt.title("Loss convergence q1")
    plt.xlabel("Epochs")
    plt.savefig(path)
    plt.close()

    if not os.path.exists('./checkpoints/q1/'):
        os.mkdir('./checkpoints/q1/')
    torch.save(model.state_dict(), f'./checkpoints/q1/{label}_autoencoder.pth')
