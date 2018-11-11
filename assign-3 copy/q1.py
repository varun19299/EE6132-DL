"""
Q1 Tutorial 4
"""
import torch
from torch import nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from sys import platform as sys_pf
if sys_pf == 'darwin':
    import matplotlib
    matplotlib.use("MacOSX")

import argparse
import os
import matplotlib.pyplot as plt

# torch.manual_seed(1)    # reproducible

# Hyper Parameters
EPOCH = 1               # train the training data n times, to save time, we just train 1 epoch
BATCH_SIZE = 64
TIME_STEP = 28          # rnn time step / image height
INPUT_SIZE = 28         # rnn input size / image width
LR = 0.01               # learning rate
DOWNLOAD_MNIST = True   # set to True if haven't download the data

parser = argparse.ArgumentParser()
parser.add_argument("--model", default ="LSTM" , choices = ["GRU","LSTM","RNN"],help = "Model to use")
parser.add_argument('-l','--list', nargs='+', type =int, help='<Required> Set flag', required=True)
parser.add_argument("--bi", default =0, type = int,help = " Bi directional or not")
args = parser.parse_args()

experiment={"hidden-layers":1, "width":[128],"type": "LSTM"}

torch_model = {"LSTM": nn.LSTM, "RNN": nn.RNN, "GRU":nn.GRU}

# Mnist digital dataset
train_data = dsets.MNIST(
    root='/tmp/mnist/',
    train=True,                         # this is training data
    transform=transforms.ToTensor(),    # Converts a PIL.Image or numpy.ndarray to
                                        # torch.FloatTensor of shape (C x H x W) and normalize in the range [0.0, 1.0]
    download=DOWNLOAD_MNIST,            # download it if you don't have it
)

# plot one example
print(train_data.train_data.size())     # (60000, 28, 28)
print(train_data.train_labels.size())   # (60000)
# plt.imshow(train_data.train_data[0].numpy(), cmap='gray')
# plt.title('%i' % train_data.train_labels[0])
# plt.show()

# Data Loader for easy mini-batch return in training
train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)

# convert test data into Variable, pick 2000 samples to speed up testing
test_data = dsets.MNIST(root='/tmp/mnist/', train=False, transform=transforms.ToTensor())
test_x = test_data.test_data.type(torch.FloatTensor)[:2000]/255.   # shape (2000, 28, 28) value in range(0,1)
test_y = test_data.test_labels.numpy().squeeze()[:2000]    # covert to numpy array


class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()

        model = torch_model[args.model]

        self.embedding={}

        for i in range(len(args.list)-1):
            if i ==0:
                self.embedding[0] = nn.Linear(28, args.list[0])
            else:
                self.embedding[i] = nn.Linear(args.list[i-1], args.list[i])
        INPUT_SIZE = 28
        if len(args.list)-1:
            INPUT_SIZE = args.list[-2]

        self.rnn = model(         # if use nn.RNN(), it hardly learns
            input_size=INPUT_SIZE,
            hidden_size= args.list[-1],         # rnn hidden unit
            num_layers=1,           # number of rnn layer
            batch_first=True,       # input & output will has batch size as 1s dimension. e.g. (batch, time_step, input_size)
            bidirectional = bool(args.bi)
        )

        self.out = nn.Linear(args.list[-1], 10)

    def forward(self, x):
        # x shape (batch, time_step, input_size)
        # r_out shape (batch, time_step, output_size)
        # h_n shape (n_layers, batch, hidden_size)
        # h_c shape (n_layers, batch, hidden_size)

        if len(self.embedding):
            for i in range(len(self.embedding)):
                if i == 0:
                    embed = self.embedding[0](x)
                else:
                    embed = self.embedding[i](embed)
        else:
            embed = x

        if args.model == "LSTM":
            r_out, (h_n, h_c) = self.rnn(embed, None)   # None represents zero initial hidden state
        else:
            r_out, _ = self.rnn (embed, None)

        # choose r_out at the last time step
        out = self.out(r_out[:, -1, :])
        return out


rnn = RNN()
print(rnn)

test_acc = []
train_ll = []
test_ll = []

optimizer = torch.optim.Adam(rnn.parameters(), lr=LR)   # optimize all cnn parameters
loss_func = nn.CrossEntropyLoss()                       # the target label is not one-hotted

# training and testing
for epoch in range(EPOCH):
    for step, (b_x, b_y) in enumerate(train_loader):        # gives batch data
        b_x = b_x.view(-1, 28, 28)              # reshape x to (batch, time_step, input_size)

        output = rnn(b_x)                               # rnn output
        loss = loss_func(output, b_y)                   # cross entropy loss
        optimizer.zero_grad()                           # clear gradients for this training step
        loss.backward()                                 # backpropagation, compute gradients
        optimizer.step()                                # apply gradients

        if step % 50 == 0:
            test_output = rnn(test_x)                   # (samples, time_step, input_size)
            test_loss = loss_func(test_output, torch.Tensor(test_y).long())
            pred_y = torch.max(test_output, 1)[1].data.numpy().squeeze()
            accuracy = float((pred_y == test_y).astype(int).sum()) / float(test_y.size)
            print('Epoch: ', epoch, '| Step: ', step,'| train loss: %.4f' % loss.data.numpy(), '| test accuracy: %.2f' % accuracy)

            test_acc.append(accuracy)
            train_ll.append(float(loss.detach().numpy()))
            test_ll.append(float(test_loss.detach().numpy()))

# print 10 predictions from test data
test_output = rnn(test_x[:10].view(-1, 28, 28))
pred_y = torch.max(test_output, 1)[1].data.numpy().squeeze()
print(pred_y, 'prediction number')
print(test_y[:10], 'real number')

print(f"Len {len(test_acc)}")
print(f"Len {len(test_ll)}")
print(f"Len {len(train_ll)}")

print(train_ll)
plt.plot(np.arange(len(test_acc))*50,test_acc )
plt.plot(np.arange(len(test_acc))*50, train_ll)
plt.plot(np.arange(len(test_acc))*50, test_ll)
plt.legend(['Accuracy (test)','Loss (train)', 'Loss (test)'])
plt.title(f"Accuracy-Loss versus Steps for Hidden units {args.list}")
if not args.bi:
    plt.savefig(f"logs/Q1-MNIST-{args.model}-{args.list}.png")
else:
    plt.savefig(f"logs/Q1-MNIST-{args.model}-bidirectional-{args.list}.png")
plt.close()
