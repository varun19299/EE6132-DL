'''
Q2 Assignment 4
'''

import torch
from torch import nn
import numpy as np
import argparse
import os
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("--H", type = int, default =5 ,help = "Hidden length")
args = parser.parse_args()

LR = 0.01
EPOCH = 15
ITER = 100
ITER_EVAL = 20
batch = 32

HIDDEN = args.H

acc = []
ll = []

def gen_seq(L, batch = 1, i=1):
    '''
    Generate a sequence of 0-10, length L.
    And output as an index (default 0)

    Return as [batch, L, 10]; [batch, 10]
    '''
    d = np.random.randint(low= 0, high = 9, size = (batch, L))

    x = np.zeros((batch, L,10))
    for b in range(batch):
        x[b,np.arange(L), d[b]] = 1
    
    # x[np.arange(L), d] = 1

    d = torch.tensor(d, dtype=torch.int)
    x = torch.tensor(x, dtype=torch.int)

    y_o = d[:,i]
    y = np.zeros((batch,10))

    for b in range(batch):
        y[b,y_o[b]] = 1

    y = torch.tensor(y, dtype=torch.int)

    return d,x,y_o,y


class LSTM(nn.Module):
    def __init__(self, input_size = 10,
            hidden_size = HIDDEN):
        super(LSTM, self).__init__()

        self.lstm = nn.LSTM(         # if use nn.RNN(), it hardly learns
            input_size=input_size,
            hidden_size=hidden_size,         # rnn hidden unit
            num_layers=1,           # number of rnn layer
            batch_first=True,       # input & output will has batch size as 1s dimension. e.g. (batch, time_step, input_size)
        )

        self.out = nn.Linear(hidden_size, 10)

    def forward(self, x):
        # x shape (batch, time_step, input_size)
        # r_out shape (batch, time_step, output_size)
        # h_n shape (n_layers, batch, hidden_size)
        # h_c shape (n_layers, batch, hidden_size)
        r_out, (h_n, h_c) = self.lstm(x, None)   # None represents zero initial hidden state

        # choose r_out at the last time step
        out = self.out(r_out[:, -1, :])
        return out

rnn = LSTM()
print(rnn)

optimizer = torch.optim.Adam(rnn.parameters(), lr=LR)   # optimize all cnn parameters
loss_func = nn.CrossEntropyLoss()                       # the target label is not one-hotted

# training and testing
for epoch in range(EPOCH):
    print(f"\n\n----\n\nEpoch {epoch}")
    train_accuracy = 0

    for i in range(ITER):
        L = np.random.randint(low = 3, high = 10)
        
        d,b_x,y_o, b_y = gen_seq(L, batch = batch)

        b_x = b_x.view(batch, L, 10)              # reshape x to (batch, time_step, input_size)
        b_x = b_x.float()
        y_o = y_o.long()

        output = rnn(b_x)                               # rnn output
        
        loss = loss_func(output, y_o)                   # cross entropy loss
        if i % 500 == 0:
            print(f"Epoch {epoch} Step {i}")
            print(f"Loss is {loss}")
        optimizer.zero_grad()                           # clear gradients for this training step
        loss.backward()                                 # backpropagation, compute gradients
        optimizer.step()                                # apply gradients

        pred_y = torch.max(output, 1)[1].data.numpy().squeeze()
        train_accuracy += np.sum(pred_y == y_o.numpy()).sum()

    train_accuracy = float(train_accuracy)/(ITER * batch)
    ll.append(loss)
    test_accuracy = 0

    for i in range(ITER_EVAL):
        L = np.random.randint(low = 3, high = 10)
        
        d,b_x,y_o, b_y = gen_seq(L, batch = batch)

        b_x = b_x.view(batch, L, 10)              # reshape x to (batch, time_step, input_size)
        b_x = b_x.float()
        y_o = y_o.long()

        output = rnn(b_x)                               # rnn output
        
        loss = loss_func(output, y_o)                   # cross entropy loss

        pred_y = torch.max(output, 1)[1].data.numpy().squeeze()
        test_accuracy += np.sum(pred_y == y_o.numpy())
       

    test_accuracy = float(test_accuracy)/(ITER_EVAL * batch)
    acc.append(test_accuracy)
    print('Epoch: ', epoch, '| train loss: %.4f' % loss.data.numpy(), 
    '| train accuracy: %.4f' % train_accuracy, '| test accuracy: %.4f' % test_accuracy)

plt.plot(np.arange(EPOCH),acc, ll)
plt.legend(['Accuracy (test)','Loss (train)'])
plt.title(f"Accuracy-Loss versus Epochs for Hidden units {args.H}")
plt.savefig(f"logs/Q2-A-L-{args.H}.png")
plt.show()

L = np.random.randint(low = 3, high = 10)
d,b_x,y_o, b_y = gen_seq(L, batch = 1)

b_x = b_x.view(1, L, 10)              # reshape x to (batch, time_step, input_size)
b_x = b_x.float()
y_o = y_o.long()

output = rnn(b_x)                               # rnn output

loss = loss_func(output, y_o)                   # cross entropy loss

pred_y = torch.max(output, 1)[1].data.numpy().squeeze()
print(f"Input Sequence {d}")
print(f"Truth {y_o}")
print(f"Prediction {pred_y}")
