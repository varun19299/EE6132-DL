'''
Q2 Assignment 4
'''

import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
import argparse
import os
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("--L", type = int, default =5 ,help = "Sequence length")
parser.add_argument("--loss_type", default ="MSE", choices = ["CE", "MSE"], help = "Loss to use; MSE or Cross Entropy")
args = parser.parse_args()

LR = 0.01
EPOCH = 10
ITER = 1000
ITER_EVAL = 20
batch = 32

HIDDEN = 10

L = args.L

loss_type = args.loss_type # Or CE


def to_binary(x, L):
    '''
    Convert x to binary in reverse.

    x can go upto 2**L - 1

    size L + 1
    '''

    bin_x = str(bin(x))
    arr_x = np.array(list(bin_x[2:]), dtype = int)
    arr_x = arr_x[::-1]

    arr = np.zeros((L+1))
    arr[0: len(arr_x)] = arr_x

    return arr

def gen_seq(L, batch = 1):
    '''
    Generate a sequence of 0-10, length L.
    And output as an index (default 0)

    Return as [batch, L + 1, 2]; [batch, L + 1]
    '''

    x = np.zeros((batch, L + 1,2))
    y = np.zeros((batch, L + 1))

    for b_ in range(batch):

        c = np.random.randint(low = 0, high = 2**L )
        a = np.random.randint(low = 0, high = c + 1)
        b = c - a

        str_a = to_binary(a,L)
        str_b = to_binary(b,L)
        str_c = to_binary(c,L)

        x[b_, :, 0] = str_a 
        x[b_, : ,1] = str_b

        y[b_,:] = str_c
    
    x = torch.tensor(x, dtype=torch.int)
    y = torch.tensor(y, dtype=torch.int)

    return x,y


class LSTM(nn.Module):
    def __init__(self, input_size = 2,
            hidden_size = HIDDEN):
        super(LSTM, self).__init__()

        self.lstm = nn.LSTM(         # if use nn.RNN(), it hardly learns
            input_size=input_size,
            hidden_size=hidden_size,         # rnn hidden unit
            num_layers=1,           # number of rnn layer
            batch_first=True,       # input & output will has batch size as 1s dimension. e.g. (batch, time_step, input_size)
        )

        self.out = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # x shape (batch, time_step, input_size)
        # r_out shape (batch, time_step, output_size)
        # h_n shape (n_layers, batch, hidden_size)
        # h_c shape (n_layers, batch, hidden_size)
        r_out, (h_n, h_c) = self.lstm(x, None)   # None represents zero initial hidden state

        # choose r_out at the last time step
        # out = F.sigmoid(self.out(r_out[:, -1, :]))
        out = F.sigmoid(self.out(r_out))
        return out

rnn = LSTM()
print(rnn)

optimizer = torch.optim.Adam(rnn.parameters(), lr=LR)   # optimize all cnn parameters
if loss_type == "MSE":
    loss_func = nn.MSELoss()
else: 
    loss_func = nn.NLLLoss()                       # the target label is not one-hotted


train_acc = []
test_acc = []
train_ll = []
test_ll = []

# training and testing
for epoch in range(EPOCH):
    print(f"\n\n----\n\nEpoch {epoch}")
    train_accuracy = 0

    for i in range(ITER):
        
        b_x , b_y = gen_seq(L, batch = batch)

        # b_x = b_x.view(batch, L, 10)              # reshape x to (batch, time_step, input_size)
        b_x = b_x.float()
        b_y = b_y.long()

        output = rnn(b_x)                               # rnn output
        
        if loss_type != "MSE":
            b_y2 = b_y.numpy().copy()
            # b_y2 = np.argmax(b_y2, axis=1)

            b_y2 = torch.Tensor(b_y2)

            b_y2 = b_y2.long()
        else:
            b_y2 = b_y.float()

        loss = loss_func(output[:,:,0], b_y2 )                  
        
        optimizer.zero_grad()                           # clear gradients for this training step
        loss.backward()                                 # backpropagation, compute gradients
        optimizer.step()                                # apply gradients

        t = torch.Tensor([0.5])
        pred_y = (output > t).float() * 1

        pred_y = pred_y.numpy()[:,:,0]
        pred_y = pred_y.dot(2**np.arange(pred_y.shape[1]))

        b_y = b_y.numpy()
        b_y = b_y.dot(2**np.arange(b_y.shape[1]))

        if i % 500 == 0:
            print(f"Epoch {epoch} Step {i}")
            print(f"Loss is {loss}")
            print(pred_y[0])
            print(b_y[0])

        train_accuracy += np.sum(pred_y == b_y)

    train_accuracy = float(train_accuracy)/(ITER * batch)
    train_ll.append(loss) 
    train_acc.append(train_accuracy)

    test_accuracy = 0

    for i in range(ITER_EVAL):
        
        b_x, b_y = gen_seq(L, batch = batch)

        # b_x = b_x.view(batch, L , 2)              # reshape x to (batch, time_step, input_size)
        b_x = b_x.float()
        b_y = b_y.long()

        output = rnn(b_x)                               # rnn output

        if loss_type != "MSE":
            b_y2 = b_y.numpy().copy()
            b_y2 = np.argmax(b_y2, axis=1)

            b_y2 = torch.Tensor(b_y2)

            b_y2 = b_y2.long()
        else:
            b_y2 = b_y.float()

        test_loss = loss_func(output[:,:,0], b_y2 )   
        
        t = torch.Tensor([0.5])
        pred_y = (output > t).float() * 1

        pred_y = pred_y.numpy()[:,:,0]
        pred_y = pred_y.dot(2**np.arange(pred_y.shape[1]))

        b_y = b_y.numpy()
        b_y = b_y.dot(2**np.arange(b_y.shape[1]))

        test_accuracy += np.sum(pred_y == b_y)

    test_accuracy = float(test_accuracy)/(ITER_EVAL * batch)

    test_ll.append(test_loss) 
    test_acc.append(test_accuracy)

    print('Epoch: ', epoch, '| train loss: %.4f' % loss.data.numpy(), '| train loss: %.4f' % test_loss.data.numpy(),
    '| train accuracy: %.4f' % train_accuracy, '| test accuracy: %.4f' % test_accuracy)

# Average out error
av_acc = []
for t_L in range(1,21):

    b_x, b_y = gen_seq(t_L, batch = 100)

    b_x = b_x.float()
    b_y = b_y.long()

    output = rnn(b_x)                               # rnn output
    
    t = torch.Tensor([0.5])
    pred_y = (output > t).float() * 1

    pred_y = pred_y.numpy()[:,:,0]
    pred_y = pred_y.dot(2**np.arange(pred_y.shape[1]))

    b_y = b_y.numpy()
    b_y = b_y.dot(2**np.arange(b_y.shape[1]))

    test_accuracy = np.sum(pred_y == b_y)/ 100
    av_acc.append(test_accuracy)
    print(f"Average error | Length of seq {t_L} | test accuracy over 100 samples {test_accuracy}")

if not os.path.exists("logs"):
    os.mkdir("logs")

plt.plot(np.arange(EPOCH),train_acc, test_acc)
plt.title(f"Accuracy-versus-Epochs-{args.L}-{loss_type}.png")
plt.save(f"logs/Accuracy-{args.L}-{loss_type}.png")
plt.show()

plt.plot(np.arange(EPOCH),train_ll, test_ll)
plt.title(f"Loss-versus-Epochs-{args.L}-{loss_type}.png")
plt.save(f"logs/Loss-{args.L}-{loss_type}.png")
plt.show()

plt.plot(np.arange(20)+1, av_acc)
plt.title(f"Average accuracy versus sequece length for loss {loss_type}")
plt.save(f"logs/AV-Loss-{args.L}-{loss_type}.png")
plt.show()