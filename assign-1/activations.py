"""
Helper module to provide activation to network layers.
Four types of activations with their derivates are available:

- Sigmoid
- Softmax
- Tanh
- ReLU
"""

import numpy as np


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def sigmoid_prime(z):
    return sigmoid(z) * (1 - sigmoid(z))


def softmax(z):
    return np.exp(z) / np.sum(np.exp(z))

def softmax_prime(z):
    return softmax(z) * (1 - softmax(z))


def tanh(z):
    return np.tanh(z)

def tanh_prime(z):
    return 1 - tanh(z) * tanh(z)


def relu(z):
    return np.maximum(z, 0)

def relu_prime(z):
    return (z > 0).astype(float)

map_fn={"sigmoid":{"function": sigmoid,"derivative": sigmoid_prime},\
        "relu":{"function": relu, "derivative": relu_prime},\
        "tanh":{"function": tanh, "derivative": tanh_prime},\
        "softmax":{"function": softmax, "derivative": softmax_prime}}
