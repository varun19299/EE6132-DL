'''
Author: Varun Sundar
EE6123

Assignment 1
'''

# Standard libs
import argparse
import numpy as np

# Modules
import download_mnist
import network

parser=argparse.ArgumentParser()

parser.add_argument("--activation",help="Which activation to use",)

def main():
    '''
    Main block

    Steps:
    1. Pull data (MNIST)
    2. Initialise network
    3. Train network
    4. Save weights
    '''

    DATA=download_mnist.load_mnist()

    model= network.MLP([784,1000,500,250,10])

    validation=[]
    for key in ['fold-{f}'.format(f=f) for f in range(4)]:
        validation+=list(DATA[key])
    validation=np.array(validation)

    model.fit(np.array(list(DATA['train'])),validation,np.array(list(DATA['fold-4'])))


if __name__=='__main__':
    main()