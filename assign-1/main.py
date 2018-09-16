'''
Author: Varun Sundar
EE6123

Assignment 1
'''

# Standard libs
import argparse
import numpy as np

# Modules
import download_mnist, helper
from transform import *
import network

parser=argparse.ArgumentParser()
parser.add_argument("--question",help="Which question to display answers for.",required=True)
args=parser.parse_args()

def run_stats(mlp,DATA,tag):
    '''
    Run five fold statistics on cross val.
    '''
    for key in ['fold-{f}'.format(f=f) for f in range(4)]:
        data=DATA[key]
        mlp.pretty_stats(data,name=tag+"-cross_val-"+key)

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
    validation=[]
    for key in ['fold-{f}'.format(f=f) for f in range(4)]:
        validation+=DATA[key]
    validation=np.array(validation)

    epochs=8
    initial_lr=8e-3
    final_lr=8e-6

    if args.question in ["1","2","5"]:
        model= network.MLP([784,1000,500,250,10])
        train_losses,val_losses,test_losses=model.fit(np.array(DATA['train']),validation,np.array(DATA['fold-4']),\
        epochs=epochs,\
        initial_lr=initial_lr,\
        final_lr=final_lr)
        print(val_losses,test_losses)

        helper.plot([train_losses,val_losses,test_losses],epochs=epochs,name="sigmoid")
        run_stats(model,DATA,tag="sigmoid")

    elif args.question=="3":
        epochs=4
        initial_lr=8e-2
        final_lr=8e-6
        variance=0.0001

        model= network.MLP([784,1000,500,250,10],activation="relu",\
        variance=variance)
        train_losses,val_losses,test_losses=model.fit(np.array(DATA['train']),validation,np.array(DATA['fold-4']),\
        epochs=epochs,\
        initial_lr=initial_lr,\
        final_lr=final_lr)
        print(val_losses,test_losses)

        helper.plot([train_losses,val_losses,test_losses],epochs=epochs,name="relu")
        run_stats(model,DATA,tag="relu")

    elif args.question=="4":
        variance=0.0001
        train_data=noise_addition(DATA['train'],sigma=1e-3)

        model= network.MLP([784,1000,500,250,10],activation="relu",variance=variance)
        train_losses,val_losses,test_losses=model.fit(np.array(train_data),validation,np.array(DATA['fold-4']),\
        l2=0.1,\
        l1=0.01,\
        epochs=epochs,\
        initial_lr=initial_lr,\
        final_lr=final_lr)

        helper.plot([train_losses,val_losses,test_losses],epochs=epochs,name="relu_regularised")
        run_stats(model,DATA,tag="relu_regularised")
    
    elif args.question=="6":
        pass
    elif args.question=="7":
        pass
    else:
        print("Invalid question {}".format(args.question))

if __name__=='__main__':
    main()