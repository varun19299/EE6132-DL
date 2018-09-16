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
import network

parser=argparse.ArgumentParser()
parser.add_argument("--question",help="Which question to display answers for.",required=True)
args=parser.parse_args()

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
    initial_lr=0.08
    final_lr=0.00008

    if args.question in ["1","2"]:
        model= network.MLP([784,1000,500,250,10])
        train_losses,val_losses,test_losses=model.fit(np.array(DATA['train']),validation,np.array(DATA['fold-4']),\
        epochs=epochs,\
        initial_lr=initial_lr,\
        final_lr=final_lr)
        helper.plot([train_losses,val_losses,test_losses],epochs=epochs,name="sigmoid")

    elif args.question=="3":
        epochs=5
        initial_lr=0.0008
        final_lr=0.000008
        variance=0.0001

        model= network.MLP([784,1000,500,250,10],activation="relu",\
        variance=variance)
        train_losses,val_losses,test_losses=model.fit(np.array(DATA['train']),validation,np.array(DATA['fold-4']),\
        epochs=epochs,\
        initial_lr=initial_lr,\
        final_lr=final_lr)
        helper.plot([train_losses,val_losses,test_losses],epochs=epochs,name="relu")

    elif args.question=="4":
        model= network.MLP([784,1000,500,250,10],activation="relu")
        train_losses,val_losses,test_losses=model.fit(np.array(DATA['train']),validation,np.array(DATA['fold-4']),\
        l2=0.1,\
        l1=0.01,\
        epochs=epochs,\
        initial_lr=initial_lr,\
        final_lr=final_lr)
        helper.plot([train_losses,val_losses,test_losses],epochs=epochs,name="relu")
    
    elif args.question=="5":
        pass
    elif args.question=="6":
        pass
    elif args.question=="7":
        pass
    else:
        print("Invalid question {}".format(args.question))

if __name__=='__main__':
    main()