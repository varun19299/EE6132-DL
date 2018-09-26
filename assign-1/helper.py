import matplotlib.pyplot as plt 
import numpy as np 
import os

def plot_loss(losses,name="losses",epochs=10,labels=["train","validation","test"],path="logs"):
    '''
    Plot train,val and test loss with epochs
    '''
    if not os.path.exists(path):
        os.mkdir(path)
    xx=np.arange(epochs)
    plt.plot(xx,losses[0],'r',xx,losses[1],'b',xx,losses[2],'g')
    plt.xlabel("Epochs")
    plt.ylabel("Train Loss -- cross entropy")
    plt.legend(labels)
    plt.title("Loss curve for {}".format(name))
    plt.savefig(os.path.join(path,name+".jpg"))
    plt.show()

def plot_accuracy(acc,name="acc",epochs=10,labels=["train","validation","test"],path="logs"):
    '''
    Plot train,val and test accuracies with epochs
    '''
    if not os.path.exists(path):
        os.mkdir(path)
    xx=np.arange(epochs)
    plt.plot(xx,acc[0],'r',xx,acc[1],'b',xx,acc[2],'g')
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend(labels)
    plt.title("Accuracy curve for {}".format(name))
    plt.savefig(os.path.join(path,name+".jpg"))
    plt.show()

if __name__=="__main__":
    l=[[2.2973767276037713,1.2334, 0.8888,0.66686, 0.555,0.533,0.4445,0.441],\
    [2.4973767276037713,1.3334, 0.9888,0.78686, 0.655,0.633,0.6445,0.651],\
    [2.4973767276013,1.334, 0.988,0.7866, 0.65,0.63,0.644,0.655]]
    plot(l,name="sigmoid",epochs=8)