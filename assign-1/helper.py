import matplotlib.pyplot as plt 
import numpy as np 
import os

def plot(losses,name="losses",epochs=10,labels=["train","validation","test"],path="logs"):
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

if __name__=="__main__":
    plot(np.arange(10))