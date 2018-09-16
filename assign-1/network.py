'''
Network defenition module
'''

import os
import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt
import random

from tqdm import trange

import activations
from confusion_matrix import *
import helper

class MLP(object):
    '''
    Model for a MLP.
    '''

    def __init__(self, sizes=list(), activation='sigmoid',variance=0.02):
        """
        
        Initialize a Neural Network model.

        Args

        sizes : list, optional
            A list of integers specifying number of neurns in each layer. Not
            required if a pretrained model is used.

        learning_rate : float, optional
            Learning rate for gradient descent optimization. Defaults to 1.0

        mini_batch_size : int, optional
            Size of each mini batch of training examples as used by Stochastic
            Gradient Descent. Denotes after how many examples the weights
            and biases would be updated. Default size is 16.

        """

        # Input layer is layer 0, followed by hidden layers layer 1, 2, 3...
        self.sizes = sizes
        self.num_layers = len(sizes)
        # Map activation to the function and derivative
        if activation in ['sigmoid','relu','tanh','softmax']:
            self.activation=activations.map_fn[activation]['function']
            self.activation_prime=activations.map_fn[activation]['derivative']

        # Weights and Biases
        self.weights =  [variance*np.random.randn(y, x) for y, x in zip(sizes[1:], sizes[:-1])]
        self.biases = [variance*np.random.randn(y, 1) for y in sizes[1:]]
    
        # Velocities
        self.v=[variance*np.random.randn(*weights.shape) for weights in self.weights ]
        self.vb = [variance*np.random.randn(*bias.shape) for bias in self.biases]


    def fit(self, training_data, validation_data=[], test_data=[],initial_lr=0.08,final_lr=0.0008,momentum=0.9, mini_batch_size=64,
                 epochs=10,l2=0.0,l1=0.0):
        """
        Use SGD to train

        Args

        training_data : list of tuple
            A list of tuples of numpy arrays, ordered as (image, label).

        validation_data : list of tuple, optional
            Same as `training_data`, if provided, the network will display
            validation accuracy after each epoch.

        """

        self.mini_batch_size = mini_batch_size
        self.epochs = epochs
        self.eta = initial_lr
        self.mu=momentum

        # regularisations
        self.l2=l2
        self.l1=l1

        # Loss per epoch
        train_losses=[]
        val_losses=[]
        test_losses=[]

        lr_scheduler=np.linspace(initial_lr,final_lr,self.epochs)
        mu_scheduler=np.array([momentum]*2)
        mu_scheduler=np.append(mu_scheduler,np.linspace(momentum,0.99,self.epochs-2))
        print(mu_scheduler)

        for epoch,lr,mu in zip(range(self.epochs),lr_scheduler,mu_scheduler):

            # Linear lr scheduler
            self.eta=lr
            self.mu=mu

            # shuffle train data
            random.shuffle(training_data)
            
            mini_batches = [
                training_data[k:k + self.mini_batch_size] for k in
                range(0, len(training_data), self.mini_batch_size)]
            
            #mini_batches=mini_batches[:2]
            epoch_loss=0

            # Printing start info
            print("Starting Epoch {epoch}, number of mini_batches {mini_batches}, mini_batch size {mini_batch_size}".\
            format(epoch=epoch,mini_batches=len(mini_batches),mini_batch_size=self.mini_batch_size))

            # tqdm
            t=trange(len(mini_batches))
            for mini_batch,count in zip(mini_batches,t):
                nabla_b = [np.zeros(bias.shape) for bias in self.biases]
                nabla_w = [np.zeros(weight.shape) for weight in self.weights]

                # Mini batch loss
                loss=0; accuracy=0.0;
                # Iterate over a mini batch
                for x, y in mini_batch:
                    a=self._forward_prop(x)
                    accuracy+=int(np.argmax(a)==np.where(y==1)[0][0])

                    #print("Predi",self.predict(x))
                    #print("True",np.where(y==1)[0][0])
                    delta_nabla_b, delta_nabla_w = self._back_prop(x, y)

                    # Update deltas
                    nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
                    nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]

                    loss+=self.cross_entropy_loss(a,y)

                #print([dw for dw in nabla_w])

                self.v = [
                    v*self.mu
                    - self.eta/self.mini_batch_size * dw \
                    - self.eta * self.l2/self.mini_batch_size *w \
                    - self.eta * self.l1 * np.sign(w)/self.mini_batch_size \
                    for v,w, dw in zip(self.v,self.weights, nabla_w)]

                #print(self.v)
                self.weights = [
                    w + v
                    for w, v in zip(self.weights, self.v)]

                self.vb = [
                    vb*self.mu \
                    - (self.eta / self.mini_batch_size) * db \
                    for vb, db in zip(self.vb, nabla_b)]

                self.biases = [
                    b + vb \
                    for b, vb in zip(self.biases, self.vb)]

                loss=loss/len(mini_batch)
                accuracy=accuracy/len(mini_batch)
                epoch_loss+=loss

                t.set_description('Loss = {loss} Accuracy= {accuracy}'.format(loss=loss,accuracy=accuracy))

            epoch_loss=epoch_loss/(len(mini_batches))
            print('Epoch loss ',epoch_loss)
            train_losses.append(epoch_loss)

            if len(validation_data) :
                accuracy,cm, precision, recall, F1_score = self.validate(validation_data)
                
                val_losses.append(self.measure_loss(validation_data))

                print('Epoch {epoch}, \ncm {cm}, \n accuracy {accuracy}, \n precision {precision},\n  recall {recall}, \n F1_score {F1_score} \n'\
                .format(epoch=epoch,cm=cm,accuracy=accuracy,precision=precision,recall=recall,F1_score=F1_score ))
           
            if len(test_data) :
                accuracy, cm, precision, recall, F1_score = self.validate(test_data)

                test_losses.append(self.measure_loss(test_data))

                print('Epoch {epoch}, test loss {test_loss}, accuracy {accuracy}, precision {precision}, recall {recall}, F1_score {F1_score}\n'\
                .format(epoch=epoch,test_loss=test_losses,accuracy=accuracy,precision=precision,recall=recall,F1_score=F1_score ))

        return [train_losses,val_losses,test_losses]

    def validate(self, validation_data):
        """
        Validate the MLP on provided validation data, via a accuracy metric.

        Args:
        * validation_data : list of tuple

        Returns :
        * Accuracy 
        * Precision
        * Recall
        * F1 Score

        """
        y_pred=[self.predict(data[0]) for data in validation_data]
        validation_results = [pred == np.where(data[1]==1)[0][0] for pred,data in zip(y_pred,validation_data)]
        #validation_results = [(self.predict(data[0]) == np.where(data[1]==1)[0][0]) for data in validation_data]
        accuracy= sum(validation_results)/len(validation_data)
        
        # a scalar, not one hot
        
        cm=confusion_matrix(validation_data[:,1],y_pred)

        recall = np.diag(cm) / np.sum(cm, axis = 1)
        precision = np.diag(cm) / np.sum(cm, axis = 0)
        F1_score= 2*recall*precision/(recall+precision)

        return accuracy,cm,precision,recall,F1_score

    def measure_loss(self,data):
        '''
        Find total loss on a dataset.

        Returns:
        * float: loss
        '''
        loss=0
        for x,y in data:
            a=self._forward_prop(x)
            loss+=self.cross_entropy_loss(a,y)
        loss=loss/len(data)

    def predict(self, x):
        """
        Run forward pass

        Args:
        * x : input array

        Returns:
        * y_cap: Predicted label.
        """
        a=self._forward_prop(x)
        return np.argmax(a)

    def _forward_prop(self, x):
        '''
        RUn forward prop.
        '''
        a = np.array(x).reshape((len(x),1))
        for count, b, w in zip(range(self.num_layers-1),self.biases, self.weights):
            if count==self.num_layers-2:
                a = activations.softmax(np.dot(w, a)+b)
            else:
                a = self.activation(np.dot(w, a)+b)
        return a

    def _back_prop(self, x, y):
        """
        Compute gradients of Cost
        
        Returns:
        * (nabla_b, nabla_w) representing the
        gradient for the cost function C_x.  
        
        nabla_b and nabla_w are similar
        to self.biases and self.weights.
        """

        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        # feedforward
        activation = np.array(x).reshape((len(x),1))

        # list to store all the activations, layer by layer
        a_ss = [activation] 
        
        # list to store all the z vectors, layer by layer
        zs = [] 

        count=0
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation)+b
            zs.append(z)
            if count==self.num_layers-2:
                activation=activations.softmax(z)
            else:
                activation = self.activation(z)
            a_ss.append(activation)
            count+=1

        # backward pass
        delta = self.cost_derivative(a_ss[-1], y) * activations.softmax_prime(zs[-1])

        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, a_ss[-2].transpose())
        
        for l in range(2, self.num_layers):
            delta = np.dot(self.weights[-l+1].transpose(), delta) * self.activation_prime(zs[-l])
            #print(delta)
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, a_ss[-l-1].transpose())

        return (nabla_b, nabla_w)

    def cost_derivative(self,a,y):
        '''
        Compute cost derivative
        '''
        return a-y

    def cross_entropy_loss(self,a, y):
        '''
        Used for sigmoid final layer
        '''
        def _safe_ln(x, minval=0.0000000001):
            return np.log(x.clip(min=minval))
        return -np.sum(y*_safe_ln(a))/y.shape[1]

    def log_likelihood_loss(self,a, y):
        '''
        Used for softmax final layer
        '''
        return np.sum(np.dot(y, self.activation(a).transpose()))

    def visualise(self,test_data):
        '''
        Visualise some test_data
        '''

        for x,y in test_data[:20]:
            a=self._forward_prop(x)
            y_pred=np.argmax(a)
            y_label=np.where(y==1)[0][0]

            x=np.array(x).reshape(28,28)
            plt.imshow(x,cmap="grey")
            print(f" Prediction {y_pred} Top 3 Probabilites {np.sort(a)[-3:-1:-1]} Truth {y_label}")

    def load(self, filename='model.npz'):
        """
        Prepare a neural network from a compressed binary containing weights
        and biases arrays. Size of layers are derived from dimensions of
        numpy arrays.

        Args:
        * filename : str, optional
            Name of the ``.npz`` compressed binary in models directory.

        """
        npz_members = np.load(os.path.join(os.curdir, 'models', filename))

        self.weights = list(npz_members['weights'])
        self.biases = list(npz_members['biases'])
        self.v = list(npz_members['velocities'])
        self.vb = list(npz_members['velocity_b'])

        self.l1 = float(npz_members['l1'])
        self.l2 = float(npz_members['l2'])

        # Bias vectors of each layer has same length as the number of neurons
        # in that layer. So we can build `sizes` through biases vectors.
        self.sizes = [b.shape[0] for b in self.biases]
        self.num_layers = len(self.sizes)

        # Other hyperparameters are set as specified in model. These were cast
        # to numpy arrays for saving in the compressed binary.
        self.mini_batch_size = int(npz_members['mini_batch_size'])
        self.epochs = int(npz_members['epochs'])
        self.eta = float(npz_members['eta'])

    def save(self, filename='model.npz'):
        """
        Save weights, biases and hyperparameters of neural network to a
        compressed binary. This ``.npz`` binary is saved in 'models' directory.

        Args:
        filename : str, optional
            Name of the ``.npz`` compressed binary in to be saved.

        """
        np.savez_compressed(
            file=os.path.join(os.curdir, 'models', filename),
            weights=self.weights,
            biases=self.biases,
            velocities=self.v,
            velocity_b=self.vb,
            mini_batch_size=self.mini_batch_size,
            epochs=self.epochs,
            eta=self.eta,
            l1=self.l1,
            l2=self.l2
        )

