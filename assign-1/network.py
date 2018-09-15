'''
Network defenition module
'''

import os
import numpy as np
import scipy.sparse as sp
import random, progressbar

import activations

class MLP(object):
    '''
    Model for a MLP.
    '''

    def __init__(self, sizes=list(), activation='sigmoid',learning_rate=1.0, mini_batch_size=64,
                 epochs=10,l2=0.0,l1=0.0):
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
        
        # Map activation to the function and derivative
        if activation in ['sigmoid','relu','tanh','softmax']:
            self.activation=activations.map_fn[activation]['function']
            self.activation_prime=activations.map_fn[activation]['derivative']

        self.num_layers = len(sizes)

        # First term corresponds to layer 0 (input layer). No weights enter the
        # input layer and hence self.weights[0] is redundant.
        self.weights = [np.zeros((1))] + [np.random.randn(y, x) for y, x in
                                          zip(sizes[1:], sizes[:-1])]

        # Input layer does not have any biases. self.biases[0] is redundant.
        self.biases = [np.random.randn(y, 1) for y in sizes]

        # regularisations
        self.l2=l2
        self.l1=l1

        # Input layer has no weights, biases associated. Hence z = wx + b is not
        # defined for input layer. self.zs[0] is redundant.
        self._zs = [np.zeros(bias.shape) for bias in self.biases]

        # Training examples can be treated as activations coming out of input
        # layer. Hence self.activations[0] = (training_example).
        self._activations = [np.zeros(bias.shape) for bias in self.biases]

        self.mini_batch_size = mini_batch_size
        self.epochs = epochs
        self.eta = learning_rate

    def fit(self, training_data, validation_data=[], test_data=[]):
        """

        Use SGD to train

        Args

        training_data : list of tuple
            A list of tuples of numpy arrays, ordered as (image, label).

        validation_data : list of tuple, optional
            Same as `training_data`, if provided, the network will display
            validation accuracy after each epoch.

        """
        for epoch in range(self.epochs):
            
            #shuffle train data
            random.shuffle(training_data)
            
            mini_batches = [
                training_data[k:k + self.mini_batch_size] for k in
                range(0, len(training_data), self.mini_batch_size)]

            # Progress Bar
            bar = progressbar.ProgressBar(maxval=len(mini_batches), \
            widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
            bar.start()

            epoch_loss=0

            # Printing start info
            print("Starting Epoch {epoch}, number of mini_batches {mini_batches}, mini_batch size {mini_batch_size}".\
            format(epoch=epoch,mini_batches=len(mini_batches),mini_batch_size=self.mini_batch_size))

            for mini_batch,count in zip(mini_batches,range(len(mini_batches))):
                nabla_b = [np.zeros(bias.shape) for bias in self.biases]
                nabla_w = [np.zeros(weight.shape) for weight in self.weights]

                loss=0
                # Iterate over a mini batch
                for x, y in mini_batch:
                    self._forward_prop(x)
                    delta_nabla_b, delta_nabla_w = self._back_prop(x, y)

                    # Update deltas
                    nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
                    nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]

                    loss+=self.log_likelihood_loss(y)

                self.weights = [
                    w \
                    - self.eta / self.mini_batch_size * dw \
                    - self.eta * self.l2/self.mini_batch_size *w \
                    - self.eta * self.l1 * np.sign(w)/self.mini_batch_size \
                    for w, dw in zip(self.weights, nabla_w)]

                self.biases = [
                    b \
                    - (self.eta / self.mini_batch_size) * db \
                    for b, db in zip(self.biases, nabla_b)]

                loss=loss/len(mini_batch)
                epoch_loss+=loss

                bar.update(count+1)
                #print(f'Mini Batch loss {loss}')

            bar.finish()
            epoch_loss=epoch_loss/(len(mini_batches))
            print('Epoch loss ',epoch_loss)

            if len(validation_data) :
                accuracy,cm, precision, recall, F1_score = self.validate(validation_data)
                print('Epoch {epoch}, cm {cm}, \n accuracy {accuracy}, \n precision {precision},\n  recall {recall}, \n F1_score {F1_score}'\
                .format(epoch=epoch,cm=cm,test_loss=test_loss,accuracy=accuracy,precision=precision,recall=recall,F1_score=F1_score ))
           
            if len(test_data) :
                accuracy, cm, precision, recall, F1_score = self.validate(test_data)

                test_loss=0
                for x,y in test_data:
                    self._forward_prop(x)
                    test_loss+=self.log_likelihood_loss(y)
                test_loss=test_loss/len(test_data)

                print('Epoch {epoch}, test loss {test_loss}, accuracy {accuracy}, precision {precision}, recall {recall}, F1_score {F1_score}'\
                .format(epoch=epoch,test_loss=test_loss,accuracy=accuracy,precision=precision,recall=recall,F1_score=F1_score ))

            print('Processed epoch', epoch)

    def validate(self, validation_data, confusion_matrix=False):
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
        validation_results = [(self.predict(data[0])[0] == data[1]) for data in validation_data]
        accuracy= sum(result for result in validation_results)/len(validation_data)
        
        y_pred=[self.predict(x)[0] for x,y in validation_data]
        cm=MLP.confusion_matrix(validation_data[:,1],y_pred)

        recall = np.diag(cm) / np.sum(cm, axis = 1)
        precision = np.diag(cm) / np.sum(cm, axis = 0)
        F1_score= 2*recall*precision/(recall+precision)

        return accuracy,cm,precision,recall,F1_score

    def predict(self, x):
        """
        Run forward pass

        Args:
        * x : input array

        Returns:
        * y_cap: Predicted label.

        """

        self._forward_prop(x)
        return np.argmax(self._activations[-1]), self._activations[-1]

    def _forward_prop(self, x):
        self._activations[0] = np.array(x).reshape((len(x),1))
        for i in range(1, self.num_layers):
            self._zs[i] = (
                self.weights[i].dot(self._activations[i - 1]) + self.biases[i]
            )
            self._activations[i] = self.activation(self._zs[i])

    def _back_prop(self, x, y):
        nabla_b = [np.zeros(bias.shape) for bias in self.biases]
        nabla_w = [np.zeros(weight.shape) for weight in self.weights]

        error = (self._activations[-1] - y) * self.activation_prime(self._zs[-1])
        nabla_b[-1] = error
        nabla_w[-1] = error.dot(self._activations[-2].transpose())

        for l in range(self.num_layers - 2, 0, -1):
            error = np.multiply(
                self.weights[l + 1].transpose().dot(error),
                self.activation_prime(self._zs[l])
            )
            nabla_b[l] = error
            nabla_w[l] = error.dot(self._activations[l - 1].transpose())

        return nabla_b, nabla_w

    def cross_entropy_loss(self, y):
        '''
        Used for sigmoid final layer
        '''
        return np.sum(np.nan_to_num(-y*np.log(self._activations[-1])-(1-y)*np.log(1-self._activations[-1])))

    def log_likelihood_loss(self, y):
        '''
        Used for softmax final layer
        '''
        return np.sum(np.dot(y, self.activation(self._activations[-1]).transpose()))

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

        self.l1 = float(npz_members['l1'])
        self.l2 = float(npz_members['l2'])

        # Bias vectors of each layer has same length as the number of neurons
        # in that layer. So we can build `sizes` through biases vectors.
        self.sizes = [b.shape[0] for b in self.biases]
        self.num_layers = len(self.sizes)

        # These are declared as per desired shape.
        self._zs = [np.zeros(bias.shape) for bias in self.biases]
        self._activations = [np.zeros(bias.shape) for bias in self.biases]

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
            mini_batch_size=self.mini_batch_size,
            epochs=self.epochs,
            eta=self.eta,
            l1=self.l1,
            l2=self.l2
        )

    @staticmethod
    def confusion_matrix(y_true, y_pred, labels=None, sample_weight=None):
        """Compute confusion matrix to evaluate the accuracy of a classification
        
        Args:
        * y_true : array, shape = [n_samples]
            Ground truth (correct) target values.
        * y_pred : array, shape = [n_samples]
            Estimated targets as returned by a classifier.
        * labels : array, shape = [n_classes], optional
            List of labels to index the matrix. This may be used to reorder
            or select a subset of labels.
            If none is given, those that appear at least once
            in ``y_true`` or ``y_pred`` are used in sorted order.
        * sample_weight : array-like of shape = [n_samples], optional
            Sample weights.
        
        Returns:
        * C : array, shape = [n_classes, n_classes]
            Confusion matrix
        
        """

        if labels is None:
            labels = np.arange(0,10,1)
        else:
            labels = np.asarray(labels)
            if np.all([l not in y_true for l in labels]):
                raise ValueError("At least one label specified must be in y_true")

        if sample_weight is None:
            sample_weight = np.ones(y_true.shape[0], dtype=np.int64)
        else:
            sample_weight = np.asarray(sample_weight)

        n_labels = labels.size
        label_to_ind = dict((y, x) for x, y in enumerate(labels))
        # convert yt, yp into index
        y_pred = np.array([label_to_ind.get(x, n_labels + 1) for x in y_pred])
        y_true = np.array([label_to_ind.get(x, n_labels + 1) for x in y_true])

        # intersect y_pred, y_true with labels, eliminate items not in labels
        ind = np.logical_and(y_pred < n_labels, y_true < n_labels)
        y_pred = y_pred[ind]
        y_true = y_true[ind]
        # also eliminate weights of eliminated items
        sample_weight = sample_weight[ind]

        # Choose the accumulator dtype to always have high precision
        if sample_weight.dtype.kind in {'i', 'u', 'b'}:
            dtype = np.int64
        else:
            dtype = np.float64

        CM = sp.coo_matrix((sample_weight, (y_true, y_pred)),
                        shape=(n_labels, n_labels), dtype=dtype,
                        ).toarray()

        return CM

