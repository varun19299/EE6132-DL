'''
Network defenition module
'''

import os
import numpy as np
import random

import activations

class MLP(object):
    '''
    Model for a MLP.
    '''

    def __init__(self, sizes=list(), activation='sigmoid',learning_rate=1.0, mini_batch_size=16,
                 epochs=10):
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
        self.weights = [np.array([0])] + [np.random.randn(y, x) for y, x in
                                          zip(sizes[1:], sizes[:-1])]

        # Input layer does not have any biases. self.biases[0] is redundant.
        self.biases = [np.random.randn(y, 1) for y in sizes]

        # Input layer has no weights, biases associated. Hence z = wx + b is not
        # defined for input layer. self.zs[0] is redundant.
        self._zs = [np.zeros(bias.shape) for bias in self.biases]

        # Training examples can be treated as activations coming out of input
        # layer. Hence self.activations[0] = (training_example).
        self._activations = [np.zeros(bias.shape) for bias in self.biases]

        self.mini_batch_size = mini_batch_size
        self.epochs = epochs
        self.eta = learning_rate

    def fit(self, training_data, validation_data=None, test_data=None):
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
            random.shuffle(training_data)
            mini_batches = [
                training_data[k:k + self.mini_batch_size] for k in
                range(0, len(training_data), self.mini_batch_size)]

            for mini_batch in mini_batches:
                nabla_b = [np.zeros(bias.shape) for bias in self.biases]
                nabla_w = [np.zeros(weight.shape) for weight in self.weights]

                # Iterate over a mini batch
                for x, y in mini_batch:
                    self._forward_prop(x)
                    delta_nabla_b, delta_nabla_w = self._back_prop(x, y)

                    # Update deltas
                    nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
                    nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]

                self.weights = [
                    w - (self.eta / self.mini_batch_size) * dw for w, dw in
                    zip(self.weights, nabla_w)]
                self.biases = [
                    b - (self.eta / self.mini_batch_size) * db for b, db in
                    zip(self.biases, nabla_b)]

            if validation_data:
                accuracy, precision, recall, F1_score = self.validate(validation_data)
                print(f'Epoch {epoch}, accuracy {accuracy}, precision {precision}, recall {recall}, F1_score {F1_score}')
            else:
                print(f'Processed epoch {epoch}')

        if test_data:
            accuracy, precision, recall, F1_score = self.validate(validation_data)
            print(f'Final test accuracy {accuracy}, precision {precision}, recall {recall}, F1_score {F1_score}')
        else:
            print(f'Processed epoch {epoch}')

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
        validation_results = [(self.predict(x) == y) for x, y in validation_data]
        accuracy= sum(result for result in validation_results)/len(validation_data)
        precision= sum()

        if confusion_matrix:
            confusion_matrix=np.zeros((10,10))

    def predict(self, x):
        """
        Run forward pass

        Args:
        * x : input array

        Returns:
        * y_cap: Predicted label.

        """

        self._forward_prop(x)
        return np.argmax(self._activations[-1])

    def _forward_prop(self, x):
        self._activations[0] = x
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

    def load(self, filename='model.npz'):
        """Prepare a neural network from a compressed binary containing weights
        and biases arrays. Size of layers are derived from dimensions of
        numpy arrays.

        Parameters
        ----------
        filename : str, optional
            Name of the ``.npz`` compressed binary in models directory.

        """
        npz_members = np.load(os.path.join(os.curdir, 'models', filename))

        self.weights = list(npz_members['weights'])
        self.biases = list(npz_members['biases'])

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
        """Save weights, biases and hyperparameters of neural network to a
        compressed binary. This ``.npz`` binary is saved in 'models' directory.

        Parameters
        ----------
        filename : str, optional
            Name of the ``.npz`` compressed binary in to be saved.

        """
        np.savez_compressed(
            file=os.path.join(os.curdir, 'models', filename),
            weights=self.weights,
            biases=self.biases,
            mini_batch_size=self.mini_batch_size,
            epochs=self.epochs,
            eta=self.eta
        )

