'''
Confusion matrix code

'''

import scipy.sparse as sp
import numpy as np

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

        y_true=[np.where(r==1)[0][0] for r in y_true]
        
        CM = sp.coo_matrix((sample_weight, (y_true, y_pred)),
                        shape=(len(labels), len(labels)), dtype=np.float64,
                        ).toarray()

        return CM