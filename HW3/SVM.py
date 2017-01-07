import numpy as np
import math


class SVM(object):
    """
    This class is our implementation using SGD as discussed in class
    """

    def __init__(self, num_of_features=None):
        """
        We initialize the weights and the bias
        :param num_of_features: number of features
        """

        if num_of_features is None:
            print "You must give a number for the number of features"
            exit()
        else:
            self._Weights = np.zeros(num_of_features)
            # No bias
            self.Bias = 0

    def get_weights(self):
        """
        This function returns weights field
        :return: self._Weights
        """
        return self._Weights

    def predict(self, samples):
        """
        This function predicts the label for each samples
        :param samples: samples data
        :return: label
        """
        return np.dot(samples, self._Weights)

    def update(self, x, learning_rate_step, C):
        """
        This function trains the KernelSVM algorithm. It updates the misclassified xi list only when we are mistaking
        :param learning_rate_step: learning rate
        :param C: punishment parameter for SVM
        :param x: sample
        """

        # Increment fields
        if x is None:
            self._Weights = (1 - learning_rate_step) * self._Weights
        else:
            self._Weights = (1 - learning_rate_step) * self._Weights + learning_rate_step * C * x

