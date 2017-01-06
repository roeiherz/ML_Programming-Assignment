import numpy as np

from HW3.KernelSVM import KernelSVM
from SVM import SVM

__author__ = 'roeiherz_mosherabu'


class MultiClassSVM(object):
    """
    This class is our implementation using SGD, One Vs. ALL, as discussed in class.
    """

    def __init__(self, num_of_features=None, max_samples_size=None, num_of_classes=10, use_kernel=False,
                 kernel='quadratic'):
        """
        We initialize the weights and the bias
        :param num_of_features: number of features
        :param max_samples_size: maximum sample size
        :param num_of_classes: number of class
        :param use_kernel: use KernelSVM or LinearSVM
        :param kernel: type of kernel to use. default: quadratic
        """

        self._num_of_features = num_of_features
        self._max_samples_size = max_samples_size
        self._num_of_classes = num_of_classes

        if use_kernel:
            # Kernel SVM
            self._svm_lst = [KernelSVM(max_samples_size, kernel) for i in range(num_of_classes)]
        else:
            # Linear SVM
            self._svm_lst = [SVM(num_of_features) for i in range(num_of_classes)]

    def train(self, xs, labels, learning_rate, C=1, T=1000):
        """
        This function trains the MultiClassSVM algorithm.
        :param T: Iterations
        :param learning_rate: learning rate
        :param C: punishment parameter for SVM
        :param xs: array of samples
        :param labels: array of labels
        """

        for svm_ind in range(self._num_of_classes):
            # Take the labels per class
            # One vs All labels
            label_per_class = np.copy(labels)
            label_per_class[label_per_class != svm_ind] = -1
            label_per_class[label_per_class == svm_ind] = 1

            # Train the svm per class
            self._svm_lst[svm_ind].train(xs, label_per_class, learning_rate, C, T)

    def test(self, test, test_labels):
        """
        This function calculates the accuracy on the test sample
        :param test: test sample
        :param test_labels: test labels
        :return: accuracy
        """
        multi_svm_prediction = np.concatenate(
            [[self._svm_lst[svm_ind].predict(test)] for svm_ind in range(self._num_of_classes)], axis=0)
        prediction_labels = multi_svm_prediction.argmax(axis=0)
        error_lst = (test_labels != prediction_labels)
        error = np.sum(error_lst)

        return 1 - float(error) / np.size(prediction_labels)

    def get_svm_lst(self):
        """
        This function returns SVMs list
        :return: self._svm_lst
        """
        return self._svm_lst
