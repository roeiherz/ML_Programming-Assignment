import numpy as np

from KernelSVM import KernelSVM
from SVM import SVM

class MultiClassKernelSVM(object):
    """
    This class is our Multi Class Kernel implementation using SGD as discussed in class
    """

    def __init__(self, max_samples_size=None, num_of_features=None, num_of_classes=10, kernel='linear'):
        """
        We initialize the weights and the bias
        :param kernel: the kernel function
        """

        self._max_samples_size = max_samples_size
        self._num_of_classes = num_of_classes

        # Kernel SVM
        if kernel == "none":
            self._svm_lst = [SVM(num_of_features) for i in range(num_of_classes)]
        else:
            self._svm_lst = [KernelSVM(max_samples_size, kernel) for i in range(num_of_classes)]


    def train(self, xs, labels, learning_rate, C=1, T=1000):
        """
        This function trains the KernelSVM algorithm. It updates the misclassified xi list only when we are mistaking
        :param T: Iterations
        :param learning_rate: learning rate
        :param C: punishment parameter for SVM
        :param xs: array of samples
        :param labels: array of labels
        """

        # create empty sample
        self._empty_sample = np.zeros(np.shape(xs[0])[0])

        for index in range(1, T + 1):
            # Take specific sample x, label, learning_rate_per_itr and rand_sample ind
            rand_sample_indx = np.random.randint(0, np.shape(xs)[0])
            x = xs[rand_sample_indx]
            label = int(labels[rand_sample_indx])
            learning_rate_per_itr = float(learning_rate) / index

            true_label_predict = self._svm_lst[label].predict(x)
            per_class_predict = [self._svm_lst[i].predict(x) - true_label_predict for i in range(self._num_of_classes)]
            per_class_predict[label] = float("-inf")
            max_delta = max(per_class_predict)
            max_classes = [i for i, val in enumerate(per_class_predict) if val == max_delta]
            if max_delta > -1:
                # Update all wj's that gives max wj*x - wy*x --> incr -x
                for max_class in max_classes:
                    self._svm_lst[max_class].increment(-1 * x, learning_rate_per_itr, C)
                # Update wy  --> incr +x
                self._svm_lst[label].increment(x, learning_rate_per_itr, C)
                # Update others  --> incr 0
                for i in range(0, self._num_of_classes):
                    if i in max_classes or i == label:
                        continue
                    self._svm_lst[i].increment(self._empty_sample, learning_rate_per_itr, C)

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
