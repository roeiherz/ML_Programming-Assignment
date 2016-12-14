import numpy as np

__author__ = 'roeiherz'


class Perceptron(object):

    def __init__(self, num_of_features=None):
        """
        We initialize the weights and the bias
        :param num_of_features: number of features
        """
        if num_of_features is None:
            print "You must give a number for the number of features"
        else:
            self._Weights = np.zeros(num_of_features)
            # No bias
            self.Bias = 0

    def _predict(self, x):
        """
        This function will predict
        :param x: input
        :return: 1 if its a positive prediction, else -1
        """

        if np.size(x) == np.size(self._Weights):
            res = np.dot(self._Weights, x)
        else:
            res = None
            print "Error dimensions: input dimension: {0}, weight: {1}".format(np.size(x), np.size(self._Weights))
            exit()

        if res > 0:
            return 1
        return -1

    def train(self, xs, labels):
        """
        This function trains the Perceptron algorithm. It updates the weights only when we are mistaking
        :param xs: array of samples
        :param labels: array of labels
        """

        for index in range(np.shape(xs)[0]):
            # Take specific sample x and label
            x = xs[index]
            label = labels[index]

            # Make the prediction
            predication = self._predict(x)
            if predication != label:
                self._Weights += label * x

    def test(self, test, test_labels):
        """
        This function calculates the accuracy on the test sample
        :param test: test sample
        :param test_labels: test labels
        :return: accuracy
        """

        prediction = np.dot(test, self._Weights)
        prediction[prediction >= 0] = 1
        prediction[prediction < 0] = -1
        error_lst = (test_labels != prediction)
        error = np.sum(error_lst)

        return 1 - float(error) / np.size(prediction)

    def get_weights(self):
        """
        This function returns weights field
        :return: self._Weights
        """
        return self._Weights

    def find_misclassified_samples(self, test, test_labels):
        """
        This function find the first misclassified sample in test set
        :param test: test sample
        :param test_labels: test labels
        :return: the first misclassified sample
        """

        prediction = np.dot(test, self._Weights)
        prediction[prediction >= 0] = 1
        prediction[prediction < 0] = -1
        error_lst = (test_labels != prediction)

        misclassified_indexs = np.argwhere(error_lst == 1)
        return test[misclassified_indexs], test_labels[misclassified_indexs]
