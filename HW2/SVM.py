import numpy as np

__author__ = 'roeiherz'


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
        else:
            self._Weights = np.zeros(num_of_features)
            # No bias
            self.Bias = 0

    def train(self, xs, labels, learning_rate, C=1, T=1000):
        """
        This function trains the Perceptron algorithm. It updates the weights only when we are mistaking
        :param T: Iterations
        :param learning_rate: learning rate
        :param C: punishment parameter for SVM
        :param xs: array of samples
        :param labels: array of labels
        """

        for index in range(1, T + 1):
            # Take specific sample x, label, learning_rate_per_itr and rand_sample ind
            rand_sample_indx = np.random.randint(0, np.shape(xs)[0])
            x = xs[rand_sample_indx]
            label = labels[rand_sample_indx]
            learning_rate_per_itr = float(learning_rate) / index

            if label * np.dot(self._Weights, x) < 1:
                # Update upon fail
                self._Weights = (1 - learning_rate_per_itr) * self._Weights + learning_rate_per_itr * C * label * x

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