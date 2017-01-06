import numpy as np

__author__ = 'roeiherz'


class KernelSVM(object):
    """
    This class is our Kernel implementation using SGD as discussed in class
    """

    def __init__(self, max_samples_size=None, kernel='linear'):
        """
        We initialize the weights and the bias
        :param kernel: the kernel function
        """

        if kernel == 'quadratic':
            self._kernel = self._quad_kernel

        if kernel == 'linear':
            self._kernel = self._linear_kernel

        self._misclassified_cnt = 0

        if max_samples_size is None:
            print "You must give a number for maximum sample size"
            exit()
        else:
            self._ai = np.zeros(max_samples_size)
            self._xi_lst = []

    def train(self, xs, labels, learning_rate, C=1, T=1000):
        """
        This function trains the KernelSVM algorithm. It updates the misclassified xi list only when we are mistaking
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

            if label * self.predict(x) < 1:
                # Update upon fail
                self._ai[0:self._misclassified_cnt] *= (1 - learning_rate_per_itr)
                self._ai[self._misclassified_cnt] = learning_rate_per_itr * C * label
                self._xi_lst.append(x)
                self._misclassified_cnt += 1

    def test(self, test, test_labels):
        """
        This function calculates the accuracy on the test sample
        :param test: test sample
        :param test_labels: test labels
        :return: accuracy
        """

        prediction = self.predict(test)
        prediction[prediction >= 0] = 1
        prediction[prediction < 0] = -1
        error_lst = (test_labels != prediction)
        error = np.sum(error_lst)

        return 1 - float(error) / np.size(prediction)

    def get_misclassified_xis(self):
        """
        This function returns misclassified xi's field
        :return: self._xi_lst
        """
        return self._xi_lst

    def predict(self, sample):
        """
        This function predict the label for a single sample
        :param sample: sample data
        :return: prediction value (float)
        """

        if self._misclassified_cnt != len(self._xi_lst):
            print 'Error between the size of the np array and list'
        prediction_arr = self._ai[:self._misclassified_cnt] * self._kernel(sample)
        return np.sum(prediction_arr, axis=len(prediction_arr.shape)-1)

    def _quad_kernel(self, xx):
        """
        This function will implements a quadratic SVM Kernel = (1+x*x')^2
        :param xx: numpy array which represents x'
        :return: matrix with size: (nof_mistakes, nof_xx)
        """

        if self._misclassified_cnt == 0:
            return 0

        dot_product = np.dot(np.array(self._xi_lst), xx.transpose()) + 1
        return np.square(dot_product).transpose()

    def _linear_kernel(self, xx):
        """
        This function will implements a linear SVM Kernel = x*x'
        :param xx: numpy array which represents x'
        :return: matrix with size: (nof_mistakes, nof_xx)
        """

        if self._misclassified_cnt == 0:
            return 0

        dot_product = np.dot(np.array(self._xi_lst), xx.transpose())
        return dot_product.transpose()

