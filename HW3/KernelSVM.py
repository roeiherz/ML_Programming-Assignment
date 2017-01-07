import numpy as np

class KernelSVM(object):
    """
    This class is our Kernel implementation using SGD as discussed in class
    """

    def __init__(self, max_samples_size=None, num_of_classes=10, kernel='linear'):
        """
        We initialize the weights and the bias
        :param kernel: the kernel function
        """
        #max number of samples to train
        if max_samples_size is None:
            print "You must give a number for maximum sample size"
            exit()
        self._max_sample_size = max_samples_size

        #kernel method
        if kernel == 'quadratic':
            self._kernel = self._quad_kernel
        if kernel == 'linear':
            self._kernel = self._linear_kernel
        else:
            print "Kernel not supported"
            exit()

        #init local members
        self._misclassified_cnt = 0
        self._misclassified_coeffs = np.zeros(max_samples_size)
        self._misclassified_list = []

    def increment(self, x, learning_rate_step, C):
        """
        This function trains the KernelSVM algorithm. It updates the misclassified xi list only when we are mistaking
        :param T: Iterations
        :param learning_rate: learning rate
        :param C: punishment parameter for SVM
        :param xs: array of samples
        :param labels: array of labels
        """

        # Increment fields
        self._misclassified_coeffs[0:self._misclassified_cnt] *= (1 - learning_rate_step)
        self._misclassified_coeffs[self._misclassified_cnt] = learning_rate_step * C
        self._misclassified_list.append(x)
        self._misclassified_cnt += 1


    def get_misclassified_xis(self):
        """
        This function returns misclassified xi's field
        :return: self._misclassfied_list
        """
        return self._misclassified_list

    def predict(self, sample):
        """
        This function predict the label for a single sample
        :param sample: sample data
        :return: prediction value (float)
        """
        if self._misclassified_cnt == 0:
            if len(sample.shape) == 1:
                return 0
            else:
                return np.zeros(sample.shape[0])
        prediction_arr = self._misclassified_coeffs[:self._misclassified_cnt] * self._kernel(sample)
        return np.sum(prediction_arr, axis=len(prediction_arr.shape)-1)

    def _quad_kernel(self, xx):
        """
        This function will implements a quadratic SVM Kernel = (1+x*x')^2
        :param xx: numpy array which represents x'
        :return: matrix with size: (nof_mistakes, nof_xx)
        """

        if self._misclassified_cnt == 0:
            return 0

        dot_product = np.dot(np.array(self._misclassified_list), xx.transpose()) + 1
        return np.square(dot_product).transpose()

    def _linear_kernel(self, xx):
        """
        This function will implements a linear SVM Kernel = x*x'
        :param xx: numpy array which represents x'
        :return: matrix with size: (nof_mistakes, nof_xx)
        """

        if self._misclassified_cnt == 0:
            return 0

        dot_product = np.dot(np.array(self._misclassified_list), xx.transpose())
        return dot_product.transpose()

