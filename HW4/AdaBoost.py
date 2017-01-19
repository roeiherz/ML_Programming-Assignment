import math
import numpy as np
# import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt

__author__ = 'roeiherz'


class AdaBoost:
    """
    This class is AdaBoost algorithm
    """

    def __init__(self, nof_features, T, m):
        """
        Init the following 3 params
        :param nof_features: Number of features
        :param T: Number of iterations
        :param m: maximum train size
        """

        # Init 3 fields member
        self._nof_features = nof_features
        # Nof iterations
        self._T = T
        # Size of samples
        self._m = m
        # This is the alpha coeff array per iteration
        self._alpha = np.zeros(T)
        # This is a distribution array
        self._dist_arr = np.ones(m) / m

        # This is the params for hypo per iteration
        #############################################
        # This is the theta array per iteration
        self._theta = np.zeros(T)
        # This is hypothesis 1 or -1 per iteration
        self._hypo_type = np.zeros(T)
        # This is a pixel array which is saves which is the pixel of the hypo
        self._pixel_arr = np.zeros(T)

    def train(self, xs, labels):
        """
        This function trains with AdaBoost algorithm.
        :param xs: array of samples
        :param labels: array of labels
        """

        best_hypo_type = 0
        best_theta = 0
        best_weighted_acc = 0
        best_pixel_idx = 0

        for t in range(self._T):
            for pixel_idx in range(self._nof_features):

                for hypo_type in [-1, 1]:
                    theta, weighted_acc = self._find_best_hypo(xs, pixel_idx, labels, hypo_type=hypo_type)

                    # Save the best hypo using weighted_acc
                    if best_weighted_acc < weighted_acc:
                        best_hypo_type = hypo_type
                        best_theta = theta
                        best_weighted_acc = weighted_acc
                        best_pixel_idx = pixel_idx

            epsilon = 1 - best_weighted_acc
            a = 0.5 * math.log(best_weighted_acc / epsilon)
            pixel_vals = xs[:, best_pixel_idx]
            predication_low_theta = (pixel_vals <= best_theta) * 1.0
            predication_after_theta = (pixel_vals > best_theta) * -1.0
            predication_theta = predication_low_theta + predication_after_theta
            predication_theta *= best_hypo_type
            self._dist_arr *= math.e ** (predication_theta * labels * a * -1.0)
            self._dist_arr /= np.sum(self._dist_arr)

            # Update the params each iteration
            self._update_params(t, a, best_theta, best_hypo_type, best_pixel_idx)

    def _find_best_hypo(self, xs, pixel_idx, labels, hypo_type):
        """
        This function implements finding best hypothesis
        :param xs: xs data
        :param pixel_idx: pixel index
        :param labels: labels array
        :param hypo_type: hypothesis type is 1 or -1 according to the ex. input
        :return:
        """

        # Sorted xs per pixel index (channel wise)
        sorted_indx = np.argsort(xs[:, pixel_idx])
        # This the currently smallest pixel val
        best_theta = xs[sorted_indx[0]][pixel_idx]

        pixel_vals = xs[:, pixel_idx]
        predication_low_theta = (pixel_vals <= best_theta) * 1.0
        predication_after_theta = (pixel_vals > best_theta) * -1.0
        # This is the prediction of theta
        predication_theta = predication_low_theta + predication_after_theta
        # The predication of theta is determined by the hypothesis type
        predication_theta *= hypo_type

        acc_arr = predication_theta * labels
        acc_arr[acc_arr < 0] = 0

        best_weighted_acc = np.dot(self._dist_arr, acc_arr)
        curr_weighted_acc = best_weighted_acc

        for sample_idx in sorted_indx:
            if pixel_vals[sample_idx] == pixel_vals[sorted_indx[0]]:
                continue
            curr_theta = xs[sample_idx][pixel_idx]
            curr_weighted_acc += labels[sample_idx] * self._dist_arr[sample_idx] * hypo_type
            if curr_weighted_acc > 1:
                print 'error'
            if curr_weighted_acc > best_weighted_acc:
                best_weighted_acc = curr_weighted_acc
                best_theta = curr_theta

        return best_theta, best_weighted_acc

    def _update_params(self, t, a, best_theta, best_hypo_type, best_pixel_idx):
        """
        This function updates the params
        :param t: iteration t
        :param a: alpha
        :param best_theta: the best theta
        :param best_hypo_type: the best hypothesis
        :param best_pixel_idx: the best pixel index related to the hypo
        """

        self._theta[t] = best_theta
        # This is hypothesis 1 or -1 per iteration
        self._hypo_type[t] = best_hypo_type
        # This is a pixel array which is saves which is the pixel of the hypo
        self._pixel_arr[t] = best_pixel_idx
        self._alpha[t] = a

    def test(self, test_data, test_labels, iterations):
        """
        This function performs testing
        :param test_data: test data
        :param test_labels: test label
        :param iterations: Number of iterations
        :return:
        """

        predictions = self._predict(test_data, iterations)
        acc_arr = predictions * test_labels
        acc_arr[acc_arr < 0] = 0
        acc = np.sum(acc_arr) / test_data.shape[0]
        return acc

    def _predict(self, test_data, iterations):
        """
        This function returns an array of predictions
        :param test_data: test data
        :param iterations: number of iterations
        :return: total prediction
        """

        total_predication = np.zeros(test_data.shape[0])
        for t in range(iterations):
            pixel_idx = self._pixel_arr[t]
            pixel_vals = test_data[:, pixel_idx]
            predication_low_theta = (pixel_vals <= self._theta[t]) * 1.0
            predication_after_theta = (pixel_vals > self._theta[t]) * -1.0
            # This is the prediction of theta
            predication_theta = predication_low_theta + predication_after_theta
            # The predication of theta is determined by the hypothesis type
            predication_theta *= self._hypo_type[t]
            total_predication += self._alpha[t] * predication_theta

        return np.sign(total_predication)
