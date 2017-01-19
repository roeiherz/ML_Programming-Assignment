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
        # This is the theta array per iteration
        self._theta = np.zeros(T)
        # This is hypothesis 1 or 2 per iteration
        self._hypo_type = np.zeros(T)
        # This is a distribution array
        self._dist_arr = np.ones(m) / m

    def train(self, xs, labels):
        """
        This function trains with AdaBoost algorithm.
        :param xs: array of samples
        :param labels: array of labels
        """

        best_hypo_type = 0
        best_tetha = 0
        best_weighted_acc = 0
        best_pixel_idx = 0

        for t in range(self._T):
            for pixel_idx in range(self._nof_features):
                hypo_type, teta, weighted_acc = self._find_best_hypo(xs, pixel_idx, labels)

                # Save the best hypo using weighted_acc
                if best_weighted_acc >= weighted_acc:
                    best_hypo_type = hypo_type
                    best_tetha = teta
                    best_weighted_acc = weighted_acc
                    best_pixel_idx = pixel_idx

            epsilon = 1 - best_weighted_acc
            a = 0.5 * math.log(best_weighted_acc / epsilon)
            pixel_vals = xs[:, best_pixel_idx]
            predication_low_teta = (pixel_vals <= best_tetha) * 1.0
            predication_after_teta = (pixel_vals > best_tetha) * -1.0
            predication_teta = predication_low_teta + predication_after_teta
            predication_teta *= best_hypo_type
            self._dist_arr *= math.e ** (predication_teta * labels * a * -1.0)
            self._dist_arr /= np.linalg.norm(self._dist_arr)

    def _find_best_hypo(self, xs, pixel_idx, labels):
        """
        This function implements finding best hypothesis
        :param xs: xs data
        :param pixel_idx: pixel index
        :return:
        """

        # Sorted xs per pixel index (channel wise)
        sorted_indx = np.argsort(xs[:, pixel_idx])
        best_teta = xs[sorted_indx[0]][pixel_idx]

        for idx in sorted_indx:
            pass
            # sample =
        print 'debug'