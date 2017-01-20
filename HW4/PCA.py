import math
import numpy as np
# import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt

__author__ = 'roeiherz'


class PCA:

    def __init__(self, input_data):
        self._data = input_data

    def run(self, dim):
        """
        This function run the PCA
        :param dim:
        :return:
        """
        conv_mat = np.dot(np.transpose(self._data), self._data)
        u, d, v = np.linalg.svd(conv_mat)
        return u[:dim], d[:dim]
