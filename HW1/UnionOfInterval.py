# coding=utf-8
import numpy as np
import datetime
from sklearn.datasets import fetch_mldata
import os
import matplotlib.pyplot as plt
# import matplotlib
# matplotlib.use('Agg')

__author__ = 'roeiherz & mosheraboh'

# Sample Size
M = 100


def get_probablity(xs, ys):
    """
    This function ...
    :param xs: The data input
    :param ys: The label which is according to the probability:
                Pr(y = 1|x) = (0.8 if x ∈ [0, 0.25] or x ∈ [0.5, 0.75]
                               0.1 if x ∈ [0.25, 0.5] or x ∈ [0.75, 1])
    """

    indices_x = np.where((xs < 0.25) | ((0.5 <= xs) & (xs < 0.75)))
    indices_y_lower = np.where(ys < 0.2)
    new_indices = set(indices_x[0]) & set(indices_y_lower[0])
    ys[np.array(list(new_indices))] = 0
    indices_y_upper = np.where(ys >= 0.2)
    new_indices2 = set(indices_x[0]) & set(indices_y_upper[0])
    ys[np.array(list(new_indices2))] = 1
    indices_x = np.where((xs >= 0.75) | ((0.25 <= xs) & (xs < 0.5)))
    indices_y_lower = np.where(ys < 0.9)
    new_indices = set(indices_x[0]) & set(indices_y_lower[0])
    ys[np.array(list(new_indices))] = 0
    indices_y_upper = np.where(ys >= 0.9)
    new_indices2 = set(indices_x[0]) & set(indices_y_upper[0])
    ys[np.array(list(new_indices2))] = 1


if __name__ == '__main__':
    print 'start'
    # Start time
    start_time = datetime.datetime.now()
    print 'the start is at {}'.format(start_time)
    xs = np.random.sample(M)
    ys = np.random.sample(M)
    xs.sort()
    get_probablity(xs, ys)

    end_time = datetime.datetime.now()
    time_diff = end_time - start_time
    print 'this algo runs in {0} '.format(time_diff)
    print 'end'
