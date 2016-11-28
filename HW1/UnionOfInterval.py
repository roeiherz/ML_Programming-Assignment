# coding=utf-8
import numpy as np
import datetime
from sklearn.datasets import fetch_mldata
import os
from intervals import find_best_interval
import matplotlib.pyplot as plt

import matplotlib
matplotlib.use('Agg')

__author__ = 'roeiherz & mosheraboh'


def draw_pairs(xs, ys, intervals):
    """

    :param xs: data
    :param ys: labels
    :param intervals: list of tuples
    :return: none
    """
    plt.figure()
    plt.plot(xs, ys, 'ro')
    plt.title("Samples (labels Vs X)")
    plt.ylabel("Labels")
    plt.xlabel("Xs")
    plt.vlines([0.25, 0.5, 0.75], -0.1, 1.1, 'b')
    for interval in intervals:
        x_min = interval[0]
        x_max = interval[1]
        plt.hlines(0.5, x_min, x_max, 'r', label='positive interval')
    axes = plt.gca()
    axes.set_ylim([-0.1, 1.1])
    plt.savefig('{}.png'.format("samples"))


def get_pairs(size):
    """
    Get 'size' samples  with the following distribution:
    D(x, y) = Pr(y = 1|x) * Pr(x)
    While x is uniformly distibuted [0-1)
    And:
                Pr(y = 1|x) = (0.8 if x ∈ [0, 0.25] or x ∈ [0.5, 0.75]
                               0.1 if x ∈ [0.25, 0.5] or x ∈ [0.75, 1])
    """

    # get random xs (x is ditributed uniformly)
    xs = np.random.sample(size)
    # sort xs
    xs.sort()

    #  Get labels according to the required distribution
    #  matrix operation that implement the following:
    # foreach x in xs:
    #     rand <-- get random [0-1)
    #     if x ∈ [0, 0.25] or x ∈ [0.5, 0.75] then y = 0 iff rand < 0.2
    #     else  y = 0 iff rand < 0.9
    ys = np.random.sample(size)
    indices_x = np.where((xs < 0.25) | ((0.5 <= xs) & (xs < 0.75)))
    indices_y_lower = np.where(ys < 0.2)
    new_indices = set(indices_x[0]) & set(indices_y_lower[0])
    if len(new_indices):
        # If set is not empty
        ys[np.array(list(new_indices))] = 0
    indices_y_upper = np.where(ys >= 0.2)
    new_indices2 = set(indices_x[0]) & set(indices_y_upper[0])
    if len(new_indices2):
        # If set is not empty
        ys[np.array(list(new_indices2))] = 1
    indices_x = np.where((xs >= 0.75) | ((0.25 <= xs) & (xs < 0.5)))
    indices_y_lower = np.where(ys < 0.9)
    new_indices = set(indices_x[0]) & set(indices_y_lower[0])
    if len(new_indices):
        # If set is not empty
        ys[np.array(list(new_indices))] = 0
    indices_y_upper = np.where(ys >= 0.9)
    new_indices2 = set(indices_x[0]) & set(indices_y_upper[0])
    if len(new_indices2):
        # If set is not empty
        ys[np.array(list(new_indices2))] = 1
    return xs, ys


def part_a():
    """
    This function implement part A from HW
    """
    # Sample size
    M = 100
    # Number of intervals
    K = 2
    # Get pairs according to the required distribution
    xs, ys = get_pairs(M)
    # find the intervals
    intervals, best_error = find_best_interval(xs, ys, K)
    # Draw the pairs
    draw_pairs(xs, ys, intervals)


def calc_true_error(intervals):
    """
    TBD
    :param intervals:
    :return:
    """
    true_error = 0
    error_intervals_1 = (0.0, 0.25, 0.2)
    error_intervals_2 = (0.25, 0.5, 0.9)
    error_intervals_3 = (0.5, 0.75, 0.2)
    error_intervals_4 = (0.75, 1.0, 0.9)
    # Append error intervals to a list
    error_intervals = [error_intervals_1, error_intervals_2, error_intervals_3, error_intervals_4]

    for error_interval in error_intervals:
        start = error_interval[0]
        end = error_interval[1]
        err = error_interval[2]

        pos_length = 0

        for interval in intervals:
            x_min = interval[0]
            x_max = interval[1]

            if x_max > start and x_min < end:
                pos_length += min(end, x_max) - max(start, x_min)

        true_error += pos_length * err + (0.25 - pos_length) * (1 - err)

    return true_error


def plot_graph(error_lst, m_lst, file_name='', label=''):
    """
    TL:TR
    AMALEK
    :param error_lst:
    :param m_lst:
    :param file_name:
    """
    plt.plot(m_lst, error_lst, label=label)
    plt.title("Error as a function of number of samples")
    plt.ylabel('Error')
    plt.xlabel('Number of samples')
    plt.savefig('{}.png'.format(file_name))


def part_c():
    """
    This function implement part B from HW
    """
    K = 2
    T = 100
    M_START = 10
    M_END = 100
    M_STEP = 5

    # This list is for a plot
    avg_true_error_lst = []
    avg_erm_error_lst = []
    for m in range(M_START, M_END + 1, M_STEP):
        # Temp vars for average calculation
        total_true_error = 0
        total_erm_error = 0
        for t in range(1, T + 1):
            # Calculate the averages
            xs, ys = get_pairs(m)
            intervals, error = find_best_interval(xs, ys, K)
            erm_error = float(error) / m
            true_error = calc_true_error(intervals)

            total_true_error += true_error
            total_erm_error += erm_error

        avg_true_error = total_true_error / T
        avg_erm_error = total_erm_error / T
        avg_erm_error_lst.append(avg_erm_error)
        avg_true_error_lst.append(avg_true_error)
        print "Avg TRUE error:{0}, Avg ERM error: {1}".format(avg_true_error, avg_erm_error)

    plot_graph(avg_erm_error_lst, range(M_START, M_END + 1, M_STEP), "partc", "AVG ERM ERROR")
    plot_graph(avg_true_error_lst, range(M_START, M_END + 1, M_STEP), "partc", "AVG TRUE ERROR")

if __name__ == '__main__':
    print 'start'
    # Start time
    start_time = datetime.datetime.now()
    print 'the start is at {}'.format(start_time)

    # part A
    # part_a()

    # part B
    part_c()


    end_time = datetime.datetime.now()
    time_diff = end_time - start_time
    print 'this algo runs in {0} '.format(time_diff)
    print 'end'
