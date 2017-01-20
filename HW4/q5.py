from numpy import *
import numpy.random
from sklearn.datasets import fetch_mldata
import sklearn.preprocessing
import os
import numpy as np
# import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
from HW4.AdaBoost import AdaBoost

# Training params
ITERS = 200


def get_train_validation_test_data():
    """
    This function get MNIST data and split it to train, validation and test data and labels
    :return: train_data, train_labels, validation_data, validation_labels, test_data, test_labels
    """
    mnist = fetch_mldata('MNIST original', data_home=os.getcwd())
    data = mnist['data']
    labels = mnist['target']

    neg, pos = 0, 8
    train_idx = numpy.random.RandomState(0).permutation(where((labels[:60000] == neg) | (labels[:60000] == pos))[0])
    test_idx = numpy.random.RandomState(0).permutation(where((labels[60000:] == neg) | (labels[60000:] == pos))[0])

    train_data_size = 2000
    train_data_unscaled = data[train_idx[:train_data_size], :].astype(float)
    train_labels = (labels[train_idx[:train_data_size]] == pos) * 2 - 1

    # validation_data_unscaled = data[train_idx[6000:], :].astype(float)
    # validation_labels = (labels[train_idx[6000:]] == pos)*2-1

    test_data_size = 2000
    test_data_unscaled = data[60000 + test_idx[:test_data_size], :].astype(float)
    test_labels = (labels[60000 + test_idx[:test_data_size]] == pos) * 2 - 1

    # Preprocessing
    train_data = sklearn.preprocessing.scale(train_data_unscaled, axis=0, with_std=False)
    # validation_data = sklearn.preprocessing.scale(validation_data_unscaled, axis=0, with_std=False)
    test_data = sklearn.preprocessing.scale(test_data_unscaled, axis=0, with_std=False)
    return train_data, train_labels, test_data, test_labels


def plot_graph(error_lst, x_lst, file_name='', label='', title='', ylabel='', xlabel=''):
    """
    This function is plotting the graph
    :param label: for plotting legend
    :param title: title for the plotting
    :param ylabel: ylabel for the plotting
    :param xlabel: xlabel for the plotting
    :param error_lst: the of errors
    :param x_lst: list of the number of samples
    :param file_name: file_name to be saved
    """
    plt.plot(x_lst, error_lst, label=label)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.savefig('{}.png'.format(file_name))


def part_a(t_lst, acc_test_lst, acc_train_lst):
    """
    This function implements part a.
    This function will plot Iteration Vs Accuracy training and testing error
    """

    plt.figure()
    plot_graph(acc_train_lst, t_lst, "q5_part_a", "", "Iteration vs Accuracy", "Accuracy", "Iteration")
    plot_graph(acc_test_lst, t_lst, "q5_part_a", "", "Iteration vs Accuracy", "Accuracy", "Iteration")


def part_b(t_lst, loss_test_lst, loss_train_lst):
    """
    This function implements part b.
    This function will plot Iteration Vs Accuracy training and testing error
    """

    plt.figure()
    plot_graph(loss_test_lst, t_lst, "q5_part_b_test", "", "Iteration vs Loss Test", "Loss", "Iteration")
    plt.figure()
    plot_graph(loss_train_lst, t_lst, "q5_part_b_train", "", "Iteration vs Loss Train", "Loss", "Iteration")

if __name__ == '__main__':
    # Get train, validation and test data
    org_train, org_train_labels, org_test, org_test_labels = get_train_validation_test_data()
    ada_boost = AdaBoost(nof_features=org_train.shape[1], T=ITERS, m=org_train.shape[0])
    ada_boost.train(org_train, org_train_labels)

    acc_test_lst = []
    loss_test_lst = []
    acc_train_lst = []
    loss_train_lst = []
    t_lst = range(1, ITERS + 1)
    for t in t_lst:
        acc_test, loss_test = ada_boost.test(org_test, org_test_labels, iterations=t)
        acc_test_lst.append(acc_test)
        loss_test_lst.append(loss_test)
        acc_train, loss_train = ada_boost.test(org_train, org_train_labels, iterations=t)
        acc_train_lst.append(acc_train)
        loss_train_lst.append(loss_train)
        print 'T {} - Train Acc {}, Test Acc {}, Test Loss {}, Train Loss {}'.format(t, acc_train, acc_test, loss_test,
                                                                                     loss_train)

    part_a(t_lst, acc_test_lst, acc_train_lst)
    part_b(t_lst, loss_test_lst, loss_train_lst)
    print 'end'
