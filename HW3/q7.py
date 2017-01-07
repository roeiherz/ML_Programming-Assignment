from numpy import *
import numpy.random
from sklearn.datasets import fetch_mldata
import sklearn.preprocessing
import os
import numpy as np

from MultiClassSVM import MultiClassSVM
# import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt

# C Grid search params
C_STEP = 5
C_HIGH_THR = 150
C_LOW_THR = 80

# LR Grid search params
LR_STEP = 10
LR_HIGH_THR = 350
LR_LOW_THR = 250

# Training params
NOF_ITERS = 5000
ITERS = 2


def get_train_validation_test_data():
    """
    This function get MNIST data and split it to train, validation and test data and labels
    :return: train_data, train_labels, validation_data, validation_labels, test_data, test_labels
    """
    mnist = fetch_mldata('MNIST original', data_home=os.getcwd())
    data = mnist['data']
    labels = mnist['target']

    train_idx = numpy.random.RandomState(0).permutation(range(60000))

    train_data_size = 50000
    train_data_unscaled = data[train_idx[:train_data_size], :].astype(float)
    train_labels = labels[train_idx[:train_data_size]]

    validation_data_unscaled = data[train_idx[train_data_size:60000], :].astype(float)
    validation_labels = labels[train_idx[train_data_size:60000]]

    test_data_unscaled = data[60000:, :].astype(float)
    test_labels = labels[60000:]

    # Preprocessing
    train_data = sklearn.preprocessing.scale(train_data_unscaled, axis=0, with_std=False)
    validation_data = sklearn.preprocessing.scale(validation_data_unscaled, axis=0, with_std=False)
    test_data = sklearn.preprocessing.scale(test_data_unscaled, axis=0, with_std=False)
    return train_data, train_labels, validation_data, validation_labels, test_data, test_labels


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


def find_best_c(org_train, org_train_labels, org_validation, org_validation_labels, lr=0.01, t=NOF_ITERS, nof_iters=10):
    """
    This function finds the best c and generates accuracy graphs
    :param org_train: original training data
    :param org_train_labels: original training data labels
    :param org_validation: original validation data
    :param org_validation_labels: original validation data labels
    :param lr: learning rate
    :param t: T is the number of iterations for SVM
    :param nof_iters: nof iterations for calculation mean accuracy
    :return: the best C and the max accuracy
    """

    # We used grid search to reduce the range
    c_list = np.array(list(range(C_LOW_THR, C_HIGH_THR, C_STEP))).astype("float32") / 100
    training_mean_acc_lst = []
    validating_mean_acc_lst = []
    for c in c_list:
        training_acc = 0
        validation_acc = 0
        for i in range(nof_iters):
            mc_svm = MultiClassSVM(max_samples_size=org_train.shape[0], num_of_classes=10)
            mc_svm.train(org_train, org_train_labels, lr, C=c, T=t)
            training_acc += mc_svm.test(org_train, org_train_labels)
            validation_acc += mc_svm.test(org_validation, org_validation_labels)

        training_mean_acc_lst.append(training_acc / float(nof_iters))
        validating_mean_acc_lst.append(validation_acc / float(nof_iters))
    plt.figure()
    plot_graph(validating_mean_acc_lst, c_list, "q7_part_a2", "", "Accuracy vs C for MultiClassSVM", "Accuracy",
               "C")
    plot_graph(training_mean_acc_lst, c_list, "q7_part_a2", "", "Accuracy vs C for MultiClassSVM", "Accuracy",
               "C")
    best_acc_indx = validating_mean_acc_lst.index(max(validating_mean_acc_lst))

    return c_list[best_acc_indx], max(validating_mean_acc_lst)


def find_best_lr(org_train, org_train_labels, org_validation, org_validation_labels, c=1, t=NOF_ITERS, nof_iters=10):
    """
    This function finds the best c and generates accuracy graphs
    :param org_train: original training data
    :param org_train_labels: original training data labels
    :param org_validation: original validation data
    :param org_validation_labels: original validation data labels
    :param c: learning rate
    :param t: T is the number of iterations for SVM
    :param nof_iters: nof iterations for calculation mean accuracy
    :return: the best learning rate
    """

    # We used grid search to reduce the range
    lr_list = np.array(list(range(LR_LOW_THR, LR_HIGH_THR, LR_STEP))).astype("float32") / 100.0
    training_mean_acc_lst = []
    validating_mean_acc_lst = []
    for lr in lr_list:
        training_acc = 0
        validation_acc = 0
        for i in range(nof_iters):
            mc_svm = MultiClassSVM(max_samples_size=org_train.shape[0], num_of_classes=10)
            mc_svm.train(org_train, org_train_labels, lr, C=c, T=t)
            training_acc += mc_svm.test(org_train, org_train_labels)
            validation_acc += mc_svm.test(org_validation, org_validation_labels)

        training_mean_acc_lst.append(training_acc / float(nof_iters))
        validating_mean_acc_lst.append(validation_acc / float(nof_iters))
    plt.figure()
    plot_graph(validating_mean_acc_lst, lr_list, "q7_part_a1", "", "Accuracy vs Learning Rate for MultiClassSVM",
               "Accuracy", "Learning Rate")
    plot_graph(training_mean_acc_lst, lr_list, "q7_part_a1", "", "Accuracy vs Learning Rate for MultiClassSVM",
               "Accuracy", "Learning Rate")
    best_acc_indx = validating_mean_acc_lst.index(max(validating_mean_acc_lst))

    return lr_list[best_acc_indx]


def part_a(org_train, org_train_labels, org_validation, org_validation_labels):
    """
    This function implements part A
    :param org_train: original training data
    :param org_train_labels: original training data labels
    :param org_validation: original validation data
    :param org_validation_labels: original validation data labels
    :return: best learning rate, best C
    """
    best_lr = find_best_lr(org_train, org_train_labels, org_validation, org_validation_labels, nof_iters=ITERS)
    best_c, acc = find_best_c(org_train, org_train_labels, org_validation, org_validation_labels, lr=best_lr,
                              nof_iters=ITERS)
    print "The best learning rate is {} and best c: {} with accuracy: {}".format(best_lr, best_c, acc)
    return best_lr, best_c


def part_b(best_lr, best_c, org_train, org_train_labels, org_test, org_test_labels, t=NOF_ITERS):
    """
    This function implements part B
    :param best_lr: best learning
    :param best_c: best C
    :param org_train: original train data
    :param org_train_labels: original train data labels
    :param org_test: original test data
    :param org_test_labels: original test labels data
    :param t: T is the number of iterations
    :return: multi class svm
    """

    mc_svm = MultiClassSVM(max_samples_size=org_train.shape[0], num_of_classes=10)
    mc_svm.train(org_train, org_train_labels, learning_rate=best_lr, C=best_c, T=t)
    acc = mc_svm.test(org_test, org_test_labels)
    print 'This is the accuracy from the test data: {}'.format(acc)


if __name__ == '__main__':
    # Get train, validation and test data
    org_train, org_train_labels, org_validation, org_validation_labels, org_test, org_test_labels = get_train_validation_test_data()

    best_lr, best_c = part_a(org_train, org_train_labels, org_validation, org_validation_labels)
    part_b(best_lr, best_c, org_train, org_train_labels, org_test, org_test_labels, t=NOF_ITERS)
    print 'end'
