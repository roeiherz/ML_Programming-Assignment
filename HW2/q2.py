import numpy as np
import numpy.random
from sklearn.datasets import fetch_mldata
import sklearn.preprocessing
import os
import matplotlib.pyplot as plt
from sklearn.svm import LinearSVC
from HW2.Perceptron import Perceptron
import math

NOF_ITERATIONS = 100


def get_train_validation_test_data():
    """
    This function get MNIST data and split it to train, validation and test data and labels
    :return: train_data, train_labels, validation_data, validation_labels, test_data, test_labels
    """
    mnist = fetch_mldata('MNIST original', data_home=os.getcwd())
    data = mnist['data']
    labels = mnist['target']
    neg, pos = 0, 8
    train_idx = numpy.random.RandomState(0).permutation(np.where((labels[:60000] == neg) | (labels[:60000] == pos))[0])
    test_idx = numpy.random.RandomState(0).permutation(np.where((labels[60000:] == neg) | (labels[60000:] == pos))[0])
    train_data_unscaled = data[train_idx[:6000], :].astype(float)
    train_labels = (labels[train_idx[:6000]] == pos) * 2 - 1
    validation_data_unscaled = data[train_idx[6000:], :].astype(float)
    validation_labels = (labels[train_idx[6000:]] == pos) * 2 - 1
    test_data_unscaled = data[60000 + test_idx, :].astype(float)
    test_labels = (labels[60000 + test_idx] == pos) * 2 - 1

    # Preprocessing
    train_data = sklearn.preprocessing.scale(train_data_unscaled, axis=0, with_std=False)
    validation_data = sklearn.preprocessing.scale(validation_data_unscaled, axis=0, with_std=False)
    test_data = sklearn.preprocessing.scale(test_data_unscaled, axis=0, with_std=False)
    return train_data, train_labels, validation_data, validation_labels, test_data, test_labels


def part_a(org_train, org_train_labels, org_validation, org_validation_labels):
    """
    This function running the part A - training SVM on train samples
    :param org_train: original train sample
    :param org_train_labels: original train labels
    :param org_validation: original validation samples
    :param org_validation_labels: original validation labels
    :return:
    """

    c_log_list = range(-10, 10, 1)
    c_log_list2 = range(1, 100, 1)

    # BEST C
    # Accuracy is: 0.991167301697 for c:2e-07

    for c_log in c_log_list2:
        # c = math.pow(10, c_log)
        c = float(c_log) * 1e-8
        svm = LinearSVC(loss='hinge', fit_intercept=False, C=c)
        svm.fit(org_train, org_train_labels)
        predication = svm.predict(org_validation)
        error_lst = (org_validation_labels != predication)
        error = np.sum(error_lst)
        acc = 1 - float(error) / np.size(predication)
        print "Accuracy is: {0} for c:{1}".format(acc, c)
    print 'debug'


def train_and_test(train, train_labels, test, test_labels):
    """
    This function will train the Perceptron and
    :param train: train samples
    :param train_labels: train labels
    :param test: test samples
    :param test_labels: test labels
    :return: mean accuracy
    """

    sum = 0
    for iteration in range(0, NOF_ITERATIONS):
        n = train.shape[0]
        nof_features = train.shape[1]

        # Shuffle the data
        indices_shuffle = np.random.permutation(n)
        train_shuffle = train[indices_shuffle]
        train_labels_shuffle = train_labels[indices_shuffle]

        perceptron = Perceptron(nof_features)
        perceptron.train(train_shuffle, train_labels_shuffle)
        acc = perceptron.test(test, test_labels)
        sum += acc

    mean_acc = float(sum) / NOF_ITERATIONS
    return mean_acc


def part_b(org_train, org_train_labels):
    """
    This function is calculates part b
    :param org_train: original train samples
    :param org_train_labels: original train labels
    :return: perceptron
    """

    nof_features = org_train.shape[1]
    perceptron = Perceptron(nof_features)
    perceptron.train(org_train, org_train_labels)
    weights = perceptron.get_weights()
    plt.figure()
    plt.imshow(np.reshape(weights, (28, 28)), interpolation='nearest')
    return perceptron


def part_c(perceptron, test, test_labels):
    """
    This function calculates part c
    :param perceptron:
    :param test:
    :param test_labels:
    :return:
    """

    mean_acc = perceptron.test(test, test_labels)
    print "Mean accuracy of the full train samples: {}".format(mean_acc)


def part_d(perceptron, test, test_labels):
    """
    This function calculates part d
    :param perceptron: perceptron
    :param test: test samples
    :param test_labels: test labels
    :return:
    """

    # Get the misclassified samples and labels
    misclassified_samples, misclassified_labels = perceptron.find_misclassified_samples(test, test_labels)
    # Plot the misclassified samples
    plt.figure()
    plt.imshow(np.reshape(misclassified_samples[1], (28, 28)), interpolation='nearest', cmap='gray')


if __name__ == '__main__':
    n_lst = [5, 10, 50, 100, 500, 1000, 5000]

    # Get train, validation and test data
    org_train, org_train_labels, org_validation, org_validation_labels, org_test, org_test_labels = get_train_validation_test_data()

    part_a(org_train, org_train_labels, org_validation, org_validation_labels)

    # perceptron = part_b(org_train, org_train_labels)
    # part_c(perceptron, org_test, org_test_labels)
    # part_d(perceptron, org_test, org_test_labels)

    print 'debug'
