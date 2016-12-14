import numpy as np
import numpy.random
from sklearn.datasets import fetch_mldata
import sklearn.preprocessing
import os
# import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.svm import LinearSVC

from HW2.SVM import SVM


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


def calc_accuracy(svm, dataset, dataset_labels):
    """
    This function receives dataset and dataset labels and calculates its svm accuracy
    :param svm: svm object
    :param dataset: dataset
    :param dataset_labels: dataset labels
    :return: accuracy
    """

    predication = svm.predict(dataset)
    error_lst = (dataset_labels != predication)
    error = np.sum(error_lst)
    acc = 1 - float(error) / np.size(predication)
    return acc


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


def part_a(org_train, org_train_labels):
    """
    This function calculates part A
    :return: best learning rate
    """
    learning_rate_lst = np.array(list(range(1, 99, 1))).astype("float32") / 100.0

    validating_acc_lst = []
    for lr in learning_rate_lst:
        mean_acc = 0
        for i in range(10):
            svm = SVM(org_train.shape[1])
            svm.train(org_train, org_train_labels, lr, T=1000)
            mean_acc += svm.test(org_validation, org_validation_labels)
        validating_acc_lst.append(mean_acc / (i + 1))
    plt.figure()
    plot_graph(validating_acc_lst, learning_rate_lst, "q3_part_a", "", "Accuracy vs Learning Rate for SVM", "Accuracy",
               "Learning Rate")
    best_acc_indx = validating_acc_lst.index(max(validating_acc_lst))
    best_lr = learning_rate_lst[best_acc_indx]
    print "The best learning rate is {} for accuracy: {}".format(best_lr, max(validating_acc_lst))
    return best_lr


def part_b(org_train, org_train_labels, best_lr):
    """
    This function implements part B
    :return: best learning rate
    """
    c_lst = np.array(list(range(1, 999, 10))).astype("float32") / 1000.0

    validating_acc_lst = []
    for c in c_lst:
        mean_acc = 0
        for i in range(10):
            svm = SVM(org_train.shape[1])
            svm.train(org_train, org_train_labels, best_lr, C=c, T=1000)
            mean_acc += svm.test(org_validation, org_validation_labels)
        validating_acc_lst.append(mean_acc / (i + 1))
    plt.figure()
    plot_graph(validating_acc_lst, c_lst, "q3_part_b", "", "Accuracy vs C for SVM", "Accuracy", "C")
    best_acc_indx = validating_acc_lst.index(max(validating_acc_lst))
    best_c = c_lst[best_acc_indx]
    print "The best C is {} for accuracy: {}".format(best_c, max(validating_acc_lst))
    return best_c


def part_c(org_train, org_train_labels, best_c, best_lr):
    """
    This function implements part C
    :param best_c: Best C
    :param best_lr: Best Learning rate
    :return:
    """
    svm = SVM(org_train.shape[1])
    svm.train(org_train, org_train_labels, best_lr, C=best_c, T=20000)
    acc = svm.test(org_validation, org_validation_labels)
    print "The Accuracy is: {} for best C: {} and learning rate: {}".format(acc, best_c, best_lr)
    return svm.get_weights()


def part_d(weights):
    """
    This function implements part D
    """
    plt.figure()
    plt.imshow(np.reshape(weights, (28, 28)), interpolation='nearest')


if __name__ == '__main__':

    # Get train, validation and test data
    org_train, org_train_labels, org_validation, org_validation_labels, org_test, org_test_labels = get_train_validation_test_data()

    # best_lr = part_a(org_train, org_train_labels)
    best_lr = 0.94
    best_c = part_b(org_train, org_train_labels, best_lr)
    print 'debug'
    weights = part_c(org_train, org_train_labels, best_c, best_lr)
    print 'debug'
    part_d(weights)
    print 'debug'
