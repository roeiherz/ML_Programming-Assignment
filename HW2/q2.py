import numpy as np
import numpy.random
from sklearn.datasets import fetch_mldata
import sklearn.preprocessing
import os
# import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.svm import LinearSVC


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


if __name__ == '__main__':
    n_lst = [5, 10, 50, 100, 500, 1000, 5000]

    # Get train, validation and test data
    org_train, org_train_labels, org_validation, org_validation_labels, org_test, org_test_labels = get_train_validation_test_data()

    c_list = np.array(list(range(1, 500, 5))).astype("float32") * 1e-8

    training_acc_lst = []
    validating_acc_lst = []

    for c in c_list:
        svm = LinearSVC(loss='hinge', fit_intercept=False, C=c)
        svm.fit(org_train, org_train_labels)

        training_acc = calc_accuracy(svm, org_train, org_train_labels)
        training_acc_lst.append(training_acc)
        validating_acc = calc_accuracy(svm, org_validation, org_validation_labels)
        validating_acc_lst.append(validating_acc)

    plt.figure()
    plot_graph(training_acc_lst, c_list, "q2_part_a", "", "Accuracy vs C for SVM", "Accuracy", "C")
    plot_graph(validating_acc_lst, c_list, "q2_part_a", "", "Accuracy vs C for SVM", "Accuracy", "C")
    best_acc_indx = validating_acc_lst.index(max(validating_acc_lst))
    best_c = c_list[best_acc_indx]
    print "The best c is {} for accuracy: {}".format(best_c, max(validating_acc_lst))

    plt.figure()
    plt.imshow(np.reshape(svm.coef_, (28, 28)), interpolation='nearest')
