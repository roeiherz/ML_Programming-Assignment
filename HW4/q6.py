from numpy import *
import numpy.random
from sklearn.datasets import fetch_mldata
import sklearn.preprocessing
import os
import numpy as np
# import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
from HW4.PCA import PCA

NOF_EIGENVALUES = 100
NOF_EIGENVECTORS = 5
ITERS = 300


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

    # test_data_size = 2000
    # test_data_unscaled = data[60000 + test_idx[:test_data_size], :].astype(float)
    # test_labels = (labels[60000 + test_idx[:test_data_size]] == pos) * 2 - 1

    # Preprocessing
    train_data = sklearn.preprocessing.scale(train_data_unscaled, axis=0, with_std=False)
    # validation_data = sklearn.preprocessing.scale(validation_data_unscaled, axis=0, with_std=False)
    # test_data = sklearn.preprocessing.scale(test_data_unscaled, axis=0, with_std=False)
    return train_data, train_labels


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
    plt.close()


def part_a_and_b_and_c(org_train, org_train_labels, label=1, section='a', both=0):
    """
    This function implements part a and b.
    This function will plot Iteration Vs Accuracy training and testing error
    """

    # Both check if the label is 0 and 8 we do not need for scaled the data for one of the labels
    if not both:
        # label 1 is 8 number and -1 is 0 number. The num fields is for saving the figures
        if label > 0:
            num = 8
        else:
            num = 0

        train_data_unscaled = get_label_dataset(org_train, org_train_labels, label)
        train_data_mean = np.mean(train_data_unscaled, axis=0)
        train_dataset = sklearn.preprocessing.scale(train_data_unscaled, axis=0, with_std=False)
    else:
        # the train data should not change (its already for both labels
        num = "0&8"
        train_dataset = org_train
        train_data_mean = np.mean(train_dataset, axis=0)

    # Save the mean image
    plt.figure()
    plt.imshow(np.reshape(train_data_mean, (28, 28)), interpolation='nearest')
    plt.savefig('q6_part_{}_mean_image'.format(section))
    plt.close()

    # Do PCA
    pca = PCA(train_dataset)
    u, d, v = pca.run(dim=100)

    # Save the eigen-vectors
    for i in range(NOF_EIGENVECTORS):
        plt.figure()
        plt.imshow(np.reshape(v[i], (28, 28)), interpolation='nearest')
        plt.savefig('q6_part_{}_i{}_label_{}'.format(section, i, num))
        plt.close()

    # Save the eigenvalues
    plt.figure()
    plot_graph(d, range(1, NOF_EIGENVALUES + 1), "q6_part_{}_label_{}".format(section, num), "",
               "Iteration vs EigenValues",
               "EigenValues", "Iteration")


def part_d(train_dataset, org_train_labels):
    """
    This function implements part d.
    This function will plot Iteration Vs Accuracy training and testing error
    """

    # Do PCA
    pca = PCA(train_dataset)
    u, d, v = pca.run(dim=2)
    mat_data = np.dot(train_dataset, np.transpose(v))
    plt.figure()
    plt.scatter(mat_data[:, 0], mat_data[:, 1], c=org_train_labels)
    plt.savefig("q6_part_d.png")


def part_e(train_dataset, org_train_labels):
    """
    This function implements part e.
    This function will plot Iteration Vs Accuracy training and testing error
    """

    pos_train_data = get_label_dataset(org_train, org_train_labels, 1)
    neg_train_data = get_label_dataset(org_train, org_train_labels, -1)
    images_lst = [pos_train_data[0], pos_train_data[1], neg_train_data[0], neg_train_data[1]]

    for k in [10, 30, 50]:
        # Do PCA
        pca = PCA(train_dataset)
        u, d, v = pca.run(dim=k)
        # Index for saving different names of images
        idx = 0
        for image in images_lst:
            # image is x
            x = image
            plt.figure()
            plt.imshow(np.reshape(x, (28, 28)), interpolation='nearest')
            plt.savefig("q6_part_e_original_{}_k_{}.png".format(idx, k))
            plt.close()
            # Encoding the image
            img_encode = np.dot(v, x)
            # Decoding the image
            img_decode = np.dot(u, img_encode)
            plt.figure()
            plt.imshow(np.reshape(img_decode, (28, 28)), interpolation='nearest')
            plt.savefig("q6_part_e_decode_{}_k_{}.png".format(idx, k))
            plt.close()
            idx += 1


def get_label_dataset(org_train, org_train_labels, label):
    """
    This function create only the necessary label data from the original data
    :param org_train: training original dataset
    :param org_train_labels: training labels dataset
    :return:
    """

    label_idx = np.where(org_train_labels == label)
    training_dataset_label = org_train[label_idx]
    return training_dataset_label


if __name__ == '__main__':
    # Get train, validation and test data
    org_train, org_train_labels = get_train_validation_test_data()

    # Part A - only number 8
    part_a_and_b_and_c(org_train, org_train_labels, label=1, section='a')
    # Part B - only number 0
    part_a_and_b_and_c(org_train, org_train_labels, label=-1, section='b')
    # Part C - both numbers 0 and 1
    part_a_and_b_and_c(org_train, org_train_labels, label=-1, section='c', both=1)

    # Part D
    part_d(org_train, org_train_labels)

    # Part E
    part_e(org_train, org_train_labels)
