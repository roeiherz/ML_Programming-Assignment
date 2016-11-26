import numpy as np
import datetime
from sklearn.datasets import fetch_mldata
from PIL import Image
import os

K = 100
NOF_TRAINING_DATA = 1000

__author__ = 'roeiherz & mosheraboh'


def get_data_and_labels():
    """This function downloads and load the data and labels from MNIST"""
    mnist = fetch_mldata('MNIST original', data_home=os.getcwd())
    data = mnist['data']
    labels = mnist['target']
    return data, labels


def visualize(image):
    """
    This function visualize the photo
    :param image: the image
    """
    img = Image.fromarray(np.reshape(image, (28, 28)))
    img.show()


def get_train_and_test_data(data, labels):
    """
    :param data: data
    :param labels: labels
    :return:
    """
    idx = np.random.RandomState(0).choice(70000, 11000)
    train = data[idx[:10000], :]
    train_labels = labels[idx[:10000]]
    test = data[idx[10000:], :]
    test_labels = labels[idx[10000:]]
    return train, train_labels, test, test_labels


def knn(train, train_labels, test_image, k):
    """
    This function will implemented K-NN algorithm to return a prediction of the query image
    :param train: train set
    :param train_labels: train labels corresponding to the train set
    :param test_image: one image for testing
    :param k: hyper-parameter
    :return chosen label - predicated label
    """
    dist_train = np.sum(np.square(train - test_image), axis=1)
    # Get the nearest k indices using L2 matric
    k_nearest_ind = np.argpartition(dist_train, k)
    # Get the neares the label
    k_nearest_labels = train_labels[k_nearest_ind[:k]]
    # Get the most frequent label
    chosen_label = get_chosen_label(k_nearest_labels)
    return chosen_label


def get_chosen_label(k_nearest_labels):
    """
    This function get the k labels and returns the most frequent label (in case of a tie between the k labels
    of neighbors it will choose randomly)
    :param k_nearest_labels: k nearest label
    :return: chosen label
    """
    # Init the array labels
    count_labels = np.zeros([10])
    # Performs a count sort
    idx, counts = np.unique(k_nearest_labels, return_counts=True)
    count_labels[idx.astype(int)] += counts
    # Get the maximum indices
    max_indices = np.where(count_labels == np.max(count_labels))
    # Get randomly index in case of a tie
    max_index_random = np.random.choice(max_indices[0])
    return max_index_random


def run_knn(train, train_labels, test, test_labels, k, n):
    """
    This function will split the training data and labels by n and will run the knn algorithm
    :param train_labels: train label
    :param train: train dataset
    :param test_labels: test label
    :param k: K
    :param n: Number of Training data
    :param test: the test
    :return:
    """
    test_size = len(test)
    train_n = train[:n]
    train_labels_n = train_labels[:n]
    error = 0

    for i in range(test_size):
        predicted_label = knn(train_n, train_labels_n, test[i], k=k)
        if not predicted_label == test_labels[i]:
            error += 1

    accuracy = 1.0 - (float(error) / test_size)
    return accuracy

if __name__ == '__main__':
    print 'start'
    # Start time
    start_time = datetime.datetime.now()
    data, labels = get_data_and_labels()
    train, train_labels, test, test_labels = get_train_and_test_data(data, labels)
    accuracy_lst = []
    for i in range(K):
        accuracy = run_knn(train, train_labels, test, test_labels, k=i, n=NOF_TRAINING_DATA)
        print 'The accuracy: k= {0}, n= {1} is: {2}'.format(i, NOF_TRAINING_DATA, accuracy)
        # Append the accuracy results
        accuracy_lst.append(accuracy)

    # End time
    end_time = datetime.datetime.now()
    print 'end'
