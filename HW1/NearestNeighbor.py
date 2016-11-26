import numpy as np
from sklearn.datasets import fetch_mldata
from PIL import Image
import os

__author__ = 'roeiherz&mosheraboh'

MNIST_DATA_PATH = "C:\Users\J&M\PycharmProjects\ML_Programming-Assignment\HW1"
MNIST_FILE_NAME = "mnist-original.mat"


def get_data_and_labels():
    """This function downloads and load the data and labels from MNIST"""
    mnist = fetch_mldata('MNIST original', data_home=MNIST_DATA_PATH)
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
    """


if __name__ == '__main__':
    print 'start'
    data, labels = get_data_and_labels()
    train, train_labels, test, test_labels = get_train_and_test_data(data, labels)

    print 'end'
