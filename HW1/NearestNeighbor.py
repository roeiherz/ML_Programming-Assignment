import numpy as np
import datetime
from sklearn.datasets import fetch_mldata
from PIL import Image
import os
import matplotlib.pyplot as plt

N_STEP = 100
# K = 100
# NOF_TRAINING_DATA = 1000
K = 1
NOF_TRAINING_DATA = 5000

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
    This function splits the data and the labels to train and test
    Train size of 10000 and Test size of 1000
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
    :param test: Test data
    :return: accuracy of the data
    """
    test_size = len(test)
    train_n = train[:n]
    train_labels_n = train_labels[:n]
    error = 0

    for i in range(test_size):
        # Run the knn algorithm and get the predicted
        predicted_label = knn(train_n, train_labels_n, test[i], k=k)
        if not predicted_label == test_labels[i]:
            # Count the errors
            error += 1

    # Calculate the accuracy
    accuracy = 1.0 - (float(error) / test_size)
    return accuracy


def plot_graph(accuracy_lst):
    """
    This function will save and plot graph of accuracy vs K
    :param accuracy_lst: the accuracy results for each k
    """
    plt.plot(accuracy_lst)
    plt.title("KNN algorithm for k={0} and N={1}".format(K, NOF_TRAINING_DATA))
    plt.ylabel('Accuracy')
    plt.xlabel('K')
    plt.savefig('graph_N_fixed.png')


def get_accuracy_of_k():
    """
    This function runs the KNN algorithm of K while N is fixed
    :return: accuracy_lst
    """
    for i in range(1, K + 1):
        accuracy = run_knn(train, train_labels, test, test_labels, k=i, n=NOF_TRAINING_DATA)
        print 'The accuracy: k= {0}, n= {1} is: {2}'.format(i, NOF_TRAINING_DATA, accuracy)
        # Append the accuracy results
        accuracy_lst.append(accuracy)
    return accuracy_lst


def get_accuracy_of_n():
    """
    This function runs the KNN algorithm of N while K is fixed
    :return: accuracy_lst
    """
    for i in range(N_STEP, NOF_TRAINING_DATA + 1, N_STEP):
        accuracy = run_knn(train, train_labels, test, test_labels, k=K, n=i)
        print 'The accuracy: k= {0}, n= {1} is: {2}'.format(K, i, accuracy)
        # Append the accuracy results
        accuracy_lst.append(accuracy)
    return accuracy_lst

if __name__ == '__main__':
    print 'start'
    # Start time
    start_time = datetime.datetime.now()
    print 'the start is {}'.format(start_time)
    data, labels = get_data_and_labels()
    train, train_labels, test, test_labels = get_train_and_test_data(data, labels)
    accuracy_lst = []

    # accuracy_lst = get_accuracy_of_k()
    accuracy_lst = get_accuracy_of_n()
    # Plot and save the image
    plot_graph(accuracy_lst)

    # End time
    end_time = datetime.datetime.now()
    time_diff = end_time - start_time
    print 'this algo runs in {0} '.format(time_diff)
    print 'end'
