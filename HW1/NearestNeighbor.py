import numpy as np
from sklearn.datasets import fetch_mldata

__author__ = 'roeiherz'


def get_data_and_labels():
    mnist = fetch_mldata('MNIST original')
    data = mnist['data']
    labels = mnist['target']
    return data, labels

if __name__ == '__main__':
    data, labels = get_data_and_labels()

