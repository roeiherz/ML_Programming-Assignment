import numpy as np
from sklearn.datasets import fetch_mldata
import cv2
from PIL import Image
import os

__author__ = 'roeiherz'

MNIST_DATA_PATH = "C:\Users\RoeiH\PycharmProjects\ML_Programming-Assignment\HW1"
MNIST_FILE_NAME = "mnist-original.mat"


def get_data_and_labels():
    """This function downloads and load the data and labels from MNIST"""
    mnist = fetch_mldata('MNIST original', data_home=MNIST_DATA_PATH)
    data = mnist['data']
    labels = mnist['target']
    return data, labels


def visualize(image):
    """This function visualize the photo"""
    img = Image.fromarray(np.reshape(image, (28, 28)))
    img.show()

if __name__ == '__main__':
    print 'start'
    data, labels = get_data_and_labels()
    visualize(data[2])
    print 'end'
