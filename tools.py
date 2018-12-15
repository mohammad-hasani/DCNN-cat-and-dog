import vtk
from vtk.util import numpy_support
import os
import numpy as np
from matplotlib import pyplot, cm
import cv2


def get_data():
    cases = ['case' + str(i) for i in range(1, 11)]
    X = list()
    Y = list()
    for i, v in enumerate(cases):
        images = get_images(v)
        labels = get_labels(v)
        for j in range(len(images)):
            X.append(images[j])
            Y.append(labels[j])
    X = np.array(X)
    Y = np.array(Y)
    return X, Y


def fill_data_with_zeros(data):
    n = 650
    data_len = len(data)
    tmp = (n - data_len)
    new_data = np.full((tmp), 255, dtype=float)
    result = np.concatenate((tmp, new_data), axis=0)
    print(result)


def read_image(number):
    numbers = ['./numbers/' + str(i) + '.PNG' for i in range(10)]
    data = cv2.imread(numbers[number], 0)
    data = data.flatten()
    tmp = np.empty(650)
    tmp[:len(data)] = data
    tmp[len(data):] = 255
    tmp = np.array([tmp])
    return tmp


def convert_raw_image(data):
    data = data.flatten()
    tmp = np.empty(650)
    tmp[:len(data)] = data
    tmp[len(data):] = 255
    tmp = np.array([tmp])
    return tmp
