import cv2
import numpy as np
from matplotlib import pyplot as plt
from DCNN import DCNN
from keras.models import load_model
import tools
from keras.datasets import mnist
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from keras.utils import np_utils
import random


def train_model_DCNN():
    cats = ['./cats/cat.' + str(i) + '.jpg' for i in range(10, 20)]
    dogs = ['./dogs/dog.' + str(i) + '.jpg' for i in range(50, 60)]
    print(cats)
    X = list()
    Y = list()
    ROWS = 300
    COLS = 300
    for i in cats:
        image = cv2.imread(i, 0)
        print(type(image))
        tmp = np.empty((ROWS, COLS))
        tmp[:, :] = 255
        end_rows = 0
        end_cols = 0
        if image.shape[0] > ROWS:
            end_rows = ROWS
        else:
            end_rows = image.shape[0]
        if image.shape[1] > COLS:
            end_cols = COLS
        else:
            end_cols = image.shape[1]
        tmp[0:end_rows, 0:end_cols] = image[0: end_rows, 0:end_cols]
        X.append(tmp)
        Y.append(0)
    for i in dogs:
        image = cv2.imread(i, 0)
        tmp = np.empty((ROWS, COLS))
        tmp[:, :] = 255
        end_rows = 0
        end_cols = 0
        if image.shape[0] > ROWS:
            end_rows = ROWS
        else:
            end_rows = image.shape[0]
        if image.shape[1] > COLS:
            end_cols = COLS
        else:
            end_cols = image.shape[1]
        tmp[0:end_rows, 0:end_cols] = image[0: end_rows, 0:end_cols]
        X.append(tmp)
        Y.append(1)
    X = np.array(X)

    print(X.shape)

    dcnn = DCNN(X, Y)
    model = dcnn.dcnn()
    model.save('./models/model_dcnn.h5')


def get_prediction(data):
    model = load_model('./models/model.h5')
    number = model.predict_classes(data)
    return number


def main():
    train_model_DCNN()


if __name__ == '__main__':
    main()
