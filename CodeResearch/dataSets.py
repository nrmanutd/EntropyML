import math
import os

import pandas as pd
import numpy as np
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.datasets import mnist


def loadMnist():
    num_train = 60000  # there are 60000 training examples in MNIST
    num_test = 10000  # there are 10000 test examples in MNIST

    height, width, depth = 28, 28, 1  # MNIST images are 28x28 and greyscale
    num_classes = 10  # there are 10 classes (1 per digit)

    # load dataset
    (trainX, trainY), (testX, testY) = mnist.load_data()
    # reshape dataset to have a single channel

    trainX = trainX.reshape(num_train, height * width)  # Flatten data to 1D
    testX = testX.reshape(num_test, height * width)  # Flatten data to 1D
    trainX = trainX.astype('float32')
    testX = testX.astype('float32')
    trainX /= 255  # Normalise data to [0, 1] range
    testX /= 255  # Normalise data to [0, 1] range

    return trainX, trainY


def loadCifar():
    num_train = 50000  # there are 60000 training examples in CIFAR
    num_test = 10000  # there are 10000 test examples in CIFAR

    height, width, depth = 32, 32, 3  # MNIST images are 32x32 and greyscale
    num_classes = 10  # there are 10 classes (1 per digit)

    # load dataset
    (trainX, trainY), (testX, testY) = cifar10.load_data()
    # reshape dataset to have a single channel

    trainX = trainX.reshape(num_train, height * width * 3)  # Flatten data to 1D
    testX = testX.reshape(num_test, height * width * 3)  # Flatten data to 1D
    trainX = trainX.astype('float32')
    testX = testX.astype('float32')
    trainX /= 255  # Normalise data to [0, 1] range
    testX /= 255  # Normalise data to [0, 1] range

    return trainX, trainY

def load_images_from_df(df):
    images = df.iloc[:, 1:].values.astype('float32')
    images = images.reshape(-1, 28, 28, 1)
    images /= 255.0
    return images
def load_labels_from_df(df):
    labels = df.iloc[:, 0].values.astype('int32')
    return labels

def load_raw_fashionmnist(path='DataSets/'):

    train_dir = os.path.join(path, 'fashion-mnist_train.csv')
    test_dir = os.path.join(path, 'fashion-mnist_test.csv')
    train_df = pd.read_csv(train_dir)
    test_df = pd.read_csv(test_dir)

    return train_df, test_df

def loadFashionMnist():
    num_train = 60000  # there are 60000 training examples in MNIST
    num_test = 10000  # there are 10000 test examples in MNIST

    height, width, depth = 28, 28, 1  # MNIST images are 28x28 and greyscale
    num_classes = 10  # there are 10 classes (1 per digit)

    train_df, test_df = load_raw_fashionmnist()

    train_images = load_images_from_df(train_df)
    train_labels = load_labels_from_df(train_df)
    test_images = load_images_from_df(test_df)
    test_labels = load_labels_from_df(test_df)

    train_labels = train_df.iloc[:, 0].values
    test_labels = test_df.iloc[:, 0].values

    trainX = train_images.reshape(num_train, height * width)  # Flatten data to 1D
    testX = test_images.reshape(num_test, height * width)  # Flatten data to 1D

    return trainX, train_labels

# Генерация XOR
def make_xor(n_samples=1000):
    np.random.seed(42)
    X_xor = np.array([np.random.rand(2) for k in range(n_samples)])
    y_xor = np.logical_xor(X_xor[:, 0] > 0.5, X_xor[:, 1] > 0.5).astype(int)

    return X_xor, y_xor

# Генерация spirals
def make_spirals(n_samples=1000, noise=0.005, random_state=42):
    np.random.seed(random_state)
    # Генерация углов для спиралей
    theta = np.linspace(0, 5 * np.pi, n_samples // 2)
    # Экспоненциальный рост радиуса
    r = np.exp(0.1 * theta) - 1

    # Первая спираль (класс 0)
    x0 = r * np.cos(theta)
    y0 = r * np.sin(theta)
    spiral0 = np.column_stack((x0, y0))
    class0 = np.zeros(len(spiral0))

    # Вторая спираль (класс 1), сдвинутая на 0.5π и с чуть другим ростом
    theta_shifted = theta + np.pi
    #r_shifted = np.exp(0.12 * theta_shifted)
    x1 = r * np.cos(theta_shifted)
    y1 = r * np.sin(theta_shifted)
    spiral1 = np.column_stack((x1, y1))
    class1 = np.ones(len(spiral1))

    # Объединяем данные
    X = np.vstack((spiral0, spiral1))
    y = np.hstack((class0, class1))

    # Добавляем шум
    X += noise * np.random.randn(*X.shape)
    return X, y

def make_random(n_samples=1000):
    # Генерация random
    np.random.seed(42)
    random_x = np.array([np.random.rand(2) for k in range(n_samples)])
    random_y = np.zeros(n_samples)
    for i in range(math.floor(n_samples/2)):
        random_y[i] = 1

    return random_x, random_y