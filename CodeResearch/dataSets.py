import math
import os

import pandas as pd
import numpy as np
import torch
from torchvision import datasets, transforms

#from tensorflow.keras.datasets import cifar10
#from tensorflow.keras.datasets import cifar100
#from tensorflow.keras.datasets import mnist
#from tensorflow.keras.datasets import fashion_mnist

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

def loadMnist_cnn():
    (trainX, trainY), (testX, testY) = mnist.load_data()  # trainX: [60000,28,28]

    # float32 + нормализация
    trainX = trainX.astype(np.float32) / 255.0
    testX  = testX.astype(np.float32) / 255.0

    # добавить канал: [N, 1, 28, 28] (PyTorch-формат NCHW)
    trainX = np.expand_dims(trainX, axis=1)
    testX  = np.expand_dims(testX, axis=1)

    return trainX, trainY

import torch
from torchvision import datasets, transforms

def loadMnist_torch(root="./data"):
    transform = transforms.ToTensor()  # uint8 -> float32, [0,1], CHW

    ds_tr = datasets.MNIST(root=root, train=True,  download=True, transform=transform)
    ds_te = datasets.MNIST(root=root, train=False, download=True, transform=transform)

    Xtr = torch.stack([ds_tr[i][0] for i in range(len(ds_tr))], dim=0)  # [N,1,28,28]
    ytr = torch.tensor([ds_tr[i][1] for i in range(len(ds_tr))], dtype=torch.int64)

    Xte = torch.stack([ds_te[i][0] for i in range(len(ds_te))], dim=0)
    yte = torch.tensor([ds_te[i][1] for i in range(len(ds_te))], dtype=torch.int64)

    return Xtr, ytr, Xte, yte


def loadFashionMnist():
    num_train = 60000  # there are 60000 training examples in MNIST
    num_test = 10000  # there are 10000 test examples in MNIST

    height, width, depth = 28, 28, 1  # Fashion MNIST images are 28x28 and greyscale
    num_classes = 10  # there are 10 classes (1 per digit)

    # load dataset
    (trainX, trainY), (testX, testY) = fashion_mnist.load_data()
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

def loadCifar_cnn():

    # load dataset
    (trainX, trainY), (testX, testY) = cifar10.load_data()
    # reshape dataset to have a single channel

    trainX = trainX.astype('float32')
    testX = testX.astype('float32')
    trainX /= 255  # Normalise data to [0, 1] range
    testX /= 255  # Normalise data to [0, 1] range

    return trainX, trainY

def loadCifar100():
    num_train = 50000  # there are 60000 training examples in CIFAR
    num_test = 10000  # there are 10000 test examples in CIFAR

    height, width, depth = 32, 32, 3  # MNIST images are 32x32 and greyscale

    # load dataset
    (trainX, trainY), (testX, testY) = cifar100.load_data(label_mode='fine')
    # reshape dataset to have a single channel

    trainX = trainX.reshape(num_train, height * width * 3)  # Flatten data to 1D
    testX = testX.reshape(num_test, height * width * 3)  # Flatten data to 1D
    trainX = trainX.astype('float32')/255
    testX = testX.astype('float32')/255

    return trainX, trainY

def loadCifar100_cnn():

    # load dataset
    (trainX, trainY), (testX, testY) = cifar100.load_data(label_mode='fine')
    # reshape dataset to have a single channel

    trainX = trainX.astype('float32')/255
    testX = testX.astype('float32')/255

    return trainX, trainY

import torch
from torchvision import datasets, transforms

def loadCifar10_torch(root="./data", normalize_to_01=True):
    """
    Возвращает:
        Xtr: [50000, 3, 32, 32], float32 (ToTensor => уже в [0,1])
        ytr: [50000], int64
        Xte: [10000, 3, 32, 32], float32
        yte: [10000], int64
    """
    transform = transforms.ToTensor()

    ds_tr = datasets.CIFAR10(root=root, train=True,  download=True, transform=transform)
    ds_te = datasets.CIFAR10(root=root, train=False, download=True, transform=transform)

    Xtr = torch.stack([ds_tr[i][0] for i in range(len(ds_tr))], dim=0)
    ytr = torch.tensor([ds_tr[i][1] for i in range(len(ds_tr))], dtype=torch.int64)

    Xte = torch.stack([ds_te[i][0] for i in range(len(ds_te))], dim=0)
    yte = torch.tensor([ds_te[i][1] for i in range(len(ds_te))], dtype=torch.int64)

    # ToTensor уже сделал [0,1]. Если normalize_to_01=False — оставляем как есть.
    return Xtr, ytr, Xte, yte


def loadCifar100_torch(root="./data"):
    """
    Возвращает:
        Xtr: [50000, 3, 32, 32], float32 в [0,1]
        ytr: [50000], int64
        Xte: [10000, 3, 32, 32], float32 в [0,1]
        yte: [10000], int64
    """
    transform = transforms.ToTensor()

    ds_tr = datasets.CIFAR100(root=root, train=True,  download=True, transform=transform)
    ds_te = datasets.CIFAR100(root=root, train=False, download=True, transform=transform)

    Xtr = torch.stack([ds_tr[i][0] for i in range(len(ds_tr))], dim=0)
    ytr = torch.tensor([ds_tr[i][1] for i in range(len(ds_tr))], dtype=torch.int64)

    Xte = torch.stack([ds_te[i][0] for i in range(len(ds_te))], dim=0)
    yte = torch.tensor([ds_te[i][1] for i in range(len(ds_te))], dtype=torch.int64)

    return Xtr, ytr, Xte, yte


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

def load_megamarket(path):
    category_col = "cat_level_1"
    embedding_col = "embedding"
    data = pd.read_parquet(path)

    y = np.array(data[category_col])
    x = np.vstack(data[embedding_col].values)

    return x, y

def load_proteins_bin(path):
    d = pd.read_csv(path)
    selected_columns = ['1m2z', '1pbq', '1xoq', '2rh1', '2vt4', '2ydo', '2z5x', '3b66', '3kk6', '3ln1', '3rze', '4djh',
                        '4ey7', '4iar', '4mqs', '4n6h', '5cxv', '5i71', '5tvn', '5u09', '5va1', '6cm4', '6kpf', '6kux',
                        '6lqa', '6pdj', '6x3x', '6y1z', '7f8y', '7kwe', '7ljd', '7wc9', '7xnk', '7ym8', '8e9y', '8ef6',
                        '8fhs', '8pjk', '8st0', '8wty', '8xvk', '8yn3', '9eo4', 'V1A']
    x = d[selected_columns].to_numpy()

    y = d['-lgLD50, mol/kg'].to_numpy()
    median_val = np.median(y)
    yy = np.where(y >= median_val, 1, 0)

    return x, yy

def load_proteins(path):
    d = pd.read_csv(path)
    selected_columns = ['1m2z', '1pbq', '1xoq', '2rh1', '2vt4', '2ydo', '2z5x', '3b66', '3kk6', '3ln1', '3rze', '4djh',
                        '4ey7', '4iar', '4mqs', '4n6h', '5cxv', '5i71', '5tvn', '5u09', '5va1', '6cm4', '6kpf', '6kux',
                        '6lqa', '6pdj', '6x3x', '6y1z', '7f8y', '7kwe', '7ljd', '7wc9', '7xnk', '7ym8', '8e9y', '8ef6',
                        '8fhs', '8pjk', '8st0', '8wty', '8xvk', '8yn3', '9eo4', 'V1A']
    x = d[selected_columns].to_numpy()

    y = d['-lgLD50, mol/kg'].to_numpy()

    median_val = np.median(y)
    y = np.where(y >= median_val, 1, 0)

    return x, y

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

def generate_lin_reg_dataset(
        k: float = 1.0, n_samples: int = 1000, noise: float = 0.05
    ) -> tuple[np.ndarray, np.ndarray]:

        x1 = np.random.uniform(0, 1, n_samples)
        x2 = k * x1 + np.random.normal(0, noise**0.5, n_samples)
        X = np.column_stack([x1, x2])
        y = (X[:, 0] > X[:, 1]).astype(int)
        return X, y

def make_random(n_samples=1000):
    # Генерация random
    np.random.seed(42)
    random_x = np.array([np.random.rand(2) for k in range(n_samples)])
    random_y = np.zeros(n_samples)
    for i in range(math.floor(n_samples/2)):
        random_y[i] = 1

    return random_x, random_y