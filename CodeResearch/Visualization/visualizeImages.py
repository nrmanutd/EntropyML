import math
import os

import pandas as pd
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.datasets import mnist
import numpy as np

from CodeResearch.dataSets import loadMnist, load_raw_fashionmnist


def get_image_of_index(i, target):
    for idx in range(len(target)):
        if target[idx] == i:
            return idx


def show_objects(dataSet, labels):
    rows = 2
    columns = 5

    enc = LabelEncoder()
    target = enc.fit_transform(np.ravel(labels))

    fig, axes = plt.subplots(rows, columns, figsize=(5, 5))
    for i in range(rows * columns):

        idx = get_image_of_index(i, target)

        image = dataSet[idx]
        label = labels[idx]

        rowIdx = i // columns
        colIdx = i % columns

        axes[rowIdx, colIdx].imshow(image.squeeze(), cmap='gray')
        axes[rowIdx, colIdx].set_title(f"L: {label}")
        axes[rowIdx, colIdx].axis('off')

    plt.show()

#(mnistSet, mnistLabels), (testX, testY) = mnist.load_data()
#(mnistSet, mnistLabels), (testX, testY) = cifar10.load_data()
(mnistSet, mnistLabels) = load_raw_fashionmnist('../../../DataSets/')

show_objects(mnistSet, mnistLabels)