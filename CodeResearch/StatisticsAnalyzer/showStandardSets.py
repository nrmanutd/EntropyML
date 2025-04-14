import math

import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets

from CodeResearch.dataSets import make_spirals, make_random, make_xor

nSamples = 1000

# Генерация blobs
X_blobs, y_blobs = datasets.make_blobs(n_samples=nSamples, centers=2, n_features=2, random_state=42)

# Генерация moons
X_moons, y_moons = datasets.make_moons(n_samples=nSamples, noise=0.1, random_state=42)

# Генерация circles
X_circles, y_circles = datasets.make_circles(n_samples=nSamples, factor=0.5, noise=0.1, random_state=42)

# Генерация XOR
X_xor, y_xor = make_xor(nSamples)

X_spirals, y_spirals = make_spirals(n_samples=nSamples, noise=0.05, random_state=42)

random_x, random_y = make_random(nSamples)


# Функция для визуализации
def plot_dataset(X, y, title):

    blue = "#377eb8"
    orange = "#ff7f00"

    colors = []

    for i in range(len(y)):
        colors.append(blue if y[i] == y[0] else orange)

    plt.scatter(X[:, 0], X[:, 1], c=colors, edgecolor='k')
    plt.title(title)
    plt.show()

# Визуализация всех датасетов
#plot_dataset(X_blobs, y_blobs, "Blobs")
#plot_dataset(X_moons, y_moons, "Moons")
#plot_dataset(X_circles, y_circles, "Circles")
#plot_dataset(X_xor, y_xor, "XOR")
plot_dataset(X_spirals, y_spirals, "Spirals")
#plot_dataset(random_x, random_y, "Random")