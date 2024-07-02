import math

import pandas as pd

from calcEntropy import calcAndVisualize, calcSimpleEntropy, calcConditionalEntropy, calcMultiVarianceEntropy, \
    calcConditionalMultiVarianceEntropy, calculateAndVisualizeSeveralEntropies
from ucimlrepo import fetch_ucirepo
from scipy.stats import norm, entropy
import plotly.express as px

import pandas
import matplotlib.pyplot as plt
from pandas.plotting import parallel_coordinates

import numpy as np

def showIris():
    # fetch dataset
    iris = fetch_ucirepo(id=53)

    # data (as pandas dataframes)
    y = iris.data.targets

    X1 = np.array(iris.data.features)
    idx1 = np.arange(0, 50)
    idx2 = np.arange(50, 100)
    idx3 = np.arange(100, 150)

    cl = np.array([idx1, idx2, idx3])
    calcAndVisualize(X1, cl)

#showIris()

def showCircles():

    elements = 100

    N = 100
    r0 = 0.6
    x = 0.9 * np.random.rand(N)
    y = 0.9 * np.random.rand(N)
    area = (20 * np.random.rand(N)) ** 2
    c = np.sqrt(area)

    dataSet = np.zeros((elements, 4))
    for i in np.arange(elements):
        radious = (6 if i >= elements / 2 else 4)

        value = np.random.uniform(-radious, radious)
        dataSet[i, 0] = value
        dataSet[i, 1] = 2 if value > 0 else -2 #math.sqrt(radious*radious - value*value) * (1 if value >= np.random.uniform(-1, 1) else -1)
        #dataSet[i, 2] = 1 if radious == 6 else -1
        dataSet[i, 2] = value #np.random.uniform(-radious, radious)
        dataSet[i, 3] = 1 if value > 0 else -1

    idx1 = np.arange(elements/2, dtype=int)
    idx2 = np.arange(elements/2, elements, dtype=int)
    cl = np.array([idx1, idx2])
    df = pd.DataFrame(dataSet, columns=['X1', 'X2', 'X3', 'Y'])

    parallel_coordinates(df, class_column='Y', colormap=plt.get_cmap("Set2"))

    # Hide the color scale that is useless in this case
    #fig.update_layout(coloraxis_showscale=False)

    plt.show()

    #calcAndVisualize(dataSet, cl, 10)
#showCircles()

def calculateMultiEntropyIris():
    # fetch dataset
    iris = fetch_ucirepo(id=53)

    # data (as pandas dataframes)
    y = iris.data.targets
    dataSet = np.array(iris.data.features)

    calculateAndVisualizeSeveralEntropies(dataSet, y)

    return

calculateMultiEntropyIris()