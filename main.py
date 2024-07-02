import math

import pandas as pd

from calcEntropy import calcAndVisualize, calcSimpleEntropy, calcConditionalEntropy
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

def test():
    np.random.seed(19680801)

    N = 100
    r0 = 0.6
    x = 0.9 * np.random.rand(N)
    y = 0.9 * np.random.rand(N)
    area = (20 * np.random.rand(N)) ** 2  # 0 to 10 point radii
    c = np.sqrt(area)
    r = np.sqrt(x ** 2 + y ** 2)
    area1 = np.ma.masked_where(r < r0, area)
    area2 = np.ma.masked_where(r >= r0, area)
    plt.scatter(x, y, s=area1, marker='^', c=c)
    plt.scatter(x, y, s=area2, marker='o', c=c)
    # Show the boundary between the regions:
    theta = np.arange(0, np.pi / 2, 0.01)
    plt.plot(r0 * np.cos(theta), r0 * np.sin(theta))

    plt.show()
#test()

def calculateMultiEntropyIris():
    # fetch dataset
    iris = fetch_ucirepo(id=53)

    # data (as pandas dataframes)
    y = iris.data.targets
    entries, counts = np.unique(y, return_counts=True)
    nClasses = len(entries)

    dataSet = np.array(iris.data.features)
    nFeatures = dataSet.shape[1]
    nObjects = dataSet.shape[0]

    idx1 = np.arange(50)
    idx2 = np.arange(50, 100)
    idx3 = np.arange(100, 150)

    cl = np.array([idx1, idx2, idx3])

    #totalBins = math.ceil(nObjects ** (1 / nFeatures))
    # totalBins = 10

    bins = np.arange(math.ceil(nObjects ** (1 / nFeatures)), 100)
    efficiency = np.zeros(len(bins), dtype=float)
    simple = np.zeros(len(bins), dtype=float)
    conditional = np.zeros(len(bins), dtype=float)

    for iBins in np.arange(len(bins)):
        totalBins = bins[iBins]

        simpleEntropy = np.zeros(nFeatures + 1, dtype=float)
        conditionalEntropy = np.zeros(nFeatures, dtype=float)

        for i in np.arange(nFeatures):
            h = np.histogram(dataSet[:, i], bins=totalBins, density=True)
            ent = entropy(h[0])
            simpleEntropy[i] = ent

            for jClass in np.arange(nClasses):
                hConditional = np.histogram(dataSet[cl[jClass], i], bins=h[1], density=True)
                conditionalEntropy[i] += len(cl[jClass]) / nObjects * entropy(hConditional[0])

        simpleEntropy[nFeatures] = entropy(counts/nObjects)

        #print('Total bins: ', totalBins)
        #print('Simple entropy: ', simpleEntropy)
        #print('Conditional entropy: ', conditionalEntropy)

        #print('Simple number of letters: ', math.e ** simpleEntropy)
        #print('Conditional number of letters: ', math.e ** conditionalEntropy)

        simpleTotalWords = math.e ** sum(simpleEntropy)
        conditionalTotalWords = math.e ** sum(conditionalEntropy)

        efficiency[iBins] = simpleTotalWords/conditionalTotalWords
        simple[iBins] = simpleTotalWords
        conditional[iBins] = conditionalTotalWords

        #print('Simple: {0: 5.1f}, Conditional: {1: 5.1f}, Efficiency: {2: 5.1f}'.format(simpleTotalWords, conditionalTotalWords, simpleTotalWords/conditionalTotalWords))

    fig, ax = plt.subplots(1, 1, tight_layout=True)

    ax.plot(bins, efficiency)
    #ax[1].plot(bins, simple)
    #ax[2].plot(bins, conditional)
    #ax[3].plot(simple, conditional)

    plt.show()

    return

calculateMultiEntropyIris()