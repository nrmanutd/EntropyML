import math

import numpy as np
import matplotlib.pyplot as plt
from math import log, e
from scipy.stats import entropy

def calcAndVisualize(dataSet, cl):
    """

    :type totalBins: int
    """
    nClasses = cl.shape[0]
    nFeatures = dataSet.shape[1]
    totalBins = math.ceil(dataSet.shape[0] ** (1/nFeatures))

    print (totalBins)

    # changing the style of the histogram bars just to make it
    # very clear where the boundaries of the bins are:
    style = {'facecolor': 'none', 'edgecolor': 'C0', 'linewidth': 3}

    fig, ax = plt.subplots(nFeatures, nClasses + 1, sharey=True, tight_layout=True)

    for iFeature in np.arange(nFeatures):
        h = np.histogram(dataSet[:, iFeature], bins=totalBins, density=False)
        entBase = entropy(h[0])
        buckets = h[1]

        axAll = ax[iFeature, 0]
        axAll.hist(dataSet[:, iFeature], bins=buckets, **style)
        axAll.text(0.8, 0.8, "{0:2.1f}, {1:2.1f}".format(entBase, math.e ** entBase), horizontalalignment='center', verticalalignment='center',
                   transform=axAll.transAxes)
        for jClass in np.arange(nClasses):
            axTemp = ax[iFeature, jClass + 1]
            data = dataSet[cl[jClass], iFeature]

            axTemp.hist(data, bins=buckets, **style)
            h = np.histogram(data, bins=buckets, density=False)

            ent = entropy(h[0])
            axTemp.text(0.8, 0.8, "{0:2.1f}, {1:2.1f}".format(ent, math.e ** ent),
                        horizontalalignment='center', verticalalignment='center', transform=axTemp.transAxes)

    # plot the xdata locations on the x axis:

    fig.text(0.5, 0, 'Классы', ha='center')
    fig.text(0, 0.5, 'Признаки', va='center', rotation='vertical')

    plt.show()

def calcConditionalEntropy(dataSet, buckets, classIdx):
    nObjects = dataSet.shape[0]
    nClasses = len(classIdx)
    nFeatures = dataSet.shape[1]

    conditionalEntropy = 0.0

    for iFeature in np.arange(nFeatures):
        for jClass in np.arange(nClasses):
            idx = np.digitize(dataSet[classIdx[jClass], iFeature], buckets[:, iFeature])
            v, c = np.unique(idx, return_counts=True)
            val = c / sum(c)
            conditionalEntropy += sum(c) / nObjects * entropy(c / sum(c))

    return conditionalEntropy
def calcConditionalMultiVarianceEntropy(dataSet, objectBins, classIdx):
    nObjects = dataSet.shape[0]
    nClasses = len(classIdx)

    conditionalMultiEntropy = np.zeros(nClasses, dtype=float)

    for iClass in np.arange(nClasses):
        mve = calcMultiVarianceEntropy(dataSet[classIdx[iClass], :], objectBins)
        conditionalMultiEntropy[iClass] = len(classIdx[iClass])/nObjects * mve

    return sum(conditionalMultiEntropy)

def calcSimpleEntropy(variable, buckets):
    return 1

def calcMultiVarianceEntropy(dataSet, objectBins):
    nObjects = dataSet.shape[0]
    nFeatures = dataSet.shape[1]

    objectBinsIndexes = np.zeros((nObjects, nFeatures))

    for iFeature in np.arange(nFeatures):
        objectBinsIndexes[:, iFeature] = np.digitize(dataSet[:, iFeature], objectBins[:, iFeature])

    objectBinsIndexes = 10 ** objectBinsIndexes
    bucketIdx = np.zeros(nObjects)

    for iObject in np.arange(nObjects):
        bucketIdx[iObject] = sum(objectBinsIndexes[iObject, :])

    eBins, cBins = np.unique(bucketIdx, return_counts=True)

    return entropy(cBins/sum(cBins))

def calculateAndVisualizeSeveralEntropies(dataSet, target):
    nFeatures = dataSet.shape[1]
    nObjects = dataSet.shape[0]

    entries, counts = np.unique(target, return_counts=True)

    cl = []
    for iClass in np.arange(0, len(entries)):
        cl.append(np.where(target == entries[iClass])[0])

    # totalBins = math.ceil(nObjects ** (1 / nFeatures))
    # totalBins = 10

    bins = np.arange(math.ceil(nObjects ** (1 / nFeatures)), 100)

    efficiency = np.zeros(len(bins), dtype=float)
    simple = np.zeros(len(bins), dtype=float)
    conditional = np.zeros(len(bins), dtype=float)
    multiEntropy = np.zeros(len(bins), dtype=float)
    multiConditionalEntropy = np.zeros(len(bins), dtype=float)

    for iBins in np.arange(len(bins)):
        totalBins = bins[iBins]
        objectBins = np.zeros((totalBins + 1, nFeatures))
        simpleEntropy = np.zeros(nFeatures + 1, dtype=float)

        for i in np.arange(nFeatures):
            h = np.histogram(dataSet[:, i], bins=totalBins, density=True)
            objectBins[:, i] = h[1]
            simpleEntropy[i] = entropy(h[0])

        simpleEntropy[nFeatures] = entropy(counts / nObjects)
        # print('Total bins: ', totalBins)
        # print('Simple entropy: ', simpleEntropy)
        # print('Conditional entropy: ', conditionalEntropy)

        # print('Simple number of letters: ', math.e ** simpleEntropy)
        # print('Conditional number of letters: ', math.e ** conditionalEntropy)

        simple[iBins] = sum(simpleEntropy)
        conditional[iBins] = calcConditionalEntropy(dataSet, objectBins, cl)
        multiEntropy[iBins] = calcMultiVarianceEntropy(dataSet, objectBins)
        multiConditionalEntropy[iBins] = calcConditionalMultiVarianceEntropy(dataSet, objectBins, cl)

        # efficiency[iBins] = math.e ** sum(simpleEntropy) / math.e ** sum(conditionalEntropy)
        efficiency[iBins] = conditional[iBins] - multiConditionalEntropy[iBins]

        # print('Simple: {0: 5.1f}, Conditional: {1: 5.1f}, Efficiency: {2: 5.1f}'.format(simpleTotalWords, conditionalTotalWords, simpleTotalWords/conditionalTotalWords))

    fig, ax = plt.subplots(5, 1, sharex=True, tight_layout=True)

    ax[0].plot(bins, efficiency)
    ax[1].plot(bins, simple)
    ax[2].plot(bins, conditional)
    ax[3].plot(bins, multiEntropy)
    ax[4].plot(bins, multiConditionalEntropy)

    # ax[3].plot(simple, conditional)

    plt.show()
    return