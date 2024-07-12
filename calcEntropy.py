import math

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import iqr

import scipy
from scipy.stats import entropy

def calcAndVisualize(dataSet, cl, binsPerFeature, taskName):
    """

    :type totalBins: int
    """
    nObjects = dataSet.shape[0]
    nClasses = len(cl)
    nFeatures = dataSet.shape[1]

    plt.figure()
    px = 1 / plt.rcParams['figure.dpi']
    fig, ax = plt.subplots(nFeatures, nClasses + 1, tight_layout=True, figsize=(1920 * px, 1280 * px))

    for iFeature in np.arange(nFeatures):
        h = np.histogram(dataSet[:, iFeature], bins=binsPerFeature[iFeature], density=False)
        entBase = entropy(h[0])
        buckets = h[1]

        style = {'facecolor': 'none', 'edgecolor': 'C0', 'linewidth': 3}
        axAll = ax[iFeature, 0]
        axAll.hist(dataSet[:, iFeature], bins=buckets, **style)
        axAll.text(0.8, 0.8, "{0:2.1f}, {1:2.1f}".format(entBase, math.e ** entBase), horizontalalignment='center', verticalalignment='center',
                   transform=axAll.transAxes)
        for jClass in np.arange(nClasses):
            axTemp = ax[iFeature, jClass + 1]
            data = dataSet[cl[jClass], iFeature]

            style = {'facecolor': 'none', 'edgecolor': 'C{0}'.format((jClass + 1) % 10), 'linewidth': 3}
            axTemp.hist(data, bins=buckets, **style)
            h = np.histogram(data, bins=buckets, density=False)

            ent = entropy(h[0])
            axTemp.text(0.6, 0.8, "{0:2.1f}, {1:2.1f}, {2}".format(ent, math.e ** ent, len(cl[jClass])),
                        horizontalalignment='center', verticalalignment='center', transform=axTemp.transAxes)

    # plot the xdata locations on the x axis:

    fig.text(0.5, 0, 'Классы ({0}_{1}_{2})'.format(taskName, nObjects, nFeatures), ha='center')
    fig.text(0, 0.5, 'Признаки', va='center', rotation='vertical')

    plt.savefig('{0}_{1}_{2}_histogramms.png'.format(taskName, nObjects, nFeatures), format='png')
    plt.close()

def calcConditionalEntropy(dataSet, buckets, classIdx):
    nObjects = dataSet.shape[0]
    nClasses = len(classIdx)
    nFeatures = dataSet.shape[1]

    conditionalEntropy = 0.0

    for iFeature in np.arange(nFeatures):
        for jClass in np.arange(nClasses):
            idx = np.digitize(dataSet[classIdx[jClass], iFeature], buckets[iFeature])
            v, c = np.unique(idx, return_counts=True)
            ent = entropy(c / sum(c))
            p = sum(c) / nObjects
            conditionalEntropy += p * ent

    return conditionalEntropy
def calcConditionalMultiVarianceEntropy(dataSet, objectBins, classIdx):
    nObjects = dataSet.shape[0]
    nClasses = len(classIdx)

    conditionalMultiEntropy = np.zeros(nClasses, dtype=float)

    for iClass in np.arange(nClasses):
        mve = calcMultiVarianceEntropy(dataSet[classIdx[iClass], :], objectBins)
        conditionalMultiEntropy[iClass] = len(classIdx[iClass])/nObjects * mve

    return sum(conditionalMultiEntropy)

def calcMultiVarianceEntropy(dataSet, objectBins):
    nObjects = dataSet.shape[0]
    nFeatures = dataSet.shape[1]

    objectBinsIndexes = np.zeros((nObjects, nFeatures))

    for iFeature in np.arange(nFeatures):
        objectBinsIndexes[:, iFeature] = np.digitize(dataSet[:, iFeature], objectBins[iFeature])

    #objectBinsIndexes = 2 ** objectBinsIndexes
    bucketIdx = [''] * nObjects

    for iObject in np.arange(nObjects):
        bucketIdx[iObject] = ''.join(str(x) for x in objectBinsIndexes[iObject, :])

    eBins, cBins = np.unique(bucketIdx, return_counts=True)
    ent = entropy(cBins/sum(cBins))

    return ent

def getOptimalBinsCount(dataSet):
    nFeatures = dataSet.shape[1]
    nObjects = dataSet.shape[0]

    bins = np.arange(nFeatures)
    for i in np.arange(nFeatures):
        if nObjects > 1000:
            bins[i] = math.ceil(math.log(nObjects, 2) + 1)
        else:
            width = 2 * iqr(dataSet[:, i]) /(nObjects ** (1/3))
            bins[i] = math.ceil((max(dataSet[:, i]) - min(dataSet[:, i])) / width)
    return bins

def calculateAndVisualizeSeveralEntropies(dataSet, target, taskName):
    print('Starting task: {0}'.format(taskName))

    nFeatures = dataSet.shape[1]
    nObjects = dataSet.shape[0]

    entries, counts = np.unique(target, return_counts=True)

    cl = []
    for iClass in np.arange(0, len(entries)):
        cl.append(np.where(target == entries[iClass])[0])

    upperBins = getOptimalBinsCount(dataSet)
    bins = np.arange(2, max(upperBins) + 1)

    efficiency = np.zeros(len(bins), dtype=float)
    simple = np.zeros(len(bins), dtype=float)
    conditional = np.zeros(len(bins), dtype=float)
    multiEntropy = np.zeros(len(bins), dtype=float)
    multiConditionalEntropy = np.zeros(len(bins), dtype=float)

    ct = 0.0
    for iClass in np.arange(len(cl)):
        ct += math.log(len(cl[iClass])) * len(cl[iClass]) / nObjects

    for iBins in np.arange(len(bins)):
        if (bins[iBins] %10 == 0):
            print('{2}: {0} of {1}'.format(bins[iBins], len(bins), taskName))

        objectBins = []
        simpleEntropy = np.zeros(nFeatures + 1, dtype=float)

        for i in np.arange(nFeatures):
            iFeatureBin = min(bins[iBins], upperBins[i])
            h = np.histogram(dataSet[:, i], bins=iFeatureBin)
            objectBins.append(h[1])
            simpleEntropy[i] = entropy(h[0]/sum(h[0]))

        simpleEntropy[nFeatures] = entropy(counts / nObjects)

        simple[iBins] = sum(simpleEntropy)
        conditional[iBins] = calcConditionalEntropy(dataSet, objectBins, cl)
        multiEntropy[iBins] = calcMultiVarianceEntropy(dataSet, objectBins)
        multiConditionalEntropy[iBins] = calcConditionalMultiVarianceEntropy(dataSet, objectBins, cl)
        efficiency[iBins] = conditional[iBins] - multiConditionalEntropy[iBins]

        if abs(multiConditionalEntropy[iBins]/ct - 1) < 0.01:
            break

    idx = np.where(multiConditionalEntropy != 0)[0]

    bins = bins[idx]
    simple = simple[idx]
    conditional = conditional[idx]
    multiEntropy = multiEntropy[idx]
    multiConditionalEntropy = multiConditionalEntropy[idx]
    efficiency = efficiency[idx]

    SaveEntropiesToFigure(bins, simple, conditional, multiEntropy, multiConditionalEntropy, efficiency, taskName, nObjects, nFeatures, ct)
    calcAndVisualize(dataSet, cl, upperBins, taskName)

    return

def SaveEntropiesToFigure(bins, simple, conditional, multiEntropy, multiConditionalEntropy, efficiency, taskName, nObjects, nFeatures, ct):

    plt.figure()
    # binToDisplay = max(bins)
    px = 1 / plt.rcParams['figure.dpi']
    fig, ax = plt.subplots(3, 1, sharex=True, tight_layout=True, figsize=(1920 * px, 1280 * px))

    ax[0].plot(bins, efficiency)
    ax[0].title.set_text('Efficiency (simple conditional - multi conditional)')
    ax[0].grid()

    # ct = 0.0
    # for iClass in np.arange(len(cl)):
    #    ct += -math.log(len(cl[iClass])/nObjects) * len(cl[iClass]) / nObjects

    # ax[1].plot(bins, simple)
    # ax[1].plot(bins, np.ones(len(bins)) * (math.log(nObjects) * nFeatures + ct))
    # ax[1].plot(np.ones(len(bins)) * (binToDisplay), simple)
    # ax[1].title.set_text('Simple entropy')

    # ct = 0.0
    # for iClass in np.arange(len(cl)):
    #    ct += math.log(len(cl[iClass])) * len(cl[iClass]) / nObjects

    # ax[2].plot(bins, conditional)
    # ax[2].plot(bins, np.ones(len(bins)) * ct * nFeatures)
    # ax[2].plot(np.ones(len(bins)) * (binToDisplay), conditional)
    # ax[2].title.set_text('Simple conditional entropy')

    # ax[3].plot(bins, multiEntropy)
    # ax[3].plot(bins, np.ones(len(bins)) * math.log(nObjects))
    # ax[3].plot(np.ones(len(bins)) * (binToDisplay), multiEntropy)
    # ax[3].title.set_text('Multi entropy')

    # ax[4].plot(bins, multiConditionalEntropy)
    # ax[4].plot(bins, np.ones(len(bins)) * ct)
    # ax[4].plot(np.ones(len(bins)) * (binToDisplay), multiConditionalEntropy)
    # ax[4].title.set_text('Multi conditional')

    ax[1].plot(bins, simple)
    ax[1].plot(bins, conditional)
    # ax[1].plot(np.ones(len(bins)) * (binToDisplay), simple)
    # ax[1].plot(bins, np.ones(len(bins)) * ct * nFeatures)
    ax[1].title.set_text('Simple and conditional entropy')
    ax[1].text(bins[0], 2 / 3 * max(simple) + 1 / 3 * simple[0],
               'Task: {0}\nObjects: {1}\nFeatures: {2}\nUpper: {3:3.2f}'.format(taskName, nObjects, nFeatures,                                                                                 ct * nFeatures))
    ax[1].grid()

    ax[2].plot(bins, multiEntropy)
    ax[2].plot(bins, multiConditionalEntropy)
    # ax[2].plot(bins, np.ones(len(bins)) * ct)
    # ax[2].plot(np.ones(len(bins)) * (binToDisplay), multiEntropy)
    ax[2].text(bins[0], 2 / 3 * max(multiEntropy) + 1 / 3 * multiEntropy[0], 'Upper: {0:3.2f}'.format(ct))
    ax[2].title.set_text('Simple and conditional multi entropy')
    ax[2].grid()

    plt.savefig('{0}_{1}_{2}_entropies.png'.format(taskName, nObjects, nFeatures), format='png')

    return