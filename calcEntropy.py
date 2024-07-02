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
    return 1
def calcConditionalMultiVarianceEntropy(dataSet, buckets, classIdx):
    return 1

def calcSimpleEntropy(variable, buckets):
    return 1

def calcMultiVarianceEntropy(dataSet, buckets, classIdx):
    return 1