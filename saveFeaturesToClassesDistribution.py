from matplotlib import pyplot as plt
import numpy as np
from scipy.stats import entropy
import math

def saveFeaturesToClassesDistribution(dataSet, cl, binsPerFeature, taskName):
    """

    :type totalBins: int
    """
    nObjects = dataSet.shape[0]
    nClasses = len(cl)
    nFeatures = dataSet.shape[1]

    figure = plt.figure()
    px = 1 / plt.rcParams['figure.dpi']

    if nFeatures > 20:
        return

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
    plt.close(figure)

    return