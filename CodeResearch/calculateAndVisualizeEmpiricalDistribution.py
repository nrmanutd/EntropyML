import math

import numpy as np
from matplotlib import pyplot as plt

from CodeResearch.calculateDistributionDelta import calculateDistributionDelta, calculateRademacherComplexity


def calculateAndVisualizeEmpiricalDistribution(dataSet, target, taskName):

    print('Starting task: {0}'.format(taskName))

    nAttempts = 100
    probability = 0.95

    nObjects = dataSet.shape[0]

    maxObjects = nObjects/2
    minObjects = min(1, maxObjects)

    avgDistributions = np.arange(maxObjects, dtype=float)
    avgDistributions2 = np.arange(maxObjects, dtype=float)
    avgDistributions3 = np.arange(maxObjects, dtype=float)
    accuracy = np.arange(maxObjects, dtype=float)

    sigmas = np.arange(maxObjects, dtype=float)
    mcDiarmids = np.arange(maxObjects, dtype=float)

    delta = 1 - probability

    for iDistribution in np.arange(minObjects, maxObjects, dtype=int):
        print('Calculating for {0}'.format(iDistribution))

        currentObjects = iDistribution + 1

        mcDiarmids[iDistribution] = math.sqrt(math.log(1/delta) / currentObjects)
        #avg3, sigma = calculateDistributionDelta(dataSet, currentObjects, nAttempts)
        avg3 = 0
        avg, sigma, avg2, acc = calculateRademacherComplexity(dataSet, currentObjects, nAttempts, target)

        avgDistributions[iDistribution] =avg
        sigmas[iDistribution] = sigma
        avgDistributions2[iDistribution] = avg2
        avgDistributions3[iDistribution] = avg3
        accuracy[iDistribution] = acc

    xNumberOfObjects = np.arange(1, maxObjects + 1)

    # plot lines
    plt.plot(xNumberOfObjects, avgDistributions, label="Avg")
    plt.plot(xNumberOfObjects, avgDistributions2, label="Avg2")
    plt.plot(xNumberOfObjects, avgDistributions3, label="Avg3")
    plt.plot(xNumberOfObjects, accuracy, label="Accuracy")
    #plt.plot(xNumberOfObjects, avgDistributions + mcDiarmids, label="+m")
    #plt.plot(xNumberOfObjects, avgDistributions - mcDiarmids, label="-m")
    plt.plot(xNumberOfObjects, avgDistributions + sigmas, label="+s")
    plt.plot(xNumberOfObjects, avgDistributions - sigmas, label="-s")

    plt.legend()
    plt.show()

    return