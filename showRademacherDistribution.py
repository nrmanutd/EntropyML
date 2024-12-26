import math

import numpy as np
from matplotlib import pyplot as plt


def bin_array(num, m):
    """Convert a positive integer num into an m-bit bit vector"""
    return np.array(list(np.binary_repr(num).zfill(m))).astype(np.int8)

def generateVectors(nDimension, nVectors):

    total = np.arange(2 ** nDimension)
    result = np.zeros((nVectors, nDimension))

    #vectors = np.random.permutation(total)
    vectors = total
    vectors[nVectors - 1] = (vectors[nVectors - 1] + 500) % (2 ** nDimension)
    for i in np.arange(nVectors):
        result[i, :] = 2 * bin_array(vectors[i], nDimension) - 1

    return result

def showRademacherDistirbution(nDimension, nVectors, nTimes):

    bitVectorsMatrix = np.zeros((2**nDimension, nDimension))

    for nVector in np.arange(2 ** nDimension):
        bitVectorsMatrix[nVector, :] = 2 * bin_array(nVector, nDimension) - 1

    uVectors = generateVectors(nDimension, nVectors)

    rademacherValues = np.zeros((nTimes, 2 ** nDimension))

    for nIteration in np.arange(nTimes):
        for nVector in np.arange(2 ** nDimension):
            rademacherValues[nIteration, nVector] = float(np.dot(bitVectorsMatrix[nVector, :], uVectors[0, :])) / nDimension

            if (nVectors > 1):
                for nUVector in np.arange(1, uVectors.shape[0]):
                    product = float(np.dot(bitVectorsMatrix[nVector, :], uVectors[nUVector, :])) / nDimension
                    rademacherValues[nIteration, nVector] = max(rademacherValues[nIteration, nVector], product)

    fig = plt.figure(figsize=(9, 4), layout="constrained")
    axs = fig.subplots(1, 1)

    rademacherValues = rademacherValues.ravel()
    rademacherComplexity = np.average(rademacherValues)
    sigma = math.sqrt(np.var(rademacherValues))

    # Cumulative distributions.
    #axs.ecdf(rademacherValues, label="Rademacher uVectors = {0}, dim = {1}, rep = {2}".format(nVectors, nDimension, nTimes))
    h = axs.hist(rademacherValues, 25, density=True, histtype="step", label="Rademacher uVectors = {0}, dim = {1}, rep = {2}".format(nVectors, nDimension, nTimes))
    x = np.linspace(rademacherValues.min(), rademacherValues.max())
    y = ((1 / (np.sqrt(2 * np.pi) * sigma)) *
         np.exp(-0.5 * (1 / sigma * (x - rademacherComplexity)) ** 2))
    #y = y.cumsum()
    #y /= y[-1]
    axs.plot(x, y, "k--", linewidth=1.5, label="Normal distribution")
    #axs.plot(x, np.ones(len(x))*rademacherComplexity, "k-", linewidth=1.5, label="Rademacher complexity {0:3.2f}".format(rademacherComplexity))

    # Label the figure.
    fig.suptitle("Cumulative Rademacher distributions")
    #for ax in axs:
    axs.grid(True)
    axs.legend()
    axs.set_xlabel("Rademacher values")
    axs.set_ylabel("Probability of occurrence")
    axs.label_outer()

    plt.show()

    #mlResults = {'logit {0:1.2f}'.format(regressionAccuracy): logitResults,
    #             'xgBoost {0:1.2f}'.format(xgBoostAccuracy): xgboostResults
    return