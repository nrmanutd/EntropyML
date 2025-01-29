import numpy as np

def calculateSlope(xValues, yValues):
    first = np.diff(yValues) / np.diff(xValues)
    second = np.diff(first)

    absSecond = np.abs(second).tolist()

    maxDelta = max(absSecond)
    ind = absSecond.index(maxDelta)

    return ind + 1

def calculateSlopeGradient(xValues, yValues):
    first = np.gradient(yValues, xValues)
    second = np.gradient(first, xValues)

    absSecond = np.abs(second).tolist()

    maxDelta = max(absSecond)
    ind = absSecond.index(maxDelta)

    return ind

def getBestSlopeMax(indicies):
    nInd, nCounts = np.unique(indicies, return_counts=True)

    maxInd = nInd[0]
    maxCount = nCounts[0]
    for i in range(1, len(nCounts)):
        if nCounts[i] > maxCount:
            maxInd = nInd[i]
            maxCount = nCounts[i]

    return maxInd


def getBestSlopeMedian(indicies):
    return np.median(indicies)