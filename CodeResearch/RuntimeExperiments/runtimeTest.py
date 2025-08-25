import math
import time
import numpy as np

from CodeResearch.DiviserCalculation.getDiviserFastNumba import getMaximumDiviserFastNumba
from CodeResearch.pValueCalculator import getDataSetIndexesOfTwoClasses
from CodeResearch.Helpers.permutationHelpers import extractDataSet


def runtimeForClassPairs(xx, yy):

    classes = np.unique(yy)
    nClasses = len(classes)
    times = []

    for iClass in range(nClasses):
        for jClass in range(iClass):
            iObjects = list(np.where(yy == iClass)[0])
            jObjects = list(np.where(yy == jClass)[0])
            objectsIdx = iObjects + jObjects

            iObjectsCount = len(iObjects)
            jObjectsCount = len(jObjects)

            totalObjects = (iObjectsCount + jObjectsCount)
            currentObjects = math.floor(totalObjects / 2)

            ds = xx[objectsIdx, :]
            t = yy[objectsIdx]

            iClassIdx, jClassIdx = getDataSetIndexesOfTwoClasses(currentObjects, t, iClass, jClass)
            idx = np.concatenate((iClassIdx, jClassIdx))

            tClasses = t[idx]
            dsClasses = ds[idx, :]

            t1 = time.time()
            getMaximumDiviserFastNumba(dsClasses, tClasses)
            t2 = time.time()

            times.append(t2 - t1)

    return times


def runtimeTest(x, y, n, k, attempts, firstTime):

    hotNumber = 3
    times = []

    totalIterations = attempts + hotNumber if firstTime else attempts

    for i in range(totalIterations):
        if i < hotNumber and firstTime:
            print(f'Hot attempt # {i}')

        xx, yy = extractDataSet(x.copy(), y.copy(), n, k)

        t1 = time.time()
        allPairsTimes = runtimeForClassPairs(xx, yy)
        t2 = time.time()

        if i < hotNumber and firstTime:
            continue

        curAttempts = i - hotNumber if firstTime else i
        print(f'Attempt #{curAttempts} of {attempts}, elapsed time: {t2 - t1} s')
        times.append(t2 - t1)
        times.append(np.average(allPairsTimes))

    return np.array(times)