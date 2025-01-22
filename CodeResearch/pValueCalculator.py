import math

import numpy as np

from CodeResearch.DiviserCalculation.getDiviserFast import getMaximumDiviserFast
from CodeResearch.DiviserCalculation.getDiviserRTreeStochastic import getMaximumDiviserRTreeStochastic
from CodeResearch.permutationHelpers import GetObjectsPerClass, permuteDataSet


def getDataSetOfTwoClasses(currentObjects, dataSet, target, iClass, jClass):
    iClassIdx = np.where(target == iClass)[0]
    jClassIdx = np.where(target == jClass)[0]

    partIClass = len(iClassIdx) / (len(iClassIdx) + len(jClassIdx))

    iObjectsCount = math.floor(partIClass * currentObjects)
    jObjectsCount = currentObjects - iObjectsCount

    iClassObjects = GetObjectsPerClass(target, iClass, iObjectsCount)
    jClassObjects = GetObjectsPerClass(target, jClass, jObjectsCount)

    nFeatures = dataSet.shape[1]
    newSet = np.zeros((iObjectsCount + jObjectsCount, nFeatures))

    newSet[0:iObjectsCount, :] = dataSet[iClassObjects, :]
    newSet[iObjectsCount:(iObjectsCount + jObjectsCount), :] = dataSet[jClassObjects, :]

    newTarget = np.zeros((iObjectsCount + jObjectsCount))
    newTarget[0:iObjectsCount] = target[iClassObjects]
    newTarget[iObjectsCount: (iObjectsCount + jObjectsCount)] = target[jObjectsCount]

    return newSet, newTarget

def calcPValueStochastic(currentObjects, dataSet, target, iClass, jClass, nAttempts):
    newSet, newTarget = getDataSetOfTwoClasses(currentObjects, dataSet, target, iClass, jClass)

    print('Calculating stochastic target...')
    targetValue = getMaximumDiviserRTreeStochastic(newSet, newTarget)[0]
    print(targetValue)

    values = np.zeros(nAttempts)
    for iAttempt in range(0, nAttempts):
        print('Permutation attempt {:}/{:}'.format(iAttempt, nAttempts))
        permutedSet, permutedTarget = permuteDataSet(newSet, newTarget)
        values[iAttempt] = getMaximumDiviserRTreeStochastic(permutedSet, permutedTarget)[0]

    valuesIdx = np.argsort(values)
    print(values[valuesIdx])

    pValue = len(np.where(values < targetValue)[0]) / len(values)
    return pValue

def calcPValueFast(currentObjects, dataSet, target, iClass, jClass, nAttempts):
    newSet, newTarget = getDataSetOfTwoClasses(currentObjects, dataSet, target, iClass, jClass)

    targetValue = getMaximumDiviserRTreeStochastic(newSet, newTarget)[0]
    print(targetValue)

    values = np.zeros(nAttempts)
    for iAttempt in range(0, nAttempts):
        permutedSet, permutedTarget = permuteDataSet(newSet, newTarget)
        values[iAttempt] = getMaximumDiviserFast(permutedSet, permutedTarget)[0]

    valuesIdx = np.argsort(values)
    print(values[valuesIdx])

    pValue = len(np.where(values < targetValue)[0]) / len(values)
    return pValue