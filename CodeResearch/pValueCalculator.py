import math
import time

import numpy as np
from numba import cuda
from joblib import Parallel, delayed

from CodeResearch.Cuda.cudaHelpers import updateSortedSetNumba
from CodeResearch.DiviserCalculation.diviserHelpers import getSortedSet, GetValuedAndBoolTarget
from CodeResearch.DiviserCalculation.getDiviserFast import getMaximumDiviserFast
from CodeResearch.DiviserCalculation.getDiviserFastCuda import getMaximumDiviserFastCuda, \
    getMaximumDiviserFastCudaPreloadedToDevice
from CodeResearch.DiviserCalculation.getDiviserFastNumba import getMaximumDiviserFastNumba
from CodeResearch.DiviserCalculation.getDiviserRTreeStochastic import getMaximumDiviserRTreeStochastic
from CodeResearch.calcModelEstimations import calcModel
from CodeResearch.permutationHelpers import GetObjectsPerClass

def getDataSetIndexesOfTwoClasses(currentObjects, target, iClass, jClass):
    iClassIdx = np.where(target == iClass)[0]
    jClassIdx = np.where(target == jClass)[0]

    # print('Total objects: {:}, iClass: {:}, jClass: {:}, currentObjects: {:}'.format(dataSet.shape[0], len(iClassIdx), len(jClassIdx), currentObjects))

    partIClass = len(iClassIdx) / (len(iClassIdx) + len(jClassIdx))

    iObjectsCount = math.ceil(partIClass * currentObjects) if partIClass < 0.5 else math.floor(
        partIClass * currentObjects)
    jObjectsCount = currentObjects - iObjectsCount

    iClassObjects = GetObjectsPerClass(target, iClass, iObjectsCount)
    jClassObjects = GetObjectsPerClass(target, jClass, jObjectsCount)

    return iClassObjects, jClassObjects

def getDataSetOfTwoClasses(currentObjects, dataSet, target, iClass, jClass):

    iClassObjects, jClassObjects = getDataSetIndexesOfTwoClasses(currentObjects, target, iClass, jClass)

    iObjectsCount = len(iClassObjects)
    jObjectsCount = len(jClassObjects)

    nFeatures = dataSet.shape[1]
    newSet = np.zeros((iObjectsCount + jObjectsCount, nFeatures))

    newSet[0:iObjectsCount, :] = dataSet[iClassObjects, :]
    newSet[iObjectsCount:(iObjectsCount + jObjectsCount), :] = dataSet[jClassObjects, :]

    newTarget = np.zeros(iObjectsCount + jObjectsCount)
    newTarget[0:iObjectsCount] = target[iClassObjects]
    newTarget[iObjectsCount: (iObjectsCount + jObjectsCount)] = target[jClassObjects]

    return newSet, newTarget


def calcPValueStochastic(currentObjects, dataSet, target, iClass, jClass, nAttempts):
    iObjects = list(np.where(target == iClass)[0])
    jObjects = list(np.where(target == jClass)[0])
    objectsIdx = iObjects + jObjects
    precision = calcModel(dataSet[objectsIdx, :], min(currentObjects, len(objectsIdx)), 10, target[objectsIdx])[
        'accuracy']

    parallel = Parallel(n_jobs=-1, return_as="generator")
    output_generator = parallel(
        delayed(calcStochasticParallel)(currentObjects, dataSet, target, iClass, jClass, iAttempt) for iAttempt in range(nAttempts))

    values = np.array(list(output_generator))

    targetValue = math.sqrt(2 * math.log(currentObjects) / currentObjects)
    pValue = len(np.where(values < targetValue)[0]) / len(values)

    return min(pValue, 1 - pValue), targetValue, values, precision

    newSet, newTarget = getDataSetOfTwoClasses(currentObjects, dataSet, target, iClass, jClass)

    #print('Calculating stochastic target...')
    #targetValue = getMaximumDiviserRTree(newSet, newTarget)[0]
    targetValue = getMaximumDiviserRTreeStochastic(newSet, newTarget)[0]
    #print(targetValue)
    totalTime = 0.0

    values = np.zeros(nAttempts)
    #parallel = Parallel(n_jobs=2, return_as="generator")
    #output_generator = parallel(delayed(calcRTStochastic)(newSet, newTarget, iAttempt) for iAttempt in range(nAttempts))

    #for iAttempt in range(0, nAttempts):
        #print('Permutation attempt {:}/{:}'.format(iAttempt, nAttempts))

        #s1 = time.time()
        #values[iAttempt] = getMaximumDiviserRTreeStochastic(permutedSet, permutedTarget)[0]
        #e1 = time.time()
        #totalTime += (e1 - s1)
        #print('Time is {:.2f}'.format(e1 - s1))
    #values = list()
    #valuesIdx = np.argsort(values)
    #print(values[valuesIdx])
    #print('Total time for {:} permutations is {:.2f}'.format(nAttempts, totalTime))

    pValue = len(np.where(values < targetValue)[0]) / len(values)
    return min(pValue, 1 - pValue)

def calcPValueFast(currentObjects, dataSet, target, iClass, jClass, nAttempts, nModelAttempts, beta):
    iObjects = list(np.where(target == iClass)[0])
    jObjects = list(np.where(target == jClass)[0])
    objectsIdx = iObjects + jObjects
    precision = calcModel(dataSet[objectsIdx, :], min(currentObjects, len(objectsIdx)), nModelAttempts, target[objectsIdx])

    values = np.zeros(nAttempts)
    for iAttempt in range(nAttempts):
        if iAttempt % 100 == 0:
            print('Attempt # ', iAttempt)

        newSet, newTarget = getDataSetOfTwoClasses(currentObjects, dataSet, target, iClass, jClass)
        values[iAttempt] = getMaximumDiviserFast(newSet, newTarget)[0]

    targetValue = math.sqrt(2 * math.log(currentObjects) / currentObjects)
    pValue = len(np.where(values < targetValue)[0]) / len(values)
    quantile = np.quantile(values, beta)
    quantileUp = np.quantile(values, 1 - beta)

    return quantile, quantileUp, targetValue, values, (precision['accuracy'][0], precision['modelSigma'][0])

def calcPValueFastNumba(currentObjects, dataSet, target, iClass, jClass, nAttempts, nModelAttempts, beta):
    iObjects = list(np.where(target == iClass)[0])
    jObjects = list(np.where(target == jClass)[0])
    objectsIdx = iObjects + jObjects
    precision = calcModel(dataSet[objectsIdx, :], min(currentObjects, len(objectsIdx)), nModelAttempts, target[objectsIdx])

    values = np.zeros(nAttempts)
    currentTime = time.time()

    for iAttempt in range(nAttempts):
        if iAttempt % 10 == 0:
            print('Attempt #' + str(iAttempt) + ' Time: ' + str(time.time() - currentTime))
            currentTime = time.time()

        newSet, newTarget = getDataSetOfTwoClasses(currentObjects, dataSet, target, iClass, jClass)
        values[iAttempt] = getMaximumDiviserFastNumba(newSet, newTarget)[0]

    targetValue = math.sqrt(2 * math.log(currentObjects) / currentObjects)
    #pValue = len(np.where(values < targetValue)[0]) / len(values)
    quantile = np.quantile(values, beta)
    quantileUp = np.quantile(values, 1 - beta)

    return quantile, quantileUp, targetValue, values, (precision['accuracy'][0], precision['modelSigma'][0])

def calcPValueFastCuda(currentObjects, dataSet, target, iClass, jClass, nAttempts, nModelAttempts, beta):
    iObjects = list(np.where(target == iClass)[0])
    jObjects = list(np.where(target == jClass)[0])
    objectsIdx = iObjects + jObjects
    precision = calcModel(dataSet[objectsIdx, :], min(currentObjects, len(objectsIdx)), nModelAttempts, target[objectsIdx])

    values = np.zeros(nAttempts)
    currentTime = time.time()

    ds = dataSet[objectsIdx, :]
    t = target[objectsIdx]

    nClasses, counts = np.unique(t, return_counts=True)
    valuedTarget1, boolValuedTarget1 = GetValuedAndBoolTarget(t, nClasses[0], 1 / counts[0], -1 / counts[1])
    valuedTarget2, boolValuedTarget2 = GetValuedAndBoolTarget(t, nClasses[1], 1 / counts[1], -1 / counts[0])

    sds1 = getSortedSet(ds, valuedTarget1)
    sds2 = getSortedSet(ds, valuedTarget2)

    for iAttempt in range(nAttempts):
        if iAttempt % 10 == 0:
            print('Attempt #' + str(iAttempt) + ' Time: ' + str(time.time() - currentTime))
            currentTime = time.time()

        #newSet, newTarget = getDataSetOfTwoClasses(currentObjects, dataSet, target, iClass, jClass)

        iClassIdx, jClassIdx = getDataSetIndexesOfTwoClasses(currentObjects, target, iClass, jClass)
        idx = list(iClassIdx) + list(jClassIdx)

        dsClasses = ds[idx, :]
        dsClasses_device = cuda.to_device(dsClasses)
        tClasses = t[idx]

        vt1 = valuedTarget1[idx]
        bvt1 = boolValuedTarget1[idx]
        ss1 = updateSortedSetNumba(sds1, idx)
        ss1_device = cuda.to_device(ss1)

        vt2 = valuedTarget2[idx]
        bvt2 = boolValuedTarget2[idx]
        ss2 = updateSortedSetNumba(sds2, idx)
        ss2_device = cuda.to_device(ss2)

        #values[iAttempt] = getMaximumDiviserFastCuda(newSet, newTarget)[0]
        values[iAttempt] = getMaximumDiviserFastCudaPreloadedToDevice(dsClasses, dsClasses_device, tClasses, ss1, ss1_device, ss2, ss2_device)[0]

    targetValue = math.sqrt(2 * math.log(currentObjects) / currentObjects)
    #pValue = len(np.where(values < targetValue)[0]) / len(values)
    quantile = np.quantile(values, beta)
    quantileUp = np.quantile(values, 1 - beta)

    return quantile, quantileUp, targetValue, values, (precision['accuracy'][0], precision['modelSigma'][0])

def calcPValueFastParallel(currentObjects, dataSet, target, iClass, jClass, nAttempts, nModelAttempts, beta):
    iObjects = list(np.where(target == iClass)[0])
    jObjects = list(np.where(target == jClass)[0])

    objectsIdx = iObjects + jObjects
    precision = calcModel(dataSet[objectsIdx, :], min(currentObjects, len(objectsIdx)), nModelAttempts, target[objectsIdx])

    parallel = Parallel(n_jobs=-1, return_as="generator")
    output_generator = parallel(delayed(calcRTFastParallel)(currentObjects, dataSet, target, iClass, jClass, iAttempt) for iAttempt in range(nAttempts))

    values = np.array(list(output_generator))

    targetValue = math.sqrt(2 * math.log(currentObjects) / currentObjects)
    #pValue = len(np.where(values < targetValue)[0]) / len(values)
    quantile = np.quantile(values, beta)
    quantileUp = np.quantile(values, 1 - beta)

    return quantile, quantileUp, targetValue, values, (precision['accuracy'][0], precision['modelSigma'][0])

def calcRTFastParallel(currentObjects, dataSet, target, iClass, jClass, iAttempt):
    if iAttempt%100 == 0:
        print("Attempt# {:}".format(iAttempt))
    newSet, newTarget = getDataSetOfTwoClasses(currentObjects, dataSet, target, iClass, jClass)

    return getMaximumDiviserFastNumba(newSet, newTarget)[0]
    #return getMaximumDiviserFast(newSet, newTarget)[0]

def calcStochasticParallel(currentObjects, dataSet, target, iClass, jClass, iAttempt):
    if iAttempt%100 == 0:
        print("Attempt# {:}".format(iAttempt))
    newSet, newTarget = getDataSetOfTwoClasses(currentObjects, dataSet, target, iClass, jClass)
    return getMaximumDiviserRTreeStochastic(newSet, newTarget)[0]
