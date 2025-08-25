import math
import time
import torch

import numpy as np
from joblib import Parallel, delayed
from numba import cuda
from sklearn.preprocessing import LabelEncoder

from CodeResearch.Cuda.cudaHelpers import filterSortedSetByIndex
from CodeResearch.DiviserCalculation.diviserHelpers import getSortedSet, GetValuedAndBoolTarget, prepareDataSet, \
    GetValuedTarget
from CodeResearch.DiviserCalculation.getDiviserFast import getMaximumDiviserFast
from CodeResearch.DiviserCalculation.getDiviserFastCuda import getMaximumDiviserFastCudaCore
from CodeResearch.DiviserCalculation.getDiviserFastNumba import getMaximumDiviserFastNumba, \
    getMaximumDiviserFastNumbaCore
from CodeResearch.DiviserCalculation.getDiviserRTreeStochastic import getMaximumDiviserRTreeStochastic
from CodeResearch.calcModelEstimations import calcModel, calcNN, calcXGBoost
from CodeResearch.Helpers.permutationHelpers import GetObjectsPerClass


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
    return getDataSetOfTwoClassesCore(dataSet, target, iClassObjects, jClassObjects)

def getDataSetOfTwoClassesCore(dataSet, target, iClassObjects, jClassObjects):

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

def calcPValueFastPro(currentObjects, dataSet, target, iClass, jClass, nAttempts, calculateKS=True, randomPermutation=False, calculateModel=False):
    nFeatures = dataSet.shape[1]

    if not torch.cuda.is_available():
        return calcPValuesCpuNumba(currentObjects, dataSet, target, iClass, jClass, nAttempts, calculateKS, randomPermutation, calculateModel)

    if nFeatures < 1000:
        return calcPValueFastNumba(currentObjects, dataSet, target, iClass, jClass, nAttempts, calculateKS, randomPermutation, calculateModel)
    else:
        return calcPValueFastCuda(currentObjects, dataSet, target, iClass, jClass, nAttempts, calculateKS, randomPermutation, calculateModel)

def calcPValuesCpuNumba(currentObjects, dataSet, target, iClass, jClass, nAttempts, calculateKS = True, randomPermutation=False, calculateModel=False):
    iObjects = list(np.where(target == iClass)[0])
    jObjects = list(np.where(target == jClass)[0])
    objectsIdx = iObjects + jObjects

    values = np.zeros(nAttempts)
    NNvalues = np.zeros(nAttempts)

    currentTime = time.time()

    twoClassObjects = np.arange(len(objectsIdx))
    ds = dataSet[objectsIdx, :]
    ds = prepareDataSet(ds)
    t = target[objectsIdx]

    enc = LabelEncoder()

    preparationTime = 0
    ksTime = 0
    NNTime = 0

    for iAttempt in range(nAttempts):
        if iAttempt % 10 == 0:
            print('Attempt #' + str(iAttempt) + ' Time: ' + str(time.time() - currentTime) + ' Preparation time: ' + str(preparationTime), ' KS time: ' + str(ksTime) + ' NN time: ' + str(NNTime))
            preparationTime = 0
            ksTime = 0
            NNTime = 0
            currentTime = time.time()

        t1 = time.time()

        iClassIdx, jClassIdx = getDataSetIndexesOfTwoClasses(currentObjects, t, iClass, jClass)
        idx = np.concatenate((iClassIdx, jClassIdx))

        tClasses = t[idx]
        dsClasses = ds[idx, :]
        preparationTime += (time.time() - t1)

        if calculateModel:
            t2 = time.time()
            tt= enc.fit_transform(np.ravel(t))
            tClasses = tt[idx]

            if randomPermutation:
                tClasses = np.random.permutation(tClasses)

            testIdx = np.setdiff1d(twoClassObjects, idx)
            testDs = ds[testIdx, :]
            testTClasses = tt[testIdx]

            preparationTime += (time.time() - t2)
            t2 = time.time()
            #NNvalues[iAttempt] = calcNN(dsClasses, tClasses, testDs, testTClasses)
            NNvalues[iAttempt] = calcXGBoost(dsClasses, tClasses, testDs, testTClasses)
            NNTime += (time.time() - t2)

        if calculateKS:
            t2 = time.time()

            if randomPermutation:
                tClasses = np.random.permutation(tClasses)

            nClasses, counts = np.unique(tClasses, return_counts=True)
            vt1 = GetValuedTarget(tClasses, nClasses[0], 1 / counts[0], -1 / counts[1])
            vt2 = GetValuedTarget(tClasses, nClasses[1], 1 / counts[1], -1 / counts[0])

            sds1 = getSortedSet(dsClasses, vt1)
            sds2 = getSortedSet(dsClasses, vt2)

            preparationTime += (time.time() - t2)
            t2 = time.time()
            values[iAttempt] = getMaximumDiviserFastNumbaCore(dsClasses, tClasses, vt1, sds1, vt2, sds2)[0]
            ksTime += (time.time() - t2)

    return values, NNvalues

def calcPValueFastNumba(currentObjects, dataSet, target, iClass, jClass, nAttempts, calculateKS = True, randomPermutation=False, calculateModel=False):
    iObjects = list(np.where(target == iClass)[0])
    jObjects = list(np.where(target == jClass)[0])
    objectsIdx = iObjects + jObjects

    values = np.zeros(nAttempts)
    NNvalues = np.zeros(nAttempts)

    currentTime = time.time()

    twoClassObjects = np.arange(len(objectsIdx))
    ds = dataSet[objectsIdx, :]
    ds = prepareDataSet(ds)
    t = target[objectsIdx]

    enc = LabelEncoder()

    nClasses, counts = np.unique(t, return_counts=True)
    valuedTarget1, boolValuedTarget1 = GetValuedAndBoolTarget(t, nClasses[0], 1 / counts[0], -1 / counts[1])
    valuedTarget2, boolValuedTarget2 = GetValuedAndBoolTarget(t, nClasses[1], 1 / counts[1], -1 / counts[0])

    sds1 = getSortedSet(ds, valuedTarget1)
    sds2 = getSortedSet(ds, valuedTarget2)

    sds1_device = cuda.to_device(sds1)
    sds2_device = cuda.to_device(sds2)

    preparationTime = 0
    ksTime = 0
    NNTime = 0

    for iAttempt in range(nAttempts):
        if iAttempt % 10 == 0:
            print('Attempt #' + str(iAttempt) + ' Time: ' + str(time.time() - currentTime) + ' Preparation time: ' + str(preparationTime), ' KS time: ' + str(ksTime) + ' NN time: ' + str(NNTime))
            preparationTime = 0
            ksTime = 0
            NNTime = 0
            currentTime = time.time()

        #newSet, newTarget = getDataSetOfTwoClasses(currentObjects, dataSet, target, iClass, jClass)
        # values[iAttempt] = getMaximumDiviserFastNumba(newSet, newTarget)[0]

        t1 = time.time()

        iClassIdx, jClassIdx = getDataSetIndexesOfTwoClasses(currentObjects, t, iClass, jClass)
        idx = np.concatenate((iClassIdx, jClassIdx))

        tClasses = t[idx]

        dsClasses = ds[idx, :]
        preparationTime += (time.time() - t1)

        if calculateModel:
            t2 = time.time()
            tt= enc.fit_transform(np.ravel(t))
            tClasses = tt[idx]

            if randomPermutation:
                tClasses = np.random.permutation(tClasses)

            testIdx = np.setdiff1d(twoClassObjects, idx)
            testDs = ds[testIdx, :]
            testTClasses = tt[testIdx]

            preparationTime += (time.time() - t2)
            t2 = time.time()
            #NNvalues[iAttempt] = calcNN(dsClasses, tClasses, testDs, testTClasses)
            NNvalues[iAttempt] = calcXGBoost(dsClasses, tClasses, testDs, testTClasses)
            NNTime += (time.time() - t2)

        if calculateKS:
            t2 = time.time()

            if randomPermutation:
                tClasses = np.random.permutation(tClasses)

            nClasses, counts = np.unique(tClasses, return_counts=True)
            vt1 = GetValuedTarget(tClasses, nClasses[0], 1 / counts[0], -1 / counts[1])
            vt2 = GetValuedTarget(tClasses, nClasses[1], 1 / counts[1], -1 / counts[0])
            ss1 = filterSortedSetByIndex(sds1_device, sds1.shape[0], sds1.shape[1], idx)
            ss2 = filterSortedSetByIndex(sds2_device, sds2.shape[0], sds2.shape[1], idx)

            preparationTime += (time.time() - t2)
            t2 = time.time()
            values[iAttempt] = getMaximumDiviserFastNumbaCore(dsClasses, tClasses, vt1, ss1, vt2, ss2)[0]
            ksTime += (time.time() - t2)

    return values, NNvalues

def calcPValueFastCuda(currentObjects, dataSet, target, iClass, jClass, nAttempts, calculateKS = True, randomPermutation = False, calculateModel = False):
    iObjects = list(np.where(target == iClass)[0])
    jObjects = list(np.where(target == jClass)[0])
    objectsIdx = iObjects + jObjects

    values = np.zeros(nAttempts)
    NNvalues = np.zeros(nAttempts)

    currentTime = time.time()
    enc = LabelEncoder()

    objectsIdx = np.array(objectsIdx)

    twoClassObjects = np.arange(len(objectsIdx))
    ds = dataSet[objectsIdx, :]
    ds = prepareDataSet(ds)
    t = target[objectsIdx]

    nClasses, counts = np.unique(t, return_counts=True)
    valuedTarget1, boolValuedTarget1 = GetValuedAndBoolTarget(t, nClasses[0], 1 / counts[0], -1 / counts[1])
    valuedTarget2, boolValuedTarget2 = GetValuedAndBoolTarget(t, nClasses[1], 1 / counts[1], -1 / counts[0])

    sds1 = getSortedSet(ds, valuedTarget1)
    sds1_device = cuda.to_device(sds1)

    sds2 = getSortedSet(ds, valuedTarget2)
    sds2_device = cuda.to_device(sds2)

    preparationTime = 0
    nnTime = 0
    ksTime = 0

    for iAttempt in range(nAttempts):
        if iAttempt % 10 == 0:
            print('Attempt #' + str(iAttempt) + ' Time: ' + str(time.time() - currentTime) + ' Preparation time: ' + str(preparationTime) + ' KS cuda calculation: ' + str(ksTime) + ' NN time calculation: ' + str(nnTime))
            currentTime = time.time()

            nnTime = 0
            ksTime = 0
            preparationTime = 0

        #newSet, newTarget = getDataSetOfTwoClasses(currentObjects, dataSet, target, iClass, jClass)
        #values[iAttempt] = getMaximumDiviserFastCuda(newSet, newTarget)[0]
        #continue

        t1 = time.time()

        iClassIdx, jClassIdx = getDataSetIndexesOfTwoClasses(currentObjects, t, iClass, jClass)
        idx = np.concatenate((iClassIdx, jClassIdx))

        dsClasses = ds[idx, :]

        preparationTime += (time.time() - t1)

        if calculateModel:
            t2 = time.time()
            tt= enc.fit_transform(np.ravel(t))

            tClasses = tt[idx]

            if randomPermutation:
                tClasses = np.random.permutation(tClasses)

            testIdx = np.setdiff1d(twoClassObjects, idx)
            testDs = ds[testIdx, :]
            testTClasses = tt[testIdx]

            preparationTime += (time.time() - t2)

            t2 = time.time()
            NNvalues[iAttempt] = calcNN(dsClasses, tClasses, testDs, testTClasses)
            nnTime += (time.time() - t2)

        if calculateKS:
            t2 = time.time()
            tClasses = t[idx]

            if randomPermutation:
                tClasses = np.random.permutation(tClasses)

            nClasses, counts = np.unique(tClasses, return_counts=True)
            vt1, bvt1 = GetValuedAndBoolTarget(tClasses, nClasses[0], 1 / counts[0], -1 / counts[1])
            vt2, bvt2 = GetValuedAndBoolTarget(tClasses, nClasses[1], 1 / counts[1], -1 / counts[0])

            ss1 = filterSortedSetByIndex(sds1_device, sds1.shape[0], sds1.shape[1], idx)
            ss2 = filterSortedSetByIndex(sds2_device, sds2.shape[0], sds2.shape[1], idx)

            ss1_device = cuda.to_device(ss1)
            ss2_device = cuda.to_device(ss2)

            dsClasses_device = cuda.to_device(dsClasses)
            preparationTime += time.time() - t2

            t2 = time.time()
            values[iAttempt] = getMaximumDiviserFastCudaCore(dsClasses, dsClasses_device, tClasses, ss1, ss1_device, vt1, bvt1, ss2, ss2_device, vt2, bvt2)[0]
            ksTime += time.time() - t2

    return values, NNvalues

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
