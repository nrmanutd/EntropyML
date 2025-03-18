import time

import numba as nb
import numpy as np
import cupy as cp
from numba import jit, prange, cuda, njit

from CodeResearch.Cuda.cudaHelpers import getSortedSetCuda
from CodeResearch.DiviserCalculation.diviserHelpers import prepareDataSet, getSortedSet, \
    GetValuedAndBoolTarget, fv2s


@jit(nopython=True)
def updateStateOnOtherFeatures(currentState, sortedDataSet, valuedTarget, omitedObjects, omitedDelta, updateState):
    nFeatures = len(currentState)

    for iFeature in range(0, nFeatures):
        for iSortedObject in range(currentState[iFeature], -1, -1):
            iObject = sortedDataSet[iSortedObject, iFeature]

            if not omitedObjects[iObject] and not omitedDelta[iObject]:
                if valuedTarget[iObject] < 0:
                    if updateState:
                        currentState[iFeature] = iSortedObject
                    break

                omitedDelta[iObject] = True

    return currentState, omitedDelta

@jit(nopython=True)
def makeStepForConcreteFeature(currentState, sortedFeature, feature, valuedTarget, omitedObjects, omitedDelta):

    wasPositive = False
    positiveValue = 0

    for iSortedObject in range(currentState, -1, -1):
        iObject = sortedFeature[iSortedObject]

        if wasPositive and valuedTarget[iObject] < 0 and feature[iObject] != positiveValue:
            return omitedDelta, iSortedObject

        if omitedObjects[iObject]:
            continue

        omitedDelta[iObject] = True

        if valuedTarget[iObject] > 0:
            positiveValue = positiveValue if wasPositive else feature[iObject]
            wasPositive |= True

    return omitedDelta, -1

@jit(nopython=True)
def calcDelta(iFeature, dataSet, sortedDataSet, valuedTarget, currentState, omitedObjects, omitedDelta, updateOmited):

    omitedDelta, idx = makeStepForConcreteFeature(currentState[iFeature], sortedDataSet[:, iFeature],
                                      dataSet[:, iFeature], valuedTarget, omitedObjects, omitedDelta)

    if idx == -1:
        return currentState, omitedObjects, -1, -2

    currentState, omitedDelta = updateStateOnOtherFeatures(currentState, sortedDataSet, valuedTarget, omitedObjects, omitedDelta, updateOmited)

    delta = 0
    addedPositives = 0
    for iObject in range(0, len(omitedDelta)):
        if not omitedDelta[iObject]:
            continue

        omitedDelta[iObject] = False

        if updateOmited:
            omitedObjects[iObject] = True

        delta += valuedTarget[iObject]
        addedPositives += valuedTarget[iObject] if valuedTarget[iObject] > 0 else 0

    return currentState, omitedObjects, addedPositives, delta

@cuda.jit
def calculate_best_index(scores_pos, scores_neg, posValue, negValue, bestIndex, bestValue):
    tid = cuda.threadIdx.x
    bid = cuda.blockIdx.x
    stride = cuda.blockDim.x

    thread_idx = tid + bid * stride

    if thread_idx == 0:
        bestValue[0] = -np.inf
        bestIndex[0] = scores_pos.size

    cuda.syncthreads()

    if thread_idx < scores_pos.size:
        if scores_pos[thread_idx] != -2:
            curValue = scores_pos[thread_idx] * posValue + scores_neg[thread_idx] * negValue
            cuda.atomic.max(bestValue, 0, curValue)

    cuda.syncthreads()

    if thread_idx < scores_pos.size:
        if scores_pos[thread_idx] != -2:
            curValue = scores_pos[thread_idx] * posValue + scores_neg[thread_idx] * negValue
            #print(' curValue = ' + str(curValue))
            if curValue == bestValue[0]:
                cuda.atomic.min(bestIndex, 0, thread_idx)

@cuda.jit
def calculate_scores_kernel(dataSet, sortedDataSet, valuedTargetBool, currentState, omitedObjects, scores_pos, scores_neg, scores_result, posValue, negValue, nObjects, nFeatures, shared_memory_elements):
    shared_mem = cuda.shared.array(shape=0, dtype=np.uint32)
    result_score = cuda.shared.array(shape=2, dtype=np.float64)

    thread_idx = cuda.threadIdx.x
    stride = cuda.blockDim.x

    bits_per_word = 32

    for i in range(thread_idx, shared_memory_elements, stride):
        shared_mem[i] = 0

    cuda.syncthreads()

    for i in range(thread_idx, nObjects, stride):
        if omitedObjects[i]:
            word = i // bits_per_word
            bit = i % bits_per_word
            cuda.atomic.or_(shared_mem, word, 1 << bit)

    cuda.syncthreads()

    curBlock = cuda.blockIdx.x
    startPosition = currentState[curBlock]

    currentResult = -1
    result_score[0] = 0
    result_score[1] = 0

    if thread_idx == 0:
        wasPositive = False
        positiveValue = 0

        for iSortedObject in range(startPosition, -1, -1):
            iObject = sortedDataSet[iSortedObject, curBlock]

            if wasPositive and not valuedTargetBool[iObject] and dataSet[iObject, curBlock] != positiveValue:
                currentResult = iSortedObject
                break

            curObject = iObject // bits_per_word
            bit = iObject % bits_per_word

            if shared_mem[curObject] & (1 << bit) != 0:
                continue

            shared_mem[curObject] = (shared_mem[curObject] | (1 << bit))

            if valuedTargetBool[iObject]:
                result_score[0] += 1
            else:
                result_score[1] += 1

            if valuedTargetBool[iObject]:
                positiveValue = positiveValue if wasPositive else dataSet[iObject, curBlock]
                wasPositive |= True

        if currentResult == -1:
            scores_result[curBlock] = -cp.inf
            scores_pos[curBlock] = -2
            scores_neg[curBlock] = 0

    cuda.syncthreads()

    if scores_pos[curBlock] == -2:
        return

    for iFeature in range(thread_idx, nFeatures, stride):
        for iSortedObject in range(currentState[iFeature], -1, -1):
            iObject = sortedDataSet[iSortedObject, iFeature]

            curObject = iObject // bits_per_word
            bit = iObject % bits_per_word
            wasOmitted = (shared_mem[curObject] & (1 << bit)) != 0

            if not valuedTargetBool[iObject]:
                if not wasOmitted:
                    break

                continue

            if wasOmitted:
                continue

            old_value = cuda.atomic.or_(shared_mem, curObject, 1 << bit)
            if (old_value & (1 << bit)) == 0:
                cuda.atomic.add(result_score, 0, 1)

    cuda.syncthreads()

    if thread_idx == 0:
        scores_result[curBlock] = result_score[0] * posValue + result_score[1] * negValue
        scores_pos[curBlock] = result_score[0]
        scores_neg[curBlock] = result_score[1]

    return

def getNextStepCuda(dataSet_device, sortedDataSet_device, valuedTargetBool_device, classValues, currentState, omitedObjects, scores_pos, scores_neg, scores_result):
    currentState_device = cuda.to_device(currentState)
    omitedObjects_device = cuda.to_device(omitedObjects)

    nObjects = len(omitedObjects)
    nFeatures = len(currentState)

    threads_per_block = 512
    blocks_per_grid = nFeatures

    bits_per_word = 32
    words = (nObjects + bits_per_word - 1) // bits_per_word
    shared_memory_size = words * 4

    calculate_scores_kernel[blocks_per_grid, threads_per_block, 0, shared_memory_size](dataSet_device, sortedDataSet_device, valuedTargetBool_device, currentState_device, omitedObjects_device, scores_pos, scores_neg, scores_result, classValues[0], classValues[1], nObjects, nFeatures, words)
    cuda.synchronize()

    bestValue = cuda.device_array(1, dtype=np.float64)
    bestIndex = cuda.device_array(1, dtype=np.int64)

    threads_per_block = 32
    blocks_per_grid = (nFeatures + threads_per_block - 1) // threads_per_block

    calculate_best_index[blocks_per_grid, threads_per_block](scores_pos, scores_neg, classValues[0], classValues[1], bestIndex, bestValue)
    cuda.synchronize()

    bestIndexHost = bestIndex.copy_to_host()[0]
    #alternativeIndex = cp.argmax(scores_result)

    #print('Cuda logic: ' + str(bestIndexHost) + ' cupy: ' + str(alternativeIndex))

    return bestIndexHost if bestIndexHost != nFeatures else -1

@jit(nopython=True, parallel=True)
def getMinPositives(sortedDataSet, valuedTarget):

    nFeatures = sortedDataSet.shape[1]
    nObjects = sortedDataSet.shape[0]
    firstPositiveObjects = np.zeros(nFeatures, dtype=nb.int64)

    for iFeature in prange(nFeatures):
        for iSortedObject in range(nObjects):
            iObject = sortedDataSet[iSortedObject, iFeature]

            if valuedTarget[iObject] > 0:
                firstPositiveObjects[iFeature] = iSortedObject
                break

    return firstPositiveObjects

def checkStoppingCriteriaCupy(minPositives_device, currentState):

    currentState_device = cp.array(currentState)
    comparison_result = minPositives_device > currentState_device  # Массив булевых значений

    return cp.any(comparison_result).item()

def getMaximumDiviserPerClassFastCuda(dataSet, dataSet_device, valuedTarget, valuedTargetBool, classValues):
    tt1 = time.time()
    sortedDataSet = getSortedSet(dataSet, valuedTarget)
    minPositives = getMinPositives(sortedDataSet, valuedTarget)
    tt2 = time.time()

    nObjects = dataSet.shape[0]
    nFeatures = dataSet.shape[1]

    curBalance = 0
    maxBalance = 0
    maxState = np.zeros(nFeatures)

    currentState = np.ones(nFeatures, dtype=np.int32) * (nObjects - 1)
    omitedObjects = np.full(nObjects, False, dtype=np.bool)

    sortedDataSet_device = cuda.to_device(sortedDataSet)
    valuedTargetBool_device = cuda.to_device(valuedTargetBool)
    minPositives_cupy = cp.array(minPositives)
    scores_result_device = cp.zeros(nFeatures, dtype=cp.float64)
    scores_pos_device = cuda.device_array(nFeatures, dtype=np.int32)
    scores_neg_device = cuda.device_array(nFeatures, dtype=np.int32)

    omitedDelta = np.full(nObjects, False,  dtype=np.bool)

    maxLeft = 1

    iSteps = 0

    kernelTime = 0
    deltaTime = 0
    minPositivesTime = 0
    minPositivesTimeCuPy = 0
    updatingTime = 0
    clearKernelTime = 0
    todevice = 0
    resArray = 0
    tohost = 0

    tt4 = time.time()

    while True:
        iSteps += 1

        t1 = time.time()

        #stoppingCriteria = checkStoppingCriteria(minPositives,  currentState)
        #stoppingCriteria = checkStoppingCriteriaCuda(minPositives_device, currentState)
        #cuda.synchronize()

        minPositivesTime += (time.time() - t1)

        t1 = time.time()
        stoppingCriteria = checkStoppingCriteriaCupy(minPositives_cupy, currentState)
        minPositivesTimeCuPy += time.time() - t1

        if stoppingCriteria:
            break

        t1 = time.time()
        iFeature = getNextStepCuda(dataSet_device, sortedDataSet_device, valuedTargetBool_device, classValues, currentState, omitedObjects, scores_pos_device, scores_neg_device, scores_result_device)
        t2 = time.time()
        if iFeature == -1:
            break

        kernelTime += (t2 - t1)

        t1 = time.time()
        currentState, omitedObjects, addedPositives, delta = calcDelta(iFeature, dataSet, sortedDataSet, valuedTarget, currentState, omitedObjects, omitedDelta, True)
        t2 = time.time()
        deltaTime += (t2 - t1)

        maxLeft -= addedPositives

        curBalance += delta
        #print('Feature: #{' + str(iFeature) + '}, d{' + f2s(delta, 20) + '}. Cur balance: {' + f2s(curBalance, 20) + '}, best balance: {' + f2s(maxBalance, 10) + '}')

        t1 = time.time()
        if curBalance > maxBalance:
            maxBalance = curBalance

            #for kFeature in range(nFeatures):
            #    objectIdx = sortedDataSet[currentState[kFeature], kFeature]
            #    maxState[kFeature] = dataSet[objectIdx, kFeature]

        updatingTime += (time.time() - t1)
        if maxLeft + curBalance <= maxBalance:
            break

    #print('Total time: ' + str(time.time() - tt1) + ' Preparation time: ' + str(tt2 - tt1) + ' Before while time: ' + str(tt4 - tt1) + ' Min positives time: ' + str(minPositivesTime) + ' Min positives cupy time: ' + str(minPositivesTimeCuPy) + ' Updating time: ' + str(updatingTime) + ' Clear kernel time: ' + str(clearKernelTime) + ' To device: ' + str(todevice) + ' To host: ' + str(tohost) + ' Array: ' + str(resArray) + ' Kernel time: ' + str(kernelTime) + ' Delta time: ' + str(deltaTime))
    return abs(maxBalance), maxState

def getMaximumDiviserFastCuda(dataSet, target):
    dataSet = prepareDataSet(dataSet)
    nClasses, counts = np.unique(target, return_counts=True)

    if len(nClasses) != 2:
        #raise ValueError('Number of classes should be equal to two, instead {:}'.format(len(nClasses)))
        print('Error!!! Number of classes should be equal to two, instead ', len(nClasses))

    dataSet_device = cuda.to_device(dataSet)
    valuedTarget1, boolValuedTarget1 = GetValuedAndBoolTarget(target, nClasses[0], 1 / counts[0], -1 / counts[1])
    c1Banalce, c1diviser = getMaximumDiviserPerClassFastCuda(dataSet, dataSet_device, valuedTarget1, boolValuedTarget1, [1 / counts[0], -1 / counts[1]])

    valuedTarget2, boolValuedTarget2 = GetValuedAndBoolTarget(target, nClasses[1], 1 / counts[1], -1 / counts[0])
    c2Banalce, c2diviser = getMaximumDiviserPerClassFastCuda(dataSet, dataSet_device, valuedTarget2, boolValuedTarget2, [1 / counts[1], -1 / counts[0]])

    if c1Banalce > c2Banalce:
        return c1Banalce, c1diviser

    return c2Banalce, c2diviser
