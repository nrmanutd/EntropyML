import time

import cupy as cp
import numba as nb
import numpy as np
from numba import jit, prange, cuda

from CodeResearch.DiviserCalculation.diviserHelpers import prepareDataSet, getSortedSet, \
    GetValuedAndBoolTarget


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
            curValue = scores_pos[thread_idx] / posValue - scores_neg[thread_idx] / negValue
            cuda.atomic.max(bestValue, 0, curValue)

    cuda.syncthreads()

    if thread_idx < scores_pos.size:
        if scores_pos[thread_idx] != -2:
            curValue = scores_pos[thread_idx] / posValue - scores_neg[thread_idx] / negValue
            #print(' curValue = ' + str(curValue))
            if curValue == bestValue[0]:
                cuda.atomic.min(bestIndex, 0, thread_idx)

@cuda.jit
def calculate_scores_kernel(dataSet, sortedDataSet, valuedTargetBool, currentState, omitedObjects, scores_pos, scores_neg, scores_result, posValue, negValue, nObjects, nFeatures, shared_memory_elements):
    shared_mem = cuda.shared.array(shape=0, dtype=np.uint32)
    result_score = cuda.shared.array(shape=2, dtype=np.int32)
    currentResult = cuda.shared.array(shape=1, dtype=np.int32)

    thread_idx = cuda.threadIdx.x
    stride = cuda.blockDim.x

    bits_per_word = 32

    for i in range(thread_idx, shared_memory_elements, stride):
        shared_mem[i] = omitedObjects[i]
        #shared_mem[i] = 0

    cuda.syncthreads()
    curBlock = cuda.blockIdx.x

    #for i in range(thread_idx, nObjects, stride):
    #    if omitedObjects[i]:
    #        word = i // bits_per_word
    #        bit = i % bits_per_word

    #       cuda.atomic.or_(shared_mem, word, 1 << bit)

    cuda.syncthreads()

    if thread_idx == 0:
        currentResult[0] = -1
        startPosition = currentState[curBlock]
        result_score[0] = 0
        result_score[1] = 0

        wasPositive = False
        positiveValue = 0

        for iSortedObject in range(startPosition, -1, -1):
            iObject = sortedDataSet[iSortedObject, curBlock]

            if wasPositive and not valuedTargetBool[iObject] and dataSet[iObject, curBlock] != positiveValue:
                currentResult[0] = iSortedObject
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

        if currentResult[0] == -1:
            scores_result[curBlock] = -np.inf
            scores_pos[curBlock] = -2
            scores_neg[curBlock] = 0

    cuda.syncthreads()

    if currentResult[0] == -1:
        return

    for iFeature in range(thread_idx, nFeatures, stride):
        for iSortedObject in range(currentState[iFeature], -1, -1):
            iObject = sortedDataSet[iSortedObject, iFeature]

            curObject = iObject // bits_per_word
            bit = iObject % bits_per_word
            wasOmitted = ((shared_mem[curObject] & (1 << bit)) != 0)

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
        scores_result[curBlock] = result_score[0] / posValue - result_score[1] / negValue
        scores_pos[curBlock] = result_score[0]
        scores_neg[curBlock] = result_score[1]

    return

@jit(nopython=True)
def convert_to_cuda(omitedObjects, bits_per_word):
    nObjects = len(omitedObjects)
    words = (nObjects + bits_per_word - 1) // bits_per_word

    res = np.zeros(words, dtype=np.uint32)
    for iObject in range(nObjects):
        num = iObject // bits_per_word
        bit = iObject % bits_per_word

        if omitedObjects[iObject]:
            res[num] |= 1 << bit

    return res

def getNextStepCuda(dataSet_device, sortedDataSet_device, valuedTargetBool_device, classValues, currentState, omitedObjects, scores_pos, scores_neg, scores_result):
    nObjects = len(omitedObjects)
    nFeatures = len(currentState)

    threads_per_block = 64
    blocks_per_grid = nFeatures

    bits_per_word = 32
    words = (nObjects + bits_per_word - 1) // bits_per_word
    shared_memory_size = words * 4

    t1 = time.time()
    currentState_device = cuda.to_device(currentState)
    curStateTime = time.time() - t1

    t1 = time.time()
    #omitedObjects_device = cuda.to_device(omitedObjects)
    omitedObjects_cuda = convert_to_cuda(omitedObjects, 32)
    omitedObjects_device = cuda.to_device(omitedObjects_cuda)
    omitedTime = time.time() - t1

    t1 = time.time()
    calculate_scores_kernel[blocks_per_grid, threads_per_block, 0, shared_memory_size](dataSet_device, sortedDataSet_device, valuedTargetBool_device, currentState_device, omitedObjects_device, scores_pos, scores_neg, scores_result, classValues[0], classValues[1], nObjects, nFeatures, words)
    cuda.synchronize()
    clearKernelTime = time.time() - t1

    bestValue = cuda.device_array(1, dtype=np.float64)
    bestIndex = cuda.device_array(1, dtype=np.int64)

    threads_per_block = 64
    blocks_per_grid = (nFeatures + threads_per_block - 1) // threads_per_block

    calculate_best_index[blocks_per_grid, threads_per_block](scores_pos, scores_neg, classValues[0], classValues[1], bestIndex, bestValue)
    cuda.synchronize()

    bestIndexHost = bestIndex.copy_to_host()[0]

    res =bestIndexHost if bestIndexHost != nFeatures else -1

    return res, clearKernelTime, omitedTime, curStateTime

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

    return cp.any(comparison_result)

def getMaximumDiviserPerClassFastCuda(dataSet, dataSet_device, sortedSet, sortedSet_device, valuedTarget, boolValuedTarget, classValues):
    tt1 = time.time()
    minPositives = getMinPositives(sortedSet, valuedTarget)

    nObjects = dataSet.shape[0]
    nFeatures = dataSet.shape[1]

    curBalance = 0
    maxBalance = 0
    maxState = np.zeros(nFeatures)

    currentState = np.ones(nFeatures, dtype=np.int32) * (nObjects - 1)
    omitedObjects = np.full(nObjects, False, dtype=np.bool)

    valuedTargetBool_device = cuda.to_device(boolValuedTarget)
    minPositives_cupy = cp.array(minPositives)
    scores_result_device = cuda.device_array(nFeatures, dtype=np.float32)
    scores_pos_device = cuda.device_array(nFeatures, dtype=np.int32)
    scores_neg_device = cuda.device_array(nFeatures, dtype=np.int32)

    omitedDelta = np.full(nObjects, False,  dtype=np.bool)

    maxLeft = 1

    iSteps = 0

    kernelTime = 0
    clearKernelTime = 0
    bestIndexCalculation = 0
    curStateTime = 0
    deltaTime = 0
    minPositivesTimeCuPy = 0
    updatingTime = 0

    while True:
        iSteps += 1

        t1 = time.time()
        stoppingCriteria = checkStoppingCriteriaCupy(minPositives_cupy, currentState)
        minPositivesTimeCuPy += time.time() - t1

        if stoppingCriteria:
            break

        t1 = time.time()
        iFeature, cct, ot, cst = getNextStepCuda(dataSet_device, sortedSet_device, valuedTargetBool_device, classValues, currentState, omitedObjects, scores_pos_device, scores_neg_device, scores_result_device)
        t2 = time.time()
        if iFeature == -1:
            break

        bestIndexCalculation += ot
        clearKernelTime += cct
        curStateTime += cst
        kernelTime += (t2 - t1)

        t1 = time.time()
        currentState, omitedObjects, addedPositives, delta = calcDelta(iFeature, dataSet, sortedSet, valuedTarget, currentState, omitedObjects, omitedDelta, True)
        t2 = time.time()
        deltaTime += (t2 - t1)

        maxLeft -= addedPositives
        curBalance += delta

        t1 = time.time()
        if curBalance > maxBalance:
            maxBalance = curBalance

            #for kFeature in range(nFeatures):
            #    objectIdx = sortedSet[currentState[kFeature], kFeature]
            #    maxState[kFeature] = dataSet[objectIdx, kFeature]

        updatingTime += (time.time() - t1)

        if (maxLeft + curBalance - maxBalance) <= 0.001:
            break

    #print('Total time: ' + str(time.time() - tt1) + ' Min positives cupy time: ' + str(minPositivesTimeCuPy) + ' Updating time: ' + str(updatingTime) + ' Kernel time: ' + str(kernelTime) + ' Clear kernel time: ' + str(clearKernelTime) + ' Best index calculation: ' + str(bestIndexCalculation) + ' Cur state time: ' + str(curStateTime) + ' Delta time: ' + str(deltaTime))
    return abs(maxBalance), maxState

def getMaximumDiviserFastCudaCore(dataSet, dataSet_device, target, sortedSet1, sortedSet1_device, valuedTarget1, boolValuedTarget1, sortedSet2, sortedSet2_device, valuedTarget2, boolValuedTarget2):
    nClasses, counts = np.unique(target, return_counts=True)

    if len(nClasses) != 2:
        # raise ValueError('Number of classes should be equal to two, instead {:}'.format(len(nClasses)))
        print('Error!!! Number of classes should be equal to two, instead ', len(nClasses))

    c1Banalce, c1diviser = getMaximumDiviserPerClassFastCuda(dataSet, dataSet_device, sortedSet1, sortedSet1_device, valuedTarget1, boolValuedTarget1, [counts[0], counts[1]])
    c2Banalce, c2diviser = getMaximumDiviserPerClassFastCuda(dataSet, dataSet_device, sortedSet2, sortedSet2_device, valuedTarget2, boolValuedTarget2, [counts[1], counts[0]])

    if c1Banalce > c2Banalce:
        return c1Banalce, c1diviser

    return c2Banalce, c2diviser

def getMaximumDiviserFastCudaPreloadedToDevice(dataSet, dataSet_device, target, sortedSet1, sortedSet1_device, sortedSet2, sortedSet2_device):
    nClasses, counts = np.unique(target, return_counts=True)

    if len(nClasses) != 2:
        # raise ValueError('Number of classes should be equal to two, instead {:}'.format(len(nClasses)))
        print('Error!!! Number of classes should be equal to two, instead ', len(nClasses))

    valuedTarget1, boolValuedTarget1 = GetValuedAndBoolTarget(target, nClasses[0], 1 / counts[0], -1 / counts[1])
    valuedTarget2, boolValuedTarget2 = GetValuedAndBoolTarget(target, nClasses[1], 1 / counts[1], -1 / counts[0])

    return getMaximumDiviserFastCudaCore(dataSet, dataSet_device, target, sortedSet1, sortedSet1_device, valuedTarget1, boolValuedTarget1, sortedSet2, sortedSet2_device, valuedTarget2, boolValuedTarget2)

def getMaximumDiviserFastCuda(dataSet, target):
    dataSet = prepareDataSet(dataSet)
    nClasses, counts = np.unique(target, return_counts=True)

    if len(nClasses) != 2:
        #raise ValueError('Number of classes should be equal to two, instead {:}'.format(len(nClasses)))
        print('Error!!! Number of classes should be equal to two, instead ', len(nClasses))

    dataSet_device = cuda.to_device(dataSet)

    valuedTarget1, boolValuedTarget1 = GetValuedAndBoolTarget(target, nClasses[0], 1 / counts[0], -1 / counts[1])
    sortedSet1 = getSortedSet(dataSet, valuedTarget1)
    sortedSet1_device = cuda.to_device(sortedSet1)

    valuedTarget2, boolValuedTarget2 = GetValuedAndBoolTarget(target, nClasses[1], 1 / counts[1], -1 / counts[0])
    sortedSet2 = getSortedSet(dataSet, valuedTarget2)
    sortedSet2_device = cuda.to_device(sortedSet2)

    return getMaximumDiviserFastCudaCore(dataSet, dataSet_device, target, sortedSet1, sortedSet1_device, valuedTarget1, boolValuedTarget1, sortedSet2, sortedSet2_device, valuedTarget2, boolValuedTarget2)
