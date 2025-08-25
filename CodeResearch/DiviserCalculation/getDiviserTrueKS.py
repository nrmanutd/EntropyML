from itertools import product

import numpy as np

from CodeResearch.DiviserCalculation.diviserHelpers import getIdx, getPointsUnderDiviser, prepareDataSet, \
    get_one_bit_indices
from CodeResearch.Helpers.rademacherHelpers import GetSortedData

def calcTrueKSSingleOrthant(dataSet, target):
    sortedSet = np.sort(dataSet, axis=0)

    # Шаг 1: Находим уникальные значения для каждого признака (столбца)
    unique_values_per_feature = [np.unique(sortedSet[:, i]) for i in range(sortedSet.shape[1])]

    # Шаг 2: Генерируем все возможные комбинации
    all_combinations = list(product(*unique_values_per_feature))

    # Шаг 3: Преобразуем в массив NumPy (опционально)
    all_combinations_array = np.array(all_combinations)

    nClasses = np.unique(target)

    nFeatures = dataSet.shape[1]

    negativeObjectsIdx = np.where(target == nClasses[0])[0]
    negativeCount = len(negativeObjectsIdx)
    negativeObjects = dataSet[negativeObjectsIdx, :]
    sortedNegIdx, sortedNegValues = GetSortedData(negativeObjects)

    positiveObjectsIdx = np.where(target == nClasses[1])[0]
    positiveCount = len(positiveObjectsIdx)
    positiveObjects = dataSet[positiveObjectsIdx, :]
    sortedPosIdx, sortedPosValues = GetSortedData(positiveObjects)

    positiveIdx = getIdx(dataSet[positiveObjectsIdx, :], range(0, len(positiveObjectsIdx)))
    negativeIdx = getIdx(dataSet[negativeObjectsIdx, :], range(0, len(negativeObjectsIdx)))

    basePoint = np.zeros(nFeatures)
    for i in range(0, nFeatures):
        basePoint[i] = min(sortedPosValues[0, i], sortedNegValues[0, i])

    result = 0

    for i in range(len(all_combinations_array)):
        if i %10000 == 0:
            print(f'Combination {i} of {len(all_combinations_array)}...')

        currentPoint = all_combinations_array[i]

        positivePoints = getPointsUnderDiviser(positiveIdx, currentPoint, basePoint)
        negativePoints = getPointsUnderDiviser(negativeIdx, currentPoint, basePoint)

        curResult = abs(positivePoints / positiveCount - negativePoints / negativeCount)

        result = max(result, curResult)

    return result


def calcConcreteOrthant(dataSet, target, iOrthant):
    indices = get_one_bit_indices(iOrthant)
    for i in range(len(indices)):
        dataSet[:, i] *= -1

    result = calcTrueKSSingleOrthant(dataSet, target)

    for i in range(len(indices)):
        dataSet[:, i] *= -1

    return result


def getMaximumDiviserPerClassTrueKS(dataSet, target):
    nFeatures = dataSet.shape[1]

    bestOrthantValue = 0

    for i in range(2**nFeatures):
        #print(f'Orthant # {i} of {2**nFeatures}')
        orth = calcConcreteOrthant(dataSet, target, i)
        bestOrthantValue = max(bestOrthantValue, orth)

    return bestOrthantValue

def getMaximumDiviserTrueKS(dataSet, target):
    dataSet = prepareDataSet(dataSet)
    nClasses, counts = np.unique(target, return_counts=True)

    if len(nClasses) != 2:
        raise ValueError('Number of classes should be equal to two, instead {:}'.format(len(nClasses)))

    balance = getMaximumDiviserPerClassTrueKS(dataSet, target)
    return balance