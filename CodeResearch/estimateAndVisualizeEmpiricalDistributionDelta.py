import math

import numpy as np

from CodeResearch.VisualizeAndSaveDistributionDeltas import VisualizeAndSaveDistributionDeltas
from CodeResearch.calcModelAndRademacherComplexity import calculateModelAndDistributionDelta


def estimateAndVisualizeEmpiricalDistributionDelta(dataSet, target, taskName, *args, **kwargs):

    nAttempts = 40
    nRadSets = 20
    modelAttempts = 20

    probability = 0.95
    delta = 1 - probability
    nObjects = dataSet.shape[0]
    nFeatures = dataSet.shape[1]
    nClasses = len(np.unique(target))

    totalPoints = kwargs.get('t', None)
    totalPoints = 25 if totalPoints is None else totalPoints

    step = math.floor(nObjects / 2 / totalPoints)

    print('Starting delta task: {0}. nAttempts: {1}, modelAttempts: {2}, objects: {3}'.format(taskName, nAttempts, modelAttempts, nObjects))

    maxObjects = totalPoints

    classesPairs = math.floor(nClasses * (nClasses - 1) / 2)
    minObjects = 1

    radEstimations = np.zeros((classesPairs, maxObjects - minObjects), dtype=float)
    radUpperEstimations = np.zeros((classesPairs, maxObjects - minObjects), dtype=float)

    accuracy = np.zeros(maxObjects - minObjects, dtype=float)
    modelSigma = np.zeros(maxObjects - minObjects, dtype=float)

    mcDiarmids = np.zeros(maxObjects - minObjects, dtype=float)

    objectsIterator = np.arange(minObjects, maxObjects, dtype=int)

    for iDistribution in np.arange(len(objectsIterator), dtype=int):

        currentObjects = objectsIterator[iDistribution] * step
        print('Calculating for {0}'.format(currentObjects))

        mcDiarmids[iDistribution] = math.sqrt(math.log(1/delta) / currentObjects)
        result = calculateModelAndDistributionDelta(dataSet, currentObjects, nAttempts, modelAttempts, nRadSets, target)

        radResult = result['radResult']
        modelResult = result['modelResult']

        radEstimations[:, iDistribution] = radResult['rad']
        radUpperEstimations[:, iDistribution] = radResult['upperRad']

        accuracy[iDistribution] = modelResult['accuracy']
        modelSigma[iDistribution] = modelResult['modelSigma']

    xLabels = np.arange(minObjects, maxObjects) * step

    data = {'xLabels': xLabels, 'rad': radEstimations,
            'upperRad': radUpperEstimations, 'accuracy': accuracy, 'modelSigma': modelSigma,
            'mcDiarmid': mcDiarmids}

    task = {'name': taskName, 'nObjects': nObjects, 'nFeatures': nFeatures, 'nAttempts': nAttempts,
            'modelAttempts': modelAttempts, 'step': step, 'prob': probability, 'totalPoints': totalPoints,
            'nRadSets': nRadSets, 'nClasses': nClasses}

    VisualizeAndSaveDistributionDeltas(data, task)

    return