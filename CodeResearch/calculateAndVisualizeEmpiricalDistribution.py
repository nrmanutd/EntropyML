import math

import numpy as np

from CodeResearch.Visualization.SaveRademacherResults import SaveRademacherResults
from CodeResearch.calcModelAndRademacherComplexity import calculateRademacherComplexity


def calculateAndVisualizeEmpiricalDistribution(dataSet, target, taskName, *args, **kwargs):

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

    print('Starting task: {0}. nAttempts: {1}, modelAttempts: {2}, objects: {3}'.format(taskName, nAttempts, modelAttempts, nObjects))

    maxObjects = totalPoints
    minObjects = 1

    avgDistributions = np.zeros(maxObjects - minObjects, dtype=float)
    avgDistributions2 = np.zeros(maxObjects - minObjects, dtype=float)
    avgDistributionsA = np.zeros(maxObjects - minObjects, dtype=float)
    avgDistributions2A = np.zeros(maxObjects - minObjects, dtype=float)
    avgDistributions3 = np.zeros(maxObjects - minObjects, dtype=float)
    accuracy = np.zeros(maxObjects - minObjects, dtype=float)
    modelSigma = np.zeros(maxObjects - minObjects, dtype=float)

    sigmas = np.zeros(maxObjects - minObjects, dtype=float)
    sigmasA = np.zeros(maxObjects - minObjects, dtype=float)
    mcDiarmids = np.zeros(maxObjects - minObjects, dtype=float)

    objectsIterator = np.arange(minObjects, maxObjects, dtype=int)

    for iDistribution in np.arange(len(objectsIterator), dtype=int):

        currentObjects = objectsIterator[iDistribution] * step
        print('Calculating for {0}'.format(currentObjects))

        mcDiarmids[iDistribution] = math.sqrt(math.log(1/delta) / currentObjects)
        result = calculateRademacherComplexity(dataSet, currentObjects, nAttempts, modelAttempts, nRadSets, target)

        radResult = result['radResult']
        modelResult = result['modelResult']

        avgDistributions[iDistribution] = radResult['rad']
        sigmas[iDistribution] = radResult['sigma']
        avgDistributions2[iDistribution] = radResult['upperRad']
        avgDistributions3[iDistribution] = radResult['alpha']

        avgDistributionsA[iDistribution] = radResult['radA']
        sigmasA[iDistribution] = radResult['sigmaA']
        avgDistributions2A[iDistribution] = radResult['upperRadA']

        accuracy[iDistribution] = modelResult['accuracy']
        modelSigma[iDistribution] = modelResult['modelSigma']

    xLabels = np.arange(minObjects, maxObjects) * step

    data = {'xLabels': xLabels, 'rad': avgDistributions,
            'upperRad': avgDistributions2, 'accuracy': accuracy, 'modelSigma': modelSigma,
            'mcDiarmid': mcDiarmids, 'radA': avgDistributionsA, 'upperRadA': avgDistributions2A, 'alpha': avgDistributions3}

    task = {'name': taskName, 'nObjects': nObjects, 'nFeatures': nFeatures, 'nAttempts': nAttempts,
            'modelAttempts': modelAttempts, 'step': step, 'prob': probability, 'totalPoints': totalPoints,
            'nRadSets': nRadSets, 'nClasses': nClasses}

    SaveRademacherResults(data, task)

    return