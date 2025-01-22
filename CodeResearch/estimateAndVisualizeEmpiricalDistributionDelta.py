import math

import numpy as np
from sklearn.preprocessing import LabelEncoder

from CodeResearch.Visualization.VisualizeAndSaveDistributionDeltas import VisualizeAndSaveDistributionDeltas
from CodeResearch.Visualization.visualizePValues import visualizePValues
from CodeResearch.calcModelAndRademacherComplexity import calculateModelAndDistributionDelta
from CodeResearch.pValueCalculator import calcPValueStochastic, calcPValueFast


def estimateOneOverOthers(dataSet, target, iClass, taskName, *args, **kwargs):
    enc = LabelEncoder()

    nClasses = np.unique(target)

    cIdx = np.where(target == iClass)[0]
    oIdx = np.where(target != iClass)[0]
    curTarget = np.copy(target)

    curTarget[cIdx] = iClass
    curTarget[oIdx] = -1

    curTarget = enc.fit_transform(np.ravel(curTarget))

    estimateAndVisualizeEmpiricalDistributionDeltaConcrete(dataSet, curTarget,
                                                           '{0}_c{1}_of_{2}'.format(taskName, iClass, len(nClasses)),
                                                           args, kwargs)
    pass


def estimateOneVsOne(dataSet, target, iClass, taskName, args, kwargs):
    enc = LabelEncoder()
    nClasses = np.unique(target)
    nFeatures = dataSet.shape[1]

    for jClass in np.arange(iClass):
        cIdx = np.where(target == iClass)[0]
        jIdx = np.where(target == jClass)[0]

        curTarget = np.zeros(len(cIdx) + len(jIdx))
        curSet = np.zeros((len(curTarget), nFeatures))

        curTarget[0:len(cIdx)] = target[cIdx]
        curSet[0:len(cIdx), :] = dataSet[cIdx, :]

        curTarget[len(cIdx):len(curTarget)] = target[jIdx]
        curSet[len(cIdx):len(curTarget), :] = dataSet[jIdx, :]

        curTarget = enc.fit_transform(np.ravel(curTarget))

        estimateAndVisualizeEmpiricalDistributionDeltaConcrete(curSet, curTarget,
                                                               '{0}_c{1}({4})_vs_c{2}({5})_of_{3}'.format(taskName, iClass, jClass, len(nClasses), len(cIdx), len(jIdx)),
                                                               args, kwargs)

    pass

def estimateOneVsSelf(dataSet, target, iClass, taskName, args, kwargs):
    enc = LabelEncoder()
    cIdx = np.where(target == iClass)[0]

    curTarget = target[cIdx]
    curSet = dataSet[cIdx, :]

    middle = math.floor(len(cIdx) / 2)
    curTarget[middle:(len(curTarget))] = -1

    curTarget = enc.fit_transform(np.ravel(curTarget))

    estimateAndVisualizeEmpiricalDistributionDeltaConcrete(curSet, curTarget,
                                                           '{0}_c{1}({2})_vs_self_of_{3}'.format(taskName, iClass,
                                                                                                      len(cIdx),
                                                                                                      len(curTarget)),
                                                           args, kwargs)
    pass

def estimateAndVisualizeEmpiricalDistributionDelta(dataSet, target, taskName, *args, **kwargs):
    enc = LabelEncoder()
    target = enc.fit_transform(np.ravel(target))

    nClasses = np.unique(target)

    if len(nClasses) > 2:
        for iClass in nClasses:
            #estimateOneOverOthers(dataSet, target, iClass, taskName, args, kwargs)
            estimateOneVsOne(dataSet, target, iClass, taskName, args, kwargs)
            #estimateOneVsSelf(dataSet, target, iClass, taskName, args, kwargs)

    else:
        estimateAndVisualizeEmpiricalDistributionDeltaConcrete(dataSet, target, taskName, args, kwargs)

    pass

def estimateAndVisualizeEmpiricalDistributionDeltaConcrete(dataSet, target, taskName, *args, **kwargs):
    nAttempts = 50
    nRadSets = 1
    modelAttempts = 5

    probability = 0.95
    delta = 1 - probability
    nObjects = dataSet.shape[0]
    nFeatures = dataSet.shape[1]
    nClasses = len(np.unique(target))

    totalPoints = kwargs.get('t', None)
    totalPoints = 25 if totalPoints is None else totalPoints

    step = max(1, math.floor(min(nObjects / 2, 3000) / totalPoints))

    print('Starting delta task: {0}. nAttempts: {1}, modelAttempts: {2}, objects: {3}'.format(taskName, nAttempts, modelAttempts, nObjects))

    maxObjects = totalPoints

    classesPairs = math.floor(nClasses * (nClasses - 1) / 2)
    minObjects = 1

    radEstimations = np.zeros((classesPairs, maxObjects - minObjects), dtype=float)
    radUpperEstimations = np.zeros((classesPairs, maxObjects - minObjects), dtype=float)

    nModels = 2

    accuracy = np.zeros((nModels, maxObjects - minObjects), dtype=float)
    modelSigma = np.zeros((nModels, maxObjects - minObjects), dtype=float)

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

        accuracy[:, iDistribution] = modelResult['accuracy']
        modelSigma[:, iDistribution] = modelResult['modelSigma']

    xLabels = np.arange(minObjects, maxObjects) * step

    data = {'xLabels': xLabels, 'rad': radEstimations,
            'upperRad': radUpperEstimations, 'accuracy': accuracy, 'modelSigma': modelSigma,
            'mcDiarmid': mcDiarmids}

    task = {'name': taskName, 'nObjects': nObjects, 'nFeatures': nFeatures, 'nAttempts': nAttempts,
            'modelAttempts': modelAttempts, 'step': step, 'prob': probability, 'totalPoints': totalPoints,
            'nRadSets': nRadSets, 'nClasses': nClasses}

    VisualizeAndSaveDistributionDeltas(data, task)

    return

def estimatePValuesForClassesSeparation(dataSet, target, taskName, *args, **kwargs):
    enc = LabelEncoder()
    target = enc.fit_transform(np.ravel(target))

    nObjects = len(target)

    nAttempts = 5
    #nClasses = len(np.unique(target))
    nClasses = 2

    pairs = math.floor(nClasses * (nClasses - 1) / 2)

    totalPoints = kwargs.get('t', None)
    totalPoints = 25 if totalPoints is None else totalPoints

    step = max(1, math.floor(min(nObjects / 2, 3000) / totalPoints))
    numberOfSteps = 40

    stochasticResults = np.zeros((numberOfSteps, pairs))
    fastResults = np.zeros((numberOfSteps, pairs))
    xSteps = (range(0, numberOfSteps) + np.ones(numberOfSteps)) * step
    data = {'steps': xSteps, 'taskName': taskName, 'step': 0}

    for iStep in range(0, numberOfSteps):
        currentObjects = (iStep + 1) * step
        print('Step#: {:}, objects: {:}'.format(iStep, currentObjects))
        curIdx = 0

        data['step'] = iStep
        for iClass in range(0, nClasses):
            for jClass in range(0, iClass):
                ijpValue = calcPValueStochastic(currentObjects, dataSet, target, iClass, jClass, nAttempts)
                stochasticResults[iStep, curIdx] = ijpValue

                ijpValue = calcPValueFast(currentObjects, dataSet, target, iClass, jClass, nAttempts)
                fastResults[iStep, curIdx] = ijpValue

                curIdx += 1
                data['stochastic'] = stochasticResults
                data['fast'] = fastResults

                visualizePValues(data)

    return