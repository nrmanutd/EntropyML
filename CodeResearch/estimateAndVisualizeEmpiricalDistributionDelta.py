import math
import time

import numpy as np
from sklearn.preprocessing import LabelEncoder

from CodeResearch.Visualization.VisualizeAndSaveDistributionDeltas import VisualizeAndSaveDistributionDeltas
from CodeResearch.Visualization.saveDataForVisualization import saveDataForVisualization
from CodeResearch.Visualization.visualizePValues import visualizePValues
from CodeResearch.calcModelAndRademacherComplexity import calculateModelAndDistributionDelta
from CodeResearch.pValueCalculator import calcPValueStochastic, calcPValueFast, calcPValueFastParallel


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
    estimatePValuesForClassesSeparation(dataSet, target, taskName, *args, **kwargs)
    return

    enc = LabelEncoder()
    target = enc.fit_transform(np.ravel(target))

    nClasses = np.unique(target)

    if len(nClasses) > 2:
        for iClass in nClasses:
            #estimateOneOverOthers(dataSet, target, iClass, taskName, args, kwargs)
            #estimateOneVsOne(dataSet, target, iClass, taskName, args, kwargs)
            estimateOneVsSelf(dataSet, target, iClass, taskName, args, kwargs)

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

    nAttempts = 1000
    nClasses = len(np.unique(target))

    pairs = math.floor(nClasses * (nClasses - 1) / 2)

    numberOfSteps = kwargs.get('t', None)
    numberOfSteps = 5 if numberOfSteps is None else numberOfSteps

    step = min(50, math.floor(min(nObjects, 3000) / numberOfSteps))
    #step = 200

    targetResults = np.zeros((numberOfSteps, pairs))
    fastResults = np.zeros((numberOfSteps, pairs))
    pValuesResults = np.zeros((numberOfSteps, pairs, nAttempts))
    modelPredictions = np.zeros((numberOfSteps, pairs, 2))
    meanPValues = np.zeros((numberOfSteps, pairs))
    nPoints = np.zeros(pairs)
    nThreshold = np.zeros(pairs)
    threshold = 0.05

    xSteps = (range(0, numberOfSteps) + np.ones(numberOfSteps)) * step
    data = {'steps': xSteps, 'taskName': taskName, 'step': 0, 'nAttempts': nAttempts}
    names = []
    curIdx = 0

    #setToCompare = set([6, 2])
    setToCompare = set(np.unique(target))

    for iClass in range(0, nClasses):
        for jClass in range(0, iClass):

            if iClass not in setToCompare or jClass not in setToCompare:
                continue

            print('Current pair of classes: {:}/{:}, task {:}'.format(iClass, jClass, taskName))
            names.append('{:}/{:}'.format(iClass, jClass))

            for iStep in range(0, numberOfSteps):
                currentObjects = (iStep + 1) * step
                c1 = time.time()
                print('Step#: {:}, objects: {:}'.format(iStep, currentObjects))
                data['step'] = iStep

                #ijpValue = calcPValueStochastic(currentObjects, dataSet, target, iClass, jClass, nAttempts)
                #stochasticResults[iStep, curIdx] = ijpValue

                #ijpValue, tValue, pValues, modelPrediction = calcPValueStochastic(currentObjects, dataSet, target, iClass, jClass, nAttempts)
                ijpValue, tValue, pValues, modelPrediction = calcPValueFastParallel(currentObjects, dataSet, target, iClass, jClass, nAttempts)
                #ijpValue, tValue, pValues, modelPrediction = calcPValueFast(currentObjects, dataSet, target, iClass, jClass, nAttempts)
                fastResults[iStep, curIdx] = ijpValue
                targetResults[iStep, curIdx] = tValue
                pValuesResults[iStep, curIdx, :] = pValues
                modelPredictions[iStep, curIdx, :] = modelPrediction
                meanPValues[iStep, curIdx] = np.mean(pValues)

                if iStep > 0:
                    curRatio = abs(meanPValues[iStep, curIdx] / meanPValues[iStep - 1, curIdx] - 1)
                    if curRatio < threshold and nPoints[curIdx] != 0:
                        nPoints[curIdx] = currentObjects
                        nThreshold[curIdx] = curRatio

                data['pValuesResults'] = pValuesResults
                data['meanPValue'] = meanPValues
                data['targetResults'] = targetResults
                data['pairIndex'] = curIdx
                data['classes'] = '{:} vs {:}'.format(iClass, jClass)
                data['fast'] = fastResults
                data['names'] = names
                data['model'] = modelPredictions
                data['nPoints'] = nPoints
                data['thresholds'] = nThreshold
                e1 = time.time()
                print('Time elapsed for step #{:}: {:.2f}'.format(iStep, e1 - c1))
                visualizePValues(data)
                saveDataForVisualization(data)

            curIdx += 1

    return
