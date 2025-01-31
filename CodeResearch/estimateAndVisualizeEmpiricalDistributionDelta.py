import math
import time

import numpy as np
from sklearn.preprocessing import LabelEncoder

from CodeResearch.Visualization.VisualizeAndSaveDistributionDeltas import VisualizeAndSaveDistributionDeltas
from CodeResearch.Visualization.saveDataForLatex import saveDataForTable
from CodeResearch.Visualization.saveDataForVisualization import saveDataForVisualization
from CodeResearch.Visualization.visualizePValues import visualizePValues
from CodeResearch.calcModelAndRademacherComplexity import calculateModelAndDistributionDelta
from CodeResearch.pValueCalculator import calcPValueStochastic, calcPValueFast, calcPValueFastParallel
from CodeResearch.slopeCalculator import calculateSlope, getBestSlopeMedian, getBestSlopeMax, calculateSlopeGradient


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
    nFeatures = dataSet.shape[1]
    alpha = 0.001
    beta = 0.01

    nModelAttempts = 1
    nAttempts = 10
    nClasses = len(np.unique(target))

    pairs = math.floor(nClasses * (nClasses - 1) / 2)

    numberOfSteps = kwargs.get('t', None)
    numberOfSteps = 10 if numberOfSteps is None else numberOfSteps

    targetResults = np.zeros((numberOfSteps, pairs))
    fastResults = np.zeros((numberOfSteps, pairs))
    fastResultsUp = np.zeros((numberOfSteps, pairs))
    pValuesResults = np.zeros((numberOfSteps, pairs, nAttempts))
    modelPredictions = np.zeros((numberOfSteps, pairs, 2))
    meanPValues = np.zeros((numberOfSteps, pairs))
    medianPValues = np.zeros((numberOfSteps, pairs))
    nPoints = np.zeros((pairs, 4), dtype=int)
    xSteps = np.zeros((numberOfSteps, pairs), dtype=int)
    eachTaskResult = []

    epsilon = math.sqrt(math.log(2 * nFeatures/alpha))
    data = {'taskName': taskName, 'step': 0, 'nAttempts': nAttempts, 'epsilon': epsilon, 'alpha': alpha, 'beta': beta}
    names = []
    curIdx = 0

    #setToCompare = set([5, 8])
    setToCompare = set(np.unique(target))

    for iClass in range(3, nClasses):
        for jClass in range(0, iClass):

            if iClass not in setToCompare or jClass not in setToCompare:
                continue

            iObjectsCount = len(np.where(target == iClass)[0])
            jObjectsCount = len(np.where(target == jClass)[0])

            totalObjects = (iObjectsCount + jObjectsCount)
            #step = math.floor(min(totalObjects / 2, 3000) / numberOfSteps)
            step = 100

            xSteps[:, curIdx] = (range(numberOfSteps) + np.ones(numberOfSteps, dtype=int)) * step
            data['steps'] = xSteps

            print('Current pair of classes: {:}/{:}, task {:}, objects {:}, maxObjects {:}, step {:}'.format(iClass, jClass, taskName, nObjects, step * numberOfSteps, step))
            names.append('{:} vs {:}'.format(iClass, jClass))
            meanSlopesInd = []
            medianSlopesInd = []
            lowSlopesInd = []

            for iStep in range(0, numberOfSteps):
                currentObjects = max(2, (iStep + 1) * step)

                if iObjectsCount + jObjectsCount < currentObjects:
                    print('Objects of class {:}: {:}, of class {:}: {:}'.format(iClass, iObjectsCount, jClass, jObjectsCount))
                    break

                c1 = time.time()
                print('Step#: {:}, objects: {:}'.format(iStep, currentObjects))
                data['step'] = iStep

                #ijpValue = calcPValueStochastic(currentObjects, dataSet, target, iClass, jClass, nAttempts)
                #stochasticResults[iStep, curIdx] = ijpValue

                #ijpValue, tValue, pValues, modelPrediction = calcPValueStochastic(currentObjects, dataSet, target, iClass, jClass, nAttempts)
                ijpValue, ijpValueUp, tValue, pValues, modelPrediction = calcPValueFastParallel(currentObjects, dataSet, target, iClass, jClass, nAttempts, nModelAttempts, beta)
                #ijpValue, ijpValueUp, tValue, pValues, modelPrediction = calcPValueFast(currentObjects, dataSet, target, iClass, jClass, nAttempts, nModelAttempts, beta)
                fastResults[iStep, curIdx] = ijpValue
                fastResultsUp[iStep, curIdx] = ijpValueUp
                targetResults[iStep, curIdx] = tValue
                pValuesResults[iStep, curIdx, :] = pValues
                modelPredictions[iStep, curIdx, :] = modelPrediction
                meanPValues[iStep, curIdx] = np.mean(pValues)
                medianPValues[iStep, curIdx] = np.median(pValues)

                if iStep > 1:
                    currentRange = range(iStep + 1)
                    meanSlope = calculateSlopeGradient(xSteps[currentRange, curIdx], meanPValues[currentRange, curIdx])
                    medianSlope = calculateSlopeGradient(xSteps[currentRange, curIdx], medianPValues[currentRange, curIdx])
                    lowSlope = calculateSlopeGradient(xSteps[currentRange, curIdx], fastResults[currentRange, curIdx])

                    meanSlopesInd.append(meanSlope)
                    medianSlopesInd.append(medianSlope)
                    lowSlopesInd.append(lowSlope)

                    nPoints[curIdx, 0] = getBestSlopeMedian(meanSlopesInd)
                    nPoints[curIdx, 1] = getBestSlopeMedian(medianSlopesInd)
                    nPoints[curIdx, 2] = getBestSlopeMedian(lowSlopesInd)

                    mfr = min(fastResults[currentRange, curIdx])
                    nPoints[curIdx, 3] = fastResults[currentRange, curIdx].tolist().index(mfr)

                data['pValuesResults'] = pValuesResults
                data['meanPValue'] = meanPValues
                data['medianPValue'] = medianPValues
                data['targetResults'] = targetResults
                data['pairIndex'] = curIdx
                data['classes'] = '{:} vs {:}'.format(iClass, jClass)
                data['fast'] = fastResults
                data['fastUp'] = fastResultsUp
                data['names'] = names
                data['model'] = modelPredictions
                data['nPoints'] = nPoints

                e1 = time.time()
                print('Time elapsed for step #{:}: {:.2f}'.format(iStep, e1 - c1))
                visualizePValues(data)
                saveDataForVisualization(data)

            curResult = {'Classes': names[curIdx], 'total': totalObjects, 'iClass':  iObjectsCount, 'jClass': jObjectsCount, 'nPoints':  (1 + nPoints[curIdx, :]) * step, 'KSl': meanPValues[-1, curIdx], 'KSu': fastResultsUp[-1, curIdx]}
            eachTaskResult.append(curResult)
            saveDataForTable(eachTaskResult, taskName, names[curIdx])
            visualizePValues(data)
            saveDataForVisualization(data)

            curIdx += 1

    return
