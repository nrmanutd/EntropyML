import numpy as np
import math

def getDistribution(dataSet, objectBins):
    nObjects = dataSet.shape[0]
    nFeatures = dataSet.shape[1]

    objectBinsIndexes = np.zeros((nObjects, nFeatures))

    for iFeature in np.arange(nFeatures):
        objectBinsIndexes[:, iFeature] = np.digitize(dataSet[:, iFeature], objectBins[iFeature])

    bucketIdx = [''] * nObjects

    for iObject in np.arange(nObjects):
        bucketIdx[iObject] = ''.join(str(x) for x in objectBinsIndexes[iObject, :])

    eBins, cBins = np.unique(bucketIdx, return_counts=True)

    result = {}

    if(sum(cBins) != nObjects):
        raise Exception('Incorrect number of elements')

    for iElement in np.arange(len(eBins)):
        result[eBins[iElement]] = cBins[iElement]/nObjects

    return result


def calcComparisonConcrete(dataSet, objectBins, currentTrueClassIndex, comparingClassIndex, comparisonType):

    trueDistribution = getDistribution(dataSet[currentTrueClassIndex, :], objectBins)
    oppositeDistribution = getDistribution(dataSet[comparingClassIndex, :], objectBins)

    binsKeys = list(trueDistribution.keys())

    result = 0.0

    for iBin in np.arange(len(binsKeys)):
        bucket = binsKeys[iBin]
        if bucket in oppositeDistribution:
            if comparisonType == 'bc':
                result += math.sqrt(trueDistribution[bucket] * oppositeDistribution[bucket])
            elif comparisonType == 'ce':
                result -= trueDistribution[bucket] * math.log(oppositeDistribution[bucket])
            else:
                raise Exception('Incorrect parameter: ' + comparisonType)
    return result


def calcComparison(dataSet, objectBins, trueClassIndex, regressionClassesIndex, comparisonType):
    nObjects = dataSet.shape[0]

    trueClasses = trueClassIndex['classes']
    comparingClasses = regressionClassesIndex['classes']

    trueClassesLables = list(trueClasses.keys())
    resultComparison = 0.0

    for iClass in np.arange(len(trueClassesLables)):
        currentClass = trueClassesLables[iClass]
        currentTrueClassIndex = trueClassIndex['classesIndexes'][trueClasses[currentClass]]
        if currentClass in comparingClasses:
            comparingClassIndex = regressionClassesIndex['classesIndexes'][comparingClasses[currentClass]]

            cc = calcComparisonConcrete(dataSet, objectBins, currentTrueClassIndex, comparingClassIndex, comparisonType)
            resultComparison += trueClasses[currentClass] / nObjects * cc
        else:
            pass

    return resultComparison