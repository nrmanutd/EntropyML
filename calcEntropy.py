import math

import numpy as np
from scipy.stats import entropy
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

from SaveComparisonResults import SaveComparisonResults
from SaveEntropiesToFigures import SaveEntropiesToFigure
from TrainMLs import TrainLogit, TrainXGBoost
from commonHelpers import getClassesIndex, getPermutation
from comparisonCalculation import calcComparison
from entropyHelpers import calcMultiVarianceEntropy, calcConditionalEntropy, calcConditionalMultiVarianceEntropy
from getOptimalBinsCount import getOptimalBinsCount
from saveFeaturesToClassesDistribution import saveFeaturesToClassesDistribution

def calculateAndVisualizeSeveralEntropies(dataSet, target, taskName):
    enc = LabelEncoder()
    target = enc.fit_transform(np.ravel(target))

    permutations = 50
    print('Starting task: {0}'.format(taskName))

    nFeatures = dataSet.shape[1]
    nObjects = dataSet.shape[0]

    entries, counts = np.unique(target, return_counts=True)

    cl = getClassesIndex(target)

    upperBins = getOptimalBinsCount(dataSet)
    bins = np.arange(2, max(upperBins) + 1)

    simple = np.zeros(len(bins), dtype=float)
    multiEntropy = np.zeros(len(bins), dtype=float)

    conditional = np.zeros((len(bins), permutations + 1), dtype=float)
    multiConditionalEntropy = np.zeros((len(bins), permutations + 1), dtype=float)
    efficiency = np.zeros((len(bins), permutations + 1), dtype=float)

    logitResults = np.zeros(len(bins), dtype=float)
    xgboostResults = np.zeros(len(bins), dtype=float)

    #по числу разбиений, по числу перестановок, 2 - Bhattacharyya, Cross-Entropy
    comparisonResults = np.zeros((len(bins), permutations, 2), dtype=float)
    logitComparisonResults = np.zeros((len(bins), 2))
    xgBoostComparisonResults = np.zeros((len(bins), 2))

    ct = 0.0
    ccl = cl['classesIndexes']
    for iClass in np.arange(len(ccl)):
        ct += math.log(len(ccl[iClass])) * len(ccl[iClass]) / nObjects

    print('Training logit...')
    regressionTarget = TrainLogit(dataSet, target)
    regressionAccuracy = accuracy_score(target, regressionTarget)
    regressionClassesIndex = getClassesIndex(regressionTarget)

    print('Training xgboost...')
    xgBoostTarget = TrainXGBoost(dataSet, target)
    xgBoostAccuracy = accuracy_score(target, xgBoostTarget)
    xbBoostClassesIndex = getClassesIndex(xgBoostTarget)

    for iBins in np.arange(len(bins)):
        print('{2}: {0} of {1}'.format(iBins, len(bins), taskName))

        objectBins = []
        simpleEntropy = np.zeros(nFeatures + 1, dtype=float)

        for i in np.arange(nFeatures):
            iFeatureBin = min(bins[iBins], upperBins[i])
            h = np.histogram(dataSet[:, i], bins=iFeatureBin)
            objectBins.append(h[1])
            simpleEntropy[i] = entropy(h[0]/sum(h[0]))

        simpleEntropy[nFeatures] = entropy(counts / nObjects)

        simple[iBins] = sum(simpleEntropy)
        multiEntropy[iBins] = calcMultiVarianceEntropy(dataSet, objectBins)

        conditional[iBins, 0] = calcConditionalEntropy(dataSet, objectBins, cl['classesIndexes'])
        multiConditionalEntropy[iBins, 0] = calcConditionalMultiVarianceEntropy(dataSet, objectBins, cl['classesIndexes'])
        efficiency[iBins, 0] = conditional[iBins, 0] - multiConditionalEntropy[iBins, 0]

        for iPermutation in np.arange(1, permutations + 1):
            clPermutation = getPermutation(target)

            conditional[iBins, iPermutation] = calcConditionalEntropy(dataSet, objectBins, clPermutation['classesIndexes'])
            multiConditionalEntropy[iBins, iPermutation] = calcConditionalMultiVarianceEntropy(dataSet, objectBins, clPermutation['classesIndexes'])
            efficiency[iBins, iPermutation] = conditional[iBins, iPermutation] - multiConditionalEntropy[iBins, iPermutation]

            comparisonResults[iBins, iPermutation - 1, 0] = calcComparison(dataSet, objectBins, cl, clPermutation, 'bc')
            comparisonResults[iBins, iPermutation - 1, 1] = calcComparison(dataSet, objectBins, cl, clPermutation, 'ce')

        logitResults[iBins] = calcConditionalMultiVarianceEntropy(dataSet, objectBins, regressionClassesIndex['classesIndexes'])
        xgboostResults[iBins] = calcConditionalMultiVarianceEntropy(dataSet, objectBins, xbBoostClassesIndex['classesIndexes'])

        logitComparisonResults[iBins, 0] = calcComparison(dataSet, objectBins, cl, regressionClassesIndex, 'bc')
        logitComparisonResults[iBins, 1] = calcComparison(dataSet, objectBins, cl, regressionClassesIndex, 'ce')

        xgBoostComparisonResults[iBins, 0] = calcComparison(dataSet, objectBins, cl, xbBoostClassesIndex, 'bc')
        xgBoostComparisonResults[iBins, 1] = calcComparison(dataSet, objectBins, cl, xbBoostClassesIndex, 'ce')

        if abs(multiConditionalEntropy[iBins, 0]/ct - 1) < 0.01:
            break

    idx = np.where(multiConditionalEntropy[:, 0] != 0)[0]

    bins = bins[idx]
    simple = simple[idx]
    multiEntropy = multiEntropy[idx]

    conditional = conditional[idx, :]
    multiConditionalEntropy = multiConditionalEntropy[idx, :]
    efficiency = efficiency[idx, :]

    logitResults = logitResults[idx]
    xgboostResults = xgboostResults[idx]

    comparisonResults = comparisonResults[idx, :, :]
    logitComparisonResults = logitComparisonResults[idx, :]
    xgBoostComparisonResults = xgBoostComparisonResults[idx, :]

    mlResults = {'logit {0:1.2f}'.format(regressionAccuracy): logitResults, 'xgBoost {0:1.2f}'.format(xgBoostAccuracy): xgboostResults}
    comparisonMLResults = {'logit {0:1.2f}'.format(regressionAccuracy): logitComparisonResults, 'xgBoost {0:1.2f}'.format(xgBoostAccuracy): xgBoostComparisonResults}

    SaveComparisonResults(bins, comparisonResults, comparisonMLResults, taskName, nObjects, nFeatures)
    SaveEntropiesToFigure(bins, simple, conditional, multiEntropy, multiConditionalEntropy, efficiency, mlResults, taskName, nObjects, nFeatures, ct)
    saveFeaturesToClassesDistribution(dataSet, cl['classesIndexes'], upperBins, taskName)

    return



