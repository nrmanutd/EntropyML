import math

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import iqr
from sklearn.preprocessing import LabelEncoder

import scipy
from scipy.stats import entropy
from numpy import loadtxt
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression

def calcAndVisualize(dataSet, cl, binsPerFeature, taskName):
    """

    :type totalBins: int
    """
    nObjects = dataSet.shape[0]
    nClasses = len(cl)
    nFeatures = dataSet.shape[1]

    figure = plt.figure()
    px = 1 / plt.rcParams['figure.dpi']
    fig, ax = plt.subplots(nFeatures, nClasses + 1, tight_layout=True, figsize=(1920 * px, 1280 * px))

    for iFeature in np.arange(nFeatures):
        h = np.histogram(dataSet[:, iFeature], bins=binsPerFeature[iFeature], density=False)
        entBase = entropy(h[0])
        buckets = h[1]

        style = {'facecolor': 'none', 'edgecolor': 'C0', 'linewidth': 3}
        axAll = ax[iFeature, 0]
        axAll.hist(dataSet[:, iFeature], bins=buckets, **style)
        axAll.text(0.8, 0.8, "{0:2.1f}, {1:2.1f}".format(entBase, math.e ** entBase), horizontalalignment='center', verticalalignment='center',
                   transform=axAll.transAxes)
        for jClass in np.arange(nClasses):
            axTemp = ax[iFeature, jClass + 1]
            data = dataSet[cl[jClass], iFeature]

            style = {'facecolor': 'none', 'edgecolor': 'C{0}'.format((jClass + 1) % 10), 'linewidth': 3}
            axTemp.hist(data, bins=buckets, **style)
            h = np.histogram(data, bins=buckets, density=False)

            ent = entropy(h[0])
            axTemp.text(0.6, 0.8, "{0:2.1f}, {1:2.1f}, {2}".format(ent, math.e ** ent, len(cl[jClass])),
                        horizontalalignment='center', verticalalignment='center', transform=axTemp.transAxes)

    # plot the xdata locations on the x axis:

    fig.text(0.5, 0, 'Классы ({0}_{1}_{2})'.format(taskName, nObjects, nFeatures), ha='center')
    fig.text(0, 0.5, 'Признаки', va='center', rotation='vertical')

    plt.savefig('{0}_{1}_{2}_histogramms.png'.format(taskName, nObjects, nFeatures), format='png')
    plt.close(figure)

    return

def calcConditionalEntropy(dataSet, buckets, classIdx):
    nObjects = dataSet.shape[0]
    nClasses = len(classIdx)
    nFeatures = dataSet.shape[1]

    conditionalEntropy = 0.0

    for iFeature in np.arange(nFeatures):
        for jClass in np.arange(nClasses):
            idx = np.digitize(dataSet[classIdx[jClass], iFeature], buckets[iFeature])
            v, c = np.unique(idx, return_counts=True)
            ent = entropy(c / sum(c))
            p = sum(c) / nObjects
            conditionalEntropy += p * ent

    return conditionalEntropy
def calcConditionalMultiVarianceEntropy(dataSet, objectBins, classIdx):
    nObjects = dataSet.shape[0]
    nClasses = len(classIdx)

    conditionalMultiEntropy = np.zeros(nClasses, dtype=float)

    for iClass in np.arange(nClasses):
        mve = calcMultiVarianceEntropy(dataSet[classIdx[iClass], :], objectBins)
        conditionalMultiEntropy[iClass] = len(classIdx[iClass])/nObjects * mve

    return sum(conditionalMultiEntropy)

def calcMultiVarianceEntropy(dataSet, objectBins):
    nObjects = dataSet.shape[0]
    nFeatures = dataSet.shape[1]

    objectBinsIndexes = np.zeros((nObjects, nFeatures))

    for iFeature in np.arange(nFeatures):
        objectBinsIndexes[:, iFeature] = np.digitize(dataSet[:, iFeature], objectBins[iFeature])

    bucketIdx = [''] * nObjects

    for iObject in np.arange(nObjects):
        bucketIdx[iObject] = ''.join(str(x) for x in objectBinsIndexes[iObject, :])

    eBins, cBins = np.unique(bucketIdx, return_counts=True)
    ent = entropy(cBins/sum(cBins))

    return ent

def getOptimalBinsCount(dataSet):
    nFeatures = dataSet.shape[1]
    nObjects = dataSet.shape[0]

    return np.ones(nFeatures, dtype=int) * 20

    bins = np.arange(nFeatures)
    for i in np.arange(nFeatures):
        if nObjects > 1000:
            bins[i] = math.ceil(math.log(nObjects, 2) + 1)
        else:
            width = 2 * iqr(dataSet[:, i]) /(nObjects ** (1/3))
            bins[i] = math.ceil((max(dataSet[:, i]) - min(dataSet[:, i])) / width)
    return bins


def getPermutation(target):

    target = np.random.permutation(target)

    return getClassesIndex(target)

def getClassesIndex(target):
    entries = np.unique(target)
    classes = {}

    cl = []
    for iClass in np.arange(0, len(entries)):
        cl.append(np.where(target == entries[iClass])[0])
        classes[entries[iClass]] = iClass

    return {'classes':classes, 'classesIndexes':cl}


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

def SaveComparisonResults(bins, comparisonResults, mlResults, taskName, nObjects,
                          nFeatures):

    figure = plt.figure()
    px = 1 / plt.rcParams['figure.dpi']
    fig, ax = plt.subplots(2, 1, sharex=True, tight_layout=True, figsize=(1920 * px, 1280 * px))

    permutations = comparisonResults.shape[1]
    for iPermutation in np.arange(permutations):
        color = 'C{0}'.format(iPermutation % 10)
        ax[0].plot(bins, comparisonResults[:, iPermutation, 0], color)
        ax[1].plot(bins, comparisonResults[:, iPermutation, 1], color)

    ax[0].title.set_text('Comparison Bhattacharyya (new target to baseline)')
    ax[0].grid()

    markers = 'o*'
    mlKeys = list(mlResults.keys())
    for iKey in np.arange(len(mlKeys)):
        ax[0].plot(bins, mlResults[mlKeys[iKey]][:, 0], 'C{0}:{1}'.format((iKey + 1) % 10, markers[iKey % 2]), linewidth=2,
                   label=mlKeys[iKey])
        ax[1].plot(bins, mlResults[mlKeys[iKey]][:, 1], 'C{0}:{1}'.format((iKey + 1) % 10, markers[iKey % 2]), linewidth=2,
                   label=mlKeys[iKey])

    ax[1].title.set_text('Comparison Cross entropy (new target to baseline)')
    ax[1].grid()
    ax[1].legend()
    ax[0].legend()

    plt.savefig('{0}_{1}_{2}_comparisons.png'.format(taskName, nObjects, nFeatures), format='png')
    plt.close(fig)
    plt.close(figure)

    return


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
    calcAndVisualize(dataSet, cl['classesIndexes'], upperBins, taskName)

    return

def TrainLogit(dataSet, target):
    seed = 7
    test_size = 0.25
    X_train, X_test, y_train, y_test = train_test_split(dataSet, target, test_size=test_size, random_state=seed)
    logit = LogisticRegression(random_state=0, max_iter=1000).fit(X_train, np.ravel(y_train))

    return logit.predict(dataSet)

def TrainXGBoost(dataSet, target):
    seed = 7
    test_size = 0.25
    X_train, X_test, y_train, y_test = train_test_split(dataSet, target, test_size=test_size, random_state=seed)
    model = XGBClassifier().fit(X_train, np.ravel(y_train))

    return model.predict(dataSet)

def SaveEntropiesToFigure(bins, simple, conditional, multiEntropy, multiConditionalEntropy, efficiency, mlResults, taskName, nObjects, nFeatures, ct):

    figure = plt.figure()
    px = 1 / plt.rcParams['figure.dpi']
    fig, ax = plt.subplots(3, 1, sharex=True, tight_layout=True, figsize=(1920 * px, 1280 * px))

    permutations = efficiency.shape[1]

    ax[0].plot(bins, efficiency[:, 0], 'C0-.', linewidth=3)
    for iPermutation in np.arange(1, permutations):
        color = 'C{0}'.format(iPermutation%10)
        ax[0].plot(bins, efficiency[:, iPermutation], color)
        ax[1].plot(bins, conditional[:, iPermutation], color)
        ax[2].plot(bins, multiConditionalEntropy[:, iPermutation], color)


    ax[0].title.set_text('Efficiency (simple conditional - multi conditional)')
    ax[0].grid()

    ax[1].plot(bins, simple, 'C0--', linewidth=2)
    ax[1].plot(bins, conditional[:, 0], 'C0-.', linewidth=3)
    ax[1].title.set_text('Simple and conditional entropy')
    ax[1].text(bins[0], 2 / 3 * max(simple) + 1 / 3 * simple[0],
               'Task: {0}\nObjects: {1}\nFeatures: {2}\nUpper: {3:3.2f}'.format(taskName, nObjects, nFeatures,                                                                                 ct * nFeatures))
    ax[1].grid()

    ax[2].plot(bins, multiEntropy, 'C0--', linewidth=2)
    ax[2].plot(bins, multiConditionalEntropy[:, 0], 'C0-.', linewidth=3)

    markers = 'o*'
    mlKeys = list(mlResults.keys())
    for iKey in np.arange(len(mlKeys)):
        ax[2].plot(bins, mlResults[mlKeys[iKey]], 'C{0}:{1}'.format((iKey + 1) % 10, markers[iKey%2]), linewidth=2, label=mlKeys[iKey])

    ax[2].text(bins[0], 2 / 3 * max(multiEntropy) + 1 / 3 * multiEntropy[0], 'Upper: {0:3.2f}'.format(ct))
    ax[2].title.set_text('Simple and conditional multi entropy')
    ax[2].grid()
    ax[2].legend()

    plt.savefig('{0}_{1}_{2}_entropies.png'.format(taskName, nObjects, nFeatures), format='png')
    plt.close(fig)
    plt.close(figure)

    return