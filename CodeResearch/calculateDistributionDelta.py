import bisect
import math
import statistics
import time

import numpy as np
from numpy import linalg as LA
from scipy.stats import bernoulli
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

from CodeResearch.calcSupremum import calcSupremum


def calculateDistributionDelta(dataSet, nObjects, nAttempts):

    totalObjects = dataSet.shape[0]
    mask = np.zeros(totalObjects)
    mask[np.arange(nObjects)] = 1

    supremums = np.zeros(nAttempts)

    for iPermutation in np.arange(nAttempts):

        mask = np.random.permutation(mask)
        idx = np.where(mask > 0)[0]
        subSet1 = dataSet[idx, :]

        mask = np.random.permutation(mask)
        idx = np.where(mask > 0)[0]
        subSet2 = dataSet[idx, :]

        supremums[iPermutation] = calcSupremum(subSet1, subSet2)

    avg = statistics.mean(supremums)
    sigma = statistics.stdev(supremums)

    return avg, sigma


def isLowerOrEqual(curObject, point):

    for i in np.arange(len(curObject)):
        if curObject[i] > point[i]:
            return False

    return True


def calculateVector(subSet, point, halfObjects):

    res = np.zeros(halfObjects, dtype=int)

    nObjects = subSet.shape[0]

    for iObject in np.arange(nObjects):
        curObject = subSet[iObject, :]

        if(isLowerOrEqual(curObject, point)):
            idx = iObject % halfObjects

            if iObject < halfObjects:
                res[idx] = res[idx] + 1
            else:
                res[idx] = res[idx] - 1

    return res


def calcRademacherComplexity(vectors, nAttempts):

    nDimension = len(vectors[0])
    nVectors = len(vectors)

    vs = np.array(vectors)
    bs = np.zeros((nDimension, nAttempts))
    rs = np.zeros((nVectors, nAttempts))

    for i in np.arange(nAttempts):
        bs[:, i] = 2 * (bernoulli.rvs(0.5, size=nDimension) - 0.5)

    prod = np.matmul(vs, bs, rs)
    res1 = np.amax(prod, axis=0)/nDimension
    resMean = np.mean(res1)
    resSigma = np.std(res1)

    #print('r: {0}, s: {1}', resMean, resSigma)

    return resMean, resSigma

    #multiplication without matrices - for memory
    idx = np.arange(1, nVectors)
    res = np.zeros(nAttempts)

    for i in np.arange(nAttempts):
        #b = 2 * (bernoulli.rvs(0.5, size=nDimension) - 0.5)
        b = bs[:, i]

        maxProd = abs(sum(k[0] * k[1] for k in zip(vectors[0], b)))
        for j in idx:
            curProd = abs(sum(k[0] * k[1] for k in zip(vectors[j], b)))
            maxProd = max(curProd, maxProd)

        res[i] = maxProd/nDimension

    avg = np.mean(res)
    sigma = np.std(res)

    #print('r: {0}, s: {1}', avg, sigma)

    return avg, sigma


def calculateSimpleVector(iVector, halfObjects):

    v = np.zeros(halfObjects, dtype=int)

    for i in np.arange(iVector):
        v[i%halfObjects] = 1

    return v


def calculateVectorFast(point, sortedSetIdx, sortedSet):

    nObjects = sortedSet.shape[0]
    halfObjects = math.floor(nObjects/2)
    nFeatures = sortedSet.shape[1]

    v = np.zeros(halfObjects, dtype=int)
    curSet = set()

    for iFeature in np.arange(nFeatures):
        idx = bisect.bisect_right(sortedSet[:, iFeature], point[iFeature])
        curFeaturesLessIdx = sortedSetIdx[np.arange(idx), iFeature]

        if iFeature == 0:
            curSet.update(curFeaturesLessIdx)
        else:
            curSet = curSet.intersection(curFeaturesLessIdx)

    for nIdx in curSet:
        if nIdx < halfObjects:
            v[nIdx] += 1
        else:
            v[nIdx % halfObjects] += -1

    return v


def calcRademacherVectorsFast(subSet):

    nObjects = subSet.shape[0]
    nFeatures = subSet.shape[1]
    
    sortedSetIdx = np.zeros((nObjects, nFeatures), dtype=int)
    sortedSet = np.zeros((nObjects, nFeatures))
    
    for iFeature in np.arange(nFeatures):
        sIdx = np.argsort(subSet[:, iFeature])
        sortedSetIdx[:, iFeature] = sIdx
        sortedSet[:, iFeature] = subSet[sIdx, iFeature]

    vectors = []
    vList = set()
    
    for iVector in np.arange(nObjects):
        v = calculateVectorFast(subSet[iVector, :], sortedSetIdx, sortedSet, subSet)

        prevLen = len(vList)
        vList.add(''.join(str(x) for x in v))
        curLen = len(vList)

        if curLen > prevLen:
            vectors.append(v)

    return vectors


def calcRademacherVectors(subSet):
    nObjects = subSet.shape[0]
    halfObjects = math.floor(nObjects / 2)
    
    vectors = []
    vList = set()

    sumVector = np.zeros(halfObjects)

    #s = time.time()

    for iVector in np.arange(nObjects):
        v = calculateVector(subSet, subSet[iVector, :], halfObjects)
        # v = calculateSimpleVector(iVector, halfObjects)

        prevLen = len(vList)
        vList.add(''.join(str(x) for x in v))
        curLen = len(vList)

        if curLen > prevLen:
            vectors.append(v)
            sumVector += v

    # if len(vList) != nObjects:
    #    print('Total objects: {0}, unique: {1}'.format(nObjects, len(vList)))

    # print(vList)

    #e = time.time()
    #print('Created rademacher vectors...{:.2f}'.format(e - s))

    sumVector = sumVector / len(vList)

    for iVector in np.arange(len(vectors)):
        vectors[iVector] = vectors[iVector] - sumVector
    
    return vectors


def calcRademacher(subSet, nAttempts):

    nObjects = subSet.shape[0]
    halfObjects = math.floor(nObjects / 2)

    vectors = calcRademacherVectorsFast(subSet)

    normUpper = 0.0

    for i in np.arange(len(vectors)):
        normUpper = max(normUpper, LA.norm(vectors[i]))

    upperRad = normUpper * np.sqrt(np.log(2 * 2 * len(vectors))) / halfObjects
    avg, sigma = calcRademacherComplexity(vectors, nAttempts)
    return {'rad': avg, 'sigma': sigma, 'upperRad': upperRad}


def calcConcreteModel(dataSet, nObjects, target):
    enc = LabelEncoder()
    target = enc.fit_transform(np.ravel(target))

    totalObjects = dataSet.shape[0]
    mask = np.zeros(totalObjects)
    mask[np.arange(nObjects)] = 1

    mask = np.random.permutation(mask)
    idx = np.where(mask > 0)[0]

    testIdx = np.where(mask == 0)[0]

    u, indexes = np.unique(target, return_index=True)
    s = set(np.concatenate((idx, indexes)))
    idx = list(s)

    subSet = dataSet[idx, :]
    subTarget = target[idx]

    train_idx = np.arange(nObjects, dtype=int)

    u, indexes = np.unique(subTarget, return_index=True)
    s = set(np.concatenate((train_idx, indexes)))
    train_idx = list(s)

    X_train = subSet[train_idx]
    Y_train = subTarget[train_idx]

    X_test = dataSet[testIdx]
    Y_test = target[testIdx]

    model = XGBClassifier().fit(X_train, Y_train)

    predict = model.predict(X_test)
    accuracy = accuracy_score(Y_test, predict)

    return accuracy

def calcModel(dataSet, nObjects, nAttempts, target):
    accuracy = np.zeros(nAttempts)

    for i in np.arange(nAttempts):
        accuracy[i] = calcConcreteModel(dataSet, nObjects, target)

    acc = np.mean(accuracy)
    sigma = np.std(accuracy)

    return {'accuracy': acc, 'modelSigma': sigma}


def calcRademacherForSets(dataSet, nObjects, nAttempts, nRadSets, target):
    totalObjects = dataSet.shape[0]

    upperRad = np.zeros(nRadSets)
    rad = np.zeros(nRadSets)
    sigmas = np.zeros(nRadSets)

    upperRadA = np.zeros(nRadSets)
    radA = np.zeros(nRadSets)
    sigmasA = np.zeros(nRadSets)

    for i in np.arange(nRadSets):
        mask = np.zeros(totalObjects)
        mask[np.arange(2 * nObjects)] = 1

        mask = np.random.permutation(mask)
        idx = np.where(mask > 0)[0]

        subSet = dataSet[idx, :]
        resA = calcRademacher(subSet, nAttempts)

        subTarget = target[idx]
        uTarget = np.unique(subTarget)
        nClasses = len(uTarget)

        subDTarget = np.zeros((len(subTarget), nClasses))

        for iClass in np.arange(nClasses):
            curSubIdx = np.where(subTarget == uTarget[iClass])[0]
            subDTarget[curSubIdx, iClass] = 1

        #subTarget = np.reshape(subTarget, (len(subTarget), 1))
        #subSet = np.hstack((subSet, subTarget))
        subSet = np.hstack((subSet, subDTarget))
        res = calcRademacher(subSet, nAttempts)

        rad[i] = res['rad']
        upperRad[i] = res['upperRad']
        sigmas[i] = res['sigma']

        radA[i] = resA['rad']
        upperRadA[i] = resA['upperRad']
        sigmas[i] = res['sigma']

    return {'rad': np.mean(rad), 'upperRad': np.mean(upperRad), 'sigma': np.mean(sigmas), 'radA': np.mean(radA), 'upperRadA': np.mean(upperRadA), 'sigmaA': np.mean(sigmasA)}


def calculateRademacherComplexity(dataSet, nObjects, nAttempts, modelAttempts, nRadSets, target):
    enc = LabelEncoder()
    target = enc.fit_transform(np.ravel(target))

    print('Calculating Rademacher & Model scores...')
    start = time.time()
    radResult = calcRademacherForSets(dataSet, nObjects, nAttempts, nRadSets, target)
    end = time.time()
    print('Calculated Rademacher: {:.2f}s'.format(end - start))

    start = time.time()
    modelResult = calcModel(dataSet, nObjects, modelAttempts, target)
    end = time.time()
    print('Calculated Model: {:.2f}s'.format(end - start))

    return {'radResult': radResult, 'modelResult': modelResult}
