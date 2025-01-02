import random
import statistics
from scipy.stats import bernoulli
import numpy as np
from numpy import linalg as LA
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
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

    res = np.zeros(nAttempts)
    nDimension = len(vectors[0])
    nVectors = len(vectors)

    for i in np.arange(nAttempts):
        b = 2 * (bernoulli.rvs(0.5, size=nDimension) - 0.5)

        maxProd = sum(k[0] * k[1] for k in zip(vectors[0], b))
        for j in np.arange(1, nVectors):
            curProd = sum(k[0] * k[1] for k in zip(vectors[j], b))
            maxProd = max(curProd, maxProd)

        res[i] = maxProd/nDimension

    avg = np.mean(res)
    sigma = np.std(res)

    return avg, sigma


def calculateSimpleVector(iVector, halfObjects):

    v = np.zeros(halfObjects, dtype=int)

    for i in np.arange(iVector):
        v[i%halfObjects] = 1

    return v


def calcRademacher(subSet, nAttempts):

    nObjects = subSet.shape[0]
    halfObjects = (int) (nObjects / 2)

    vectors = []
    vList = set()

    sumVector = np.zeros(halfObjects)

    for iVector in np.arange(nObjects):
        v = calculateVector(subSet, subSet[iVector, :], halfObjects)
        #v = calculateSimpleVector(iVector, halfObjects)

        prevLen = len(vList)
        vList.add(''.join(str(x) for x in v))
        curLen = len(vList)

        if curLen > prevLen:
            vectors.append(v)
            sumVector += v

    #if len(vList) != nObjects:
    #    print('Total objects: {0}, unique: {1}'.format(nObjects, len(vList)))

    #print(vList)

    sumVector = sumVector / len(vList)
    maxNorm = np.zeros(len(vectors))

    for iVector in np.arange(len(vectors)):
        vectors[iVector] = vectors[iVector] - sumVector
        maxNorm[iVector] = LA.norm(vectors[iVector])

    normUpper = max(maxNorm)
    upperRad = normUpper * np.sqrt(2 * np.log(len(vectors))) / halfObjects
    upperRad2 = np.sqrt(halfObjects) * np.sqrt(2 * np.log(len(vectors))) / halfObjects
    avg, sigma = calcRademacherComplexity(vectors, nAttempts)

    print('Rademacher monte-carlo vs estimation {0} {1}'.format(avg, upperRad))
    return {'rad': avg, 'sigma': sigma, 'upperRad': upperRad, 'upperRad2': upperRad2}


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


def calculateRademacherComplexity(dataSet, nObjects, nAttempts, modelAttempts, target):

    totalObjects = dataSet.shape[0]
    mask = np.zeros(totalObjects)
    mask[np.arange(2*nObjects)] = 1

    mask = np.random.permutation(mask)
    idx = np.where(mask > 0)[0]

    enc = LabelEncoder()
    target = enc.fit_transform(np.ravel(target))

    subSet = dataSet[idx, :]
    subTarget = target[idx]
    subTarget = np.reshape(subTarget, (len(subTarget), 1))
    subSet = np.hstack((subSet, subTarget))

    radResult = calcRademacher(subSet, nAttempts)
    modelResult = calcModel(dataSet, nObjects, modelAttempts, target)

    return {'radResult': radResult, 'modelResult': modelResult}
