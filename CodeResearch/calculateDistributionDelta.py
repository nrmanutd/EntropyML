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
            rr = res[iObject % halfObjects] + 1
            res[iObject % halfObjects] = rr % 2

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


def calcRademacher(subSet, nAttempts):

    nObjects = subSet.shape[0]
    halfObjects = (int) (nObjects / 2)

    vectors = []
    vList = set()

    sumVector = np.zeros(halfObjects)

    for iVector in np.arange(nObjects):
        v = calculateVector(subSet, subSet[iVector, :], halfObjects)

        prevLen = len(vList)
        vList.add(''.join(str(x) for x in v))
        curLen = len(vList)

        if curLen > prevLen:
            vectors.append(v)
            sumVector += v

    #if len(vList) != nObjects:
    #    print('Total objects: {0}, unique: {1}'.format(nObjects, len(vList)))

    sumVector = sumVector / len(vList)
    maxNorm = np.zeros(len(vectors))

    for iVector in np.arange(len(vectors)):
        vectors[iVector] = vectors[iVector] - sumVector
        maxNorm[iVector] = LA.norm(vectors[iVector])

    avg2 = max(maxNorm) * np.sqrt(2 * np.log(len(vectors))) / halfObjects
    avg, sigma = calcRademacherComplexity(vectors, nAttempts)

    print('Rademacher monte-carlo vs estimation {0} {1}'.format(avg, avg2))
    return avg, sigma, avg2


def calculateRademacherComplexity(dataSet, nObjects, nAttempts, target):

    totalObjects = dataSet.shape[0]
    mask = np.zeros(totalObjects)
    mask[np.arange(2*nObjects)] = 1

    mask = np.random.permutation(mask)
    idx = np.where(mask > 0)[0]
    subSet = dataSet[idx, :]

    rad, sigma, rad2 = calcRademacher(subSet, nAttempts)

    enc = LabelEncoder()
    target = enc.fit_transform(np.ravel(target))

    train_idx = np.arange(nObjects, dtype=int)
    X_train = subSet[train_idx]
    Y_train = target[idx][train_idx]

    test_idx = np.arange(nObjects, 2*nObjects, dtype=int)
    X_test = subSet[test_idx]
    Y_test = target[idx][test_idx]

    model = XGBClassifier().fit(X_train, np.ravel(Y_train))

    #model = XGBClassifier().fit(dataSet, np.ravel(target))
    predict = model.predict(X_test)
    accuracy = accuracy_score(Y_test, predict)

    return rad, sigma, rad2, accuracy
