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

from CodeResearch.calcModelEstimations import calcModel
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

def calculateVectorFast(point, sortedSetIdx, sortedSet, featuresIdx):

    nObjects = sortedSet.shape[0]
    halfObjects = math.floor(nObjects/2)
    nFeatures = sortedSet.shape[1]

    v = np.zeros(halfObjects, dtype=int)

    idx = bisect.bisect_right(sortedSet[:, featuresIdx[0]], point[featuresIdx[0]])
    curFeaturesLessIdx = sortedSetIdx[0:(idx - 1), featuresIdx[0]]
    curSet = set(curFeaturesLessIdx)

    searchDeltas = 0
    intersectDeltas = 0

    deltas = np.zeros(nFeatures)

    for fNumber in range(1, len(featuresIdx)):
        iFeature = featuresIdx[fNumber]
        #s1 = time.time()
        idx = bisect.bisect_right(sortedSet[:, iFeature], point[iFeature])
        curFeaturesLessIdx = sortedSetIdx[0:(idx - 1), iFeature]
        #print('iFeature: {:}, len: {:}, curLen: {:}, delta: {:}'.format(iFeature, len(curFeaturesLessIdx), len(curSet), deltas[featuresIdx[fNumber - 1]]))
        #e1 = time.time()
        #searchDeltas += (e1 - s1)

        #s1 = time.time()
        #tt = set()
        #temp = set(curFeaturesLessIdx)

        #for i in curSet:
        #    if i in temp:
        #        tt.add(i)

        #curSet = tt
        prevLen = len(curSet)
        curSet = curSet.intersection(curFeaturesLessIdx)
        deltas[iFeature] = prevLen - len(curSet)

        #e1 = time.time()
        #intersectDeltas += (e1 - s1)

        if (len(curSet)) == 0:
            break

    #for nIdx in curSet:
    #    if nIdx < halfObjects:
    #        v[nIdx] += 1
    #    else:
    #        v[nIdx % halfObjects] += -1

    nonZeroCoordinates = len(curSet)
    nonZeroDeltas = np.sum(np.abs(v))

    #print('Search: {:.2f}, intersection: {:.2f}, curLen: {:}'.format(searchDeltas, intersectDeltas, len(curSet)))

    return v, nonZeroCoordinates, nonZeroDeltas, deltas
    #return [], nonZeroCoordinates, -1, curSet, deltas


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
        v, nzc, nzd = calculateVectorFast(subSet[iVector, :], sortedSetIdx, sortedSet)

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

    upperRad = normUpper * np.sqrt(np.log(2 * len(vectors))) / halfObjects
    avg, sigma = calcRademacherComplexity(vectors, nAttempts)
    return {'rad': avg, 'sigma': sigma, 'upperRad': upperRad, 'alpha': normUpper**2/halfObjects}

def calcRademacherForSets(dataSet, nObjects, nAttempts, nRadSets, target):
    totalObjects = dataSet.shape[0]

    upperRad = np.zeros(nRadSets)
    rad = np.zeros(nRadSets)
    sigmas = np.zeros(nRadSets)
    upperRadAlpha = np.zeros(nRadSets)

    upperRadA = np.zeros(nRadSets)
    radA = np.zeros(nRadSets)
    sigmasA = np.zeros(nRadSets)
    upperRadAAlpha = np.zeros(nRadSets)

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

        subSet = np.hstack((subSet, subDTarget))
        res = calcRademacher(subSet, nAttempts)

        rad[i] = res['rad']
        upperRad[i] = res['upperRad']
        sigmas[i] = res['sigma']
        upperRadAlpha[i] = res['alpha']

        radA[i] = resA['rad']
        upperRadA[i] = resA['upperRad']
        sigmas[i] = resA['sigma']
        upperRadAAlpha[i] = resA['alpha']

    return {'rad': np.mean(rad), 'upperRad': np.mean(upperRad), 'sigma': np.mean(sigmas), 'radA': np.mean(radA), 'upperRadA': np.mean(upperRadA), 'sigmaA': np.mean(sigmasA), 'alpha': np.mean(upperRadAlpha), 'alphaA': np.mean(upperRadAAlpha)}


def GetSortedData(subSet):
    nObjects = subSet.shape[0]
    nFeatures = subSet.shape[1]

    sortedSetIdx = np.zeros((nObjects, nFeatures), dtype=int)
    sortedSet = np.zeros((nObjects, nFeatures))

    for iFeature in np.arange(nFeatures):
        sIdx = np.argsort(subSet[:, iFeature])
        sortedSetIdx[:, iFeature] = sIdx
        sortedSet[:, iFeature] = subSet[sIdx, iFeature]

    return sortedSetIdx, sortedSet

def ConvertVector(v, p1, p2):

    res = np.zeros(len(v))
    res[0] = v[0] * p1
    idx = np.arange(1, len(v))
    res[idx] = v[idx] * p2

    return res

def CalcRademacherDistributionDeltasXY(subSetX, subSetY):
    xObjects = subSetX.shape[0]
    yObjects = subSetY.shape[0]

    sortedSetIdxX, sortedSetX = GetSortedData(subSetX)
    sortedSetIdxY, sortedSetY = GetSortedData(subSetY)

    vectors = []

    vList = set()

    normUpper = 0.0
    featuresIdxX = np.arange(sortedSetIdxX.shape[1])
    featuresIdxY = np.arange(sortedSetIdxY.shape[1])

    for iVector in np.arange(xObjects + yObjects):

        curVector = subSetX[iVector, :] if iVector < xObjects else subSetY[iVector - xObjects, :]

        #s1 = time.time()
        vX, mX, nzdX, idxX = calculateVectorFast(curVector, sortedSetIdxX, sortedSetX, featuresIdxX)
        featuresIdxX = np.flip(np.argsort(idxX))
        vY, mY, nzdY, idxY = calculateVectorFast(curVector, sortedSetIdxY, sortedSetY, featuresIdxY)
        featuresIdxY = np.flip(np.argsort(idxY))
        #e1 = time.time()

        k = len(vX)
        m = len(vY)

        v = np.zeros(1 + k + m)
        v[0] = 0.5*(mX / k - mY / m)
        v[np.arange(1, 1 + k + m)] = 0.5 * np.concatenate((vX / k, -vY / m))

        #n1 = (0.5*(mX / k - mY / m))**2
        #nx = nzdX / (4 * k * k)
        #ny = nzdY / (4 * m * m)

        #nn = math.sqrt(n1 + nx + ny)
        #nla = LA.norm(v)

        normUpper = max(normUpper, LA.norm(v))
        #normUpper = max(normUpper, nn)

        v1 = ConvertVector(v, -1, 1)
        v2 = ConvertVector(v, 1, -1)
        v3 = ConvertVector(v, -1, -1)

        vs = [v, v1, v2, v3]

        for vv in vs:
            prevLen = len(vList)
            vList.add(''.join(str(x) for x in vv))
            curLen = len(vList)

            if curLen > prevLen:
                vectors.append(vv)

    return vectors, normUpper


def CalcRademacherDistributionDeltasForClasses(subSet, iClass, jClass, target, nAttempts):
    iIdx = np.where(target == iClass)[0]
    jIdx = np.where(target == jClass)[0]

    subSetX = subSet[iIdx, :]
    subSetY = subSet[jIdx, :]

    vectors, normUpper = CalcRademacherDistributionDeltasXY(subSetX, subSetY)

    upperRad = normUpper * np.sqrt(np.log(2 * len(vectors)))
    rad, sigma = calcRademacherComplexity(vectors, nAttempts)

    multiplier = (len(iIdx) + len(jIdx))/2 + 1

    return rad * multiplier, upperRad


def GetObjectsPerClass(target, seekingClass, nObjects):
    idx = np.where(target == seekingClass)[0]

    mask = np.zeros(len(idx))
    mask[np.arange(nObjects)] = 1

    mask = np.random.permutation(mask)
    idxM = np.where(mask > 0)[0]

    return idx[idxM].tolist()


def GetSubSet(dataSet, target, nObjects):
    vClasses, parts = np.unique(target, return_counts=True)
    parts = parts / len(target)

    nParts = np.floor(nObjects * parts).astype(int)

    objectsPerClass = 2 * np.maximum(np.ones(len(nParts), dtype=int), nParts)

    subSetIdx = []

    for iClass in np.arange(len(vClasses)):
        idx = GetObjectsPerClass(target, vClasses[iClass], objectsPerClass[iClass])
        subSetIdx = subSetIdx + idx

    return dataSet[subSetIdx], target[subSetIdx]

def calcRademacherDistributionDeltas(dataSet, nObjects, nAttempts, target):
    nClasses = len(np.unique(target))

    pairs = math.floor(nClasses * (nClasses - 1) / 2)
    rad = np.zeros(pairs)
    upperRad = np.zeros(pairs)

    subSet, subTarget = GetSubSet(dataSet, target, nObjects)
    curIdx = 0

    for iClass in np.arange(nClasses):
        for jClass in np.arange(iClass):
            r, ur = CalcRademacherDistributionDeltasForClasses(subSet, iClass, jClass, subTarget, nAttempts)

            rad[curIdx] = r
            upperRad[curIdx] = ur
            curIdx += 1

    return {'rad': rad, 'upperRad': upperRad}

def calcRademacherDeltasForSets(dataSet, nObjects, nAttempts, nRadSets, target):
    vClasses= np.unique(target)
    nClasses = len(vClasses)
    pairs = math.floor(nClasses * (nClasses - 1) / 2)

    upperRad = np.zeros((pairs, nRadSets), dtype=float)
    rad = np.zeros((pairs, nRadSets), dtype=float)

    for i in np.arange(nRadSets):
        print('Calculating for rad set #{0} of {1}'.format(i, nRadSets))
        res = calcRademacherDistributionDeltas(dataSet, nObjects, nAttempts, target)

        rad[:, i] = res['rad']
        upperRad[:, i] = res['upperRad']

    resRad = np.zeros(pairs)
    resUpperRad = np.zeros(pairs)

    for i in np.arange(pairs):
        resRad[i] = np.mean(rad[i, :])
        resUpperRad[i] = np.mean(upperRad[i, :])

    return {'rad': resRad, 'upperRad': resUpperRad}

def prepareData(dataSet):
    nObjects = dataSet.shape[0]
    nFeatures = dataSet.shape[1]
    resVectors = []

    print('Preparing data...')
    s1 = time.time()
    sortedIdx, sortedValues = GetSortedData(dataSet)
    e1 = time.time()
    print('Data prepared...{:.2f}'.format(e1 - s1))

    s1 = time.time()

    featuresIdx = np.arange(nFeatures)

    for iVector in range(0, nObjects):
        curVector = dataSet[iVector, :]

        #if iVector%100 == 0:
        e1 = time.time()
        print('Vector: {:}, time: {:.2f}'.format(iVector, e1-s1))

        v, m, nzd, idx, deltas = calculateVectorFast(curVector, sortedIdx, sortedValues, featuresIdx)
        featuresIdx = np.flip(np.argsort(deltas))
        resVectors.append(idx)

    return resVectors