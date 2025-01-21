import bisect
import math
import statistics
import time

import numpy as np
from numpy import linalg as LA
from scipy.stats import bernoulli

from CodeResearch.DiviserCalculation.getCorrectDiviser import getMaximumDiviserCorrect
from CodeResearch.DiviserCalculation.getDiviserFast import getMaximumDiviserFast
from CodeResearch.DiviserCalculation.getDiviserRTree import getMaximumDiviserRTree
from CodeResearch.DiviserCalculation.statisticsCalculation import getMaximumPossibleByAnalysis
from CodeResearch.calcSupremum import calcSupremum
from CodeResearch.rademacherHelpers import GetSortedData, ConvertVector, GetSubSet


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

def calculateVectorFast(point, sortedSetIdx, sortedSet, isSelfVector):
    nObjects = sortedSet.shape[0]
    halfObjects = math.floor(nObjects/2)
    nFeatures = sortedSet.shape[1]

    v = np.zeros(halfObjects, dtype=int)

    featuresIdx = np.zeros(nFeatures)

    minCount = 1 if isSelfVector else 0

    for iFeature in range(0, nFeatures):
        idx = bisect.bisect_right(sortedSet[:, iFeature], point[iFeature])
        featuresIdx[iFeature] = idx

    featuresIdx = np.argsort(featuresIdx)
    #featuresIdx = np.arange(nFeatures)

    idx = bisect.bisect_right(sortedSet[:, featuresIdx[0]], point[featuresIdx[0]])
    curFeaturesLessIdx = sortedSetIdx[0:idx, featuresIdx[0]]
    curSet = set(curFeaturesLessIdx)

    for fNumber in range(1, len(featuresIdx)):
        iFeature = featuresIdx[fNumber]
        #s1 = time.time()
        idx = bisect.bisect_right(sortedSet[:, iFeature], point[iFeature])
        curFeaturesLessIdx = sortedSetIdx[0:idx, iFeature]
        curSet = curSet.intersection(curFeaturesLessIdx)

        if len(curSet) == minCount:
            break

    for nIdx in curSet:
        if nIdx < halfObjects:
            v[nIdx] += 1
        else:
            v[nIdx % halfObjects] += -1

    nonZeroCoordinates = len(curSet)
    nonZeroDeltas = np.sum(np.abs(v))

    #print('Search: {:.2f}, intersection: {:.2f}, curLen: {:}'.format(searchDeltas, intersectDeltas, len(curSet)))

    return v, nonZeroCoordinates, nonZeroDeltas

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

def CalcRademacherDistributionDeltasXY(subSetX, subSetY, maxDelta):
    xObjects = subSetX.shape[0]
    yObjects = subSetY.shape[0]

    sortedSetIdxX, sortedSetX = GetSortedData(subSetX)
    sortedSetIdxY, sortedSetY = GetSortedData(subSetY)

    vectors = []
    vList = set()
    normUpper = 0.0

    v0x_max = 0.0
    v0y_max = 0.0
    vx_max = 0.0
    vy_max = 0.0

    for iVector in np.arange(xObjects + yObjects):
        curVector = subSetX[iVector, :] if iVector < xObjects else subSetY[iVector - xObjects, :]

        #s1 = time.time()
        if iVector < xObjects:
            vX, mX, nzdX = calculateVectorFast(curVector, sortedSetIdxX, sortedSetX, True)
            vY, mY, nzdY = calculateVectorFast(curVector, sortedSetIdxY, sortedSetY, False)
        else:
            vX, mX, nzdX = calculateVectorFast(curVector, sortedSetIdxX, sortedSetX, False)
            vY, mY, nzdY = calculateVectorFast(curVector, sortedSetIdxY, sortedSetY, True)
        #e1 = time.time()

        k = len(vX)
        m = len(vY)

        v = np.zeros(1 + k + m)
        v[0] = 0.5*(mX / k - mY / m)
        v[1:(1 + k)] = 0.5 * vX / k
        v[(1 + k):(1 + k + m)] = - 0.5 * vY / m

        curNorm = LA.norm(v)

        if curNorm > normUpper:
            normUpper = curNorm
            v0x_max = mX
            v0y_max = mY
            vx_max = np.sum(np.abs(vX))
            vy_max = np.sum(np.abs(vY))

        if abs(normUpper) < 0.00001:
            raise ValueError('Error: {:}'.format(normUpper))

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

    v0 = (v0x_max/xObjects - v0y_max/yObjects)
    if abs(v0) > abs(maxDelta):
        raise ValueError('Error: maxDelta <= v0')

    print('NormUpper: {:}, v[0] = {:}, maxDelta = {:} v0x = {:}, v0y = {:}, vx = {:}, vy = {:}, 2k = {:}, 2m = {:}'.format(
            normUpper, v0, maxDelta, v0x_max, v0y_max, vx_max, vy_max, xObjects, yObjects))

    return vectors, normUpper


def CalcRademacherDistributionDeltasForClasses(subSet, iClass, jClass, target, nAttempts):
    iIdx = np.where(target == iClass)[0]
    jIdx = np.where(target == jClass)[0]

    subSetX = subSet[iIdx, :]
    subSetY = subSet[jIdx, :]

    #maxDelta, maxValues = getMaximumDiviser(subSet, target)
    s1 = time.time()
    maxDelta3, maxValues3 = getMaximumDiviserFast(subSet, target)
    e1 = time.time()
    dFast = e1 - s1

    #s1 = time.time()
    maxDelta5, maxValues5, possibleBest = getMaximumDiviserRTree(subSet, target)
    #maxDelta5 = maxDelta3
    #e1 = time.time()
    #dRTree = e1 - s1
    #print('Stable diviser: {:}/{:}, prod diviser: {:}/{:}'.format(maxDelta, maxValues, maxDelta2, maxValues2))
    #print('Prod diviser: {:}/{:.2f}s, fast diviser: {:}/{:.2f}s'.format(maxDelta2, dProd, maxDelta3, dFast))
    #print('Correct diviser: {:}/{:.2f}s, fast diviser: {:}/{:.2f}s'.format(maxDelta4, dCorrect, maxDelta3, dFast))
    #print('RTree diviser: {:}/{:.2f}s, fast diviser: {:}/{:.2f}s'.format(maxDelta5, dRTree, maxDelta3, dFast))

    #if abs(maxDelta5-maxDelta3) > 0.000001:
    #    print(subSet)
    #    print(target)

    #   raise ValueError('RTree diviser differs with prod. RTree diviser: {:}/{:}, fast diviser: {:}/{:}'.format(maxDelta5, maxValues5, maxDelta3, maxValues3))
        #print('RTree diviser differs with prod. RTree diviser: {:}/{:}, fast diviser: {:}/{:}'.format(maxDelta5, maxValues5, maxDelta3, maxValues3))

    #vectors, normUpper = CalcRademacherDistributionDeltasXY(subSetX, subSetY, maxDelta2)

    nFeatures = subSet.shape[1]
    xObjects = len(iIdx)
    yObjects = len(jIdx)

    #upperRad = normUpper * np.sqrt(np.log(2 * len(vectors)))
    #normUpper = math.sqrt(maxDelta3 ** 2 + 1 / xObjects + 1 / yObjects)
    #upperRad = normUpper * np.sqrt(math.log(8) + nFeatures * math.log(xObjects + yObjects))
    upperRad = maxDelta3
    upperRad2 = maxDelta5
    print('KS1(Fast) = {:}, KS2(RTree) = {:}, K2(Upper) = {:}'.format(maxDelta3, maxDelta5, possibleBest))
    #rad, sigma = calcRademacherComplexity(vectors, nAttempts)

    multiplier = (len(iIdx) + len(jIdx))/2 + 1

    #return rad * multiplier, upperRad
    return upperRad2, upperRad

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