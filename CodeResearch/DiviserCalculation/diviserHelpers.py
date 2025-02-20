import math

import numba as nb
import numpy as np
from numba import jit, prange
from rtree import index
from sortedcontainers import SortedDict

@jit(nopython=True)
def GetValuedTarget(target, c1, c1p, c2p):

    res = np.zeros(len(target))
    for iObject in range(0, len(target)):
        res[iObject] = c1p if target[iObject] == c1 else c2p

    return res

def GetSortedDict(dataSet):
    res = []
    nFeatures = dataSet.shape[1]
    nObjects = dataSet.shape[0]

    for iFeature in range(0, nFeatures):
        curDict = SortedDict()
        res.append(curDict)

        for iObject in range(0, nObjects):
            v = -dataSet[iObject, iFeature]
            if v not in curDict:
                curDict[v] = {iObject}
            else:
                curDict[v].add(iObject)

    return res

@jit(nopython=True)
def fv2s(f):
    s = ''
    for _ in f:
        s = s + f2s(_) + ', '
    return s

@jit(nopython=True)
def iv2s(f):
    s = ''
    for _ in f:
        s = s + str(_) + ', '
    return s

@jit(nopython=True)
def bv2s(f):
    s = ''
    for _ in f:
        s = s + ' ' + ('T' if _ else 'F')
    return s

@jit(nopython=True)
def f2s(f, precision=2):
    sign = '' if f >= 0 else '-'
    f = abs(f)

    if np.isnan(f):
        return 'NaN'
    ss = int(np.floor(f))
    s = sign + str(ss) + '.'
    digits = f - ss
    for _ in range(precision):
        digits *= 10
        ss = int(np.floor(digits))
        digits = digits - ss
        s += str(ss)
    return s

@jit(nopython=True)
def cut_trail(f_str):
    cut = 0
    for c in f_str[::-1]:
        if c == "0":
            cut += 1
        else:
            break
    if cut == 0:
        for c in f_str[::-1]:
            if c == "9":
                cut += 1
            else:
                cut -= 1
                break
    if cut > 0:
        f_str = f_str[:-cut]
    if f_str == "":
        f_str = "0"
    return f_str

@jit(nopython=True)
def float2str(value):
    if math.isnan(value):
        return "nan"
    elif value == 0.0:
        return "0.0"
    elif value < 0.0:
        return "-" + float2str(-value)
    elif math.isinf(value):
        return "inf"
    else:
        max_digits = 16
        min_digits = -4
        e10 = math.floor(math.log10(value)) if value != 0.0 else 0
        if min_digits < e10 < max_digits:
            i_part = math.floor(value)
            f_part = math.floor((1 + value % 1) * 10.0 ** max_digits)
            i_str = str(i_part)
            f_str = cut_trail(str(f_part)[1:max_digits - e10])
            return i_str + "." + f_str
        else:
            m10 = value / 10.0 ** e10
            exp_str_len = 4
            i_part = math.floor(m10)
            f_part = math.floor((1 + m10 % 1) * 10.0 ** max_digits)
            i_str = str(i_part)
            f_str = cut_trail(str(f_part)[1:max_digits])
            e_str = str(e10)
            if e10 >= 0:
                e_str = "+" + e_str
            return i_str + "." + f_str + "e" + e_str


@jit(nopython=True)
def getSortedByTarget(sortedObjects, dataSet, target):

    nObjects = len(dataSet)
    startIndex = 0
    prevObject = dataSet[sortedObjects[0]]
    result = np.zeros(nObjects, dtype=nb.int64)

    for iObject in range(1, nObjects):
        curObject = dataSet[sortedObjects[iObject]]

        if curObject == prevObject:
            continue

        curSlice = sortedObjects[startIndex:iObject]
        idx = np.argsort(target[curSlice])
        result[startIndex:iObject] = curSlice[idx]

        startIndex = iObject
        prevObject = curObject

    curSlice = sortedObjects[startIndex:nObjects]
    idx = np.argsort(target[curSlice])
    result[startIndex:nObjects] = curSlice[idx]

    return result

@jit(nopython=True, parallel=True)
def getSortedSet(dataSet, target):

    nFeatures = dataSet.shape[1]
    nObjects = dataSet.shape[0]

    res = np.zeros((nObjects, nFeatures), dtype=nb.int64)

    for iFeature in prange(0, nFeatures):
        sortedObjects = np.argsort(dataSet[:, iFeature])
        sortedByTarget = getSortedByTarget(sortedObjects, dataSet[:, iFeature], -target)
        res[:, iFeature] = sortedByTarget

    return res

def GetSortedDictByIndex(sortedIdx, objects):
    res = []
    nFeatures = objects.shape[1]
    nObjects = objects.shape[0]

    for iFeature in range(0, nFeatures):
        curDict = dict()
        res.append(curDict)

        for iObject in range(nObjects - 1, -1, -1):
            origIdx = sortedIdx[iObject, iFeature]
            curDict[origIdx] = objects[origIdx, iFeature]

    return res

def GetSortedDictList(dataSet):
    res = []
    nFeatures = dataSet.shape[1]
    nObjects = dataSet.shape[0]

    for iFeature in range(0, nFeatures):
        curDict = SortedDict()
        res.append(curDict)

        for iObject in range(0, nObjects):
            v = -dataSet[iObject, iFeature]
            if v not in curDict:
                curDict[v] = {iObject}
            else:
                curDict[v].add(iObject)

    return res

def getIdx(dataSet, id):
    nObjects = dataSet.shape[0]
    nFeatures = dataSet.shape[1]

    p = index.Property()
    p.dimension = nFeatures

    capacity = max(100, nObjects)

    p.index_capacity = capacity
    p.index_pool_capacity = capacity
    p.point_pool_capacity = capacity
    p.buffering_capacity = capacity
    p.leaf_capacity = capacity

    idx = index.Index(properties=p)
    idx.properties.dimension = nFeatures

    points = np.zeros((nObjects, 2*nFeatures))
    points[:, 0:nFeatures] = dataSet
    points[:, nFeatures:2*nFeatures] = dataSet

    #s1 = time.time()
    for iObject in range(0, nObjects):
        #print(iObject)
        idx.insert(id[iObject], points[iObject, :])
    #e1 = time.time()
    #print('Generated by {:}'.format(e1 - s1))

    return idx

def GetPowerOfSet(sortedNegDataSet):
    powers = np.zeros(len(sortedNegDataSet))

    for idx in range(0, len(sortedNegDataSet)):
        powers[idx] = len(sortedNegDataSet[idx])

    return powers

def getPointsUnderDiviser(idx, currentDiviser, basePoint):

    nFeatures = len(currentDiviser)
    query = np.zeros(2 * nFeatures)

    query[0:nFeatures] = basePoint
    query[nFeatures:2*nFeatures] = currentDiviser

    res = list(idx.intersection(query))

    return len(res)

def getPointsIdxUnderDiviser(idx, currentDiviser, basePoint):

    nFeatures = len(currentDiviser)
    query = np.zeros(2 * nFeatures)

    query[0:nFeatures] = basePoint
    query[nFeatures:2*nFeatures] = currentDiviser

    return set(idx.intersection(query))

def getBestStartDiviser(sortedNegValues, positiveScore, positiveCount, positiveIdx, basePoint):
    bestDiviser = sortedNegValues[- 1, :]

    positivePointsUnderDiviser = getPointsUnderDiviser(positiveIdx, bestDiviser, basePoint)
    #positivePointsUnderDiviser = getPointsBeforeDiviserIntersection(sortedPosValues, sortedPosIdx, bestDiviser)
    bestScore = (positiveCount - positivePointsUnderDiviser) * positiveScore

    return bestDiviser, bestScore

@jit(nopython=True)
def prepareDataSet(dataSet):
    nFeatures = dataSet.shape[1]

    usefulFeatures = np.zeros(nFeatures)

    for iFeature in range(0, nFeatures):
        uf = np.unique(dataSet[:, iFeature])
        if len(uf) != 1:
            usefulFeatures[iFeature] = 1

    idx = np.nonzero(usefulFeatures)[0]
    return dataSet[:, idx]
