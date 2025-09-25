import time

import numpy as np
import torch
from numba import cuda
from sklearn.preprocessing import LabelEncoder

from CodeResearch.Cuda.cudaHelpers import filterSortedSetByIndex
from CodeResearch.DataSeparationFramework.Metrics import BaseMetricCalculator
from CodeResearch.DiviserCalculation.diviserHelpers import getSortedSet, GetValuedAndBoolTarget, prepareDataSet, \
    GetValuedTarget
from CodeResearch.DiviserCalculation.getDiviserFastCuda import getMaximumDiviserFastCudaCore
from CodeResearch.DiviserCalculation.getDiviserFastNumba import getMaximumDiviserFastNumbaCore
from CodeResearch.Helpers.permutationHelpers import getDataSetIndexesOfTwoClasses
from CodeResearch.ObjectComplexity.Factory import BaseComplexityCalculatorFactory
from CodeResearch.calcModelEstimations import calcNN, calcXGBoost

class PValueCalculator:

    def __init__(self, complexityCalculatorFactory: BaseComplexityCalculatorFactory, metricCalculator: BaseMetricCalculator, nAttempts, calculateKS, randomPermutation, calculateModel):

        self.complexityCalculatorFactory = complexityCalculatorFactory
        self.metricCalculator = metricCalculator
        self.calculateModel = calculateModel
        self.randomPermutation = randomPermutation
        self.calculateKS = calculateKS
        self.nAttempts = nAttempts

    def calcPValueFastPro(self, currentObjects, dataSet, target, iClass, jClass):
        nFeatures = dataSet.shape[1]

        if not torch.cuda.is_available():
            return self.calcPValuesCpuNumba(currentObjects, dataSet, target, iClass, jClass, self.nAttempts, self.calculateKS, self.randomPermutation, self.calculateModel)

        if nFeatures < 1000:
            return self.calcPValueFastNumba(currentObjects, dataSet, target, iClass, jClass, self.nAttempts, self.calculateKS, self.randomPermutation, self.calculateModel)
        else:
            return self.calcPValueFastCuda(currentObjects, dataSet, target, iClass, jClass, self.nAttempts, self.calculateKS, self.randomPermutation, self.calculateModel)

    def calcPValuesCpuNumba(self, currentObjects, dataSet, target, iClass, jClass, nAttempts, calculateKS = True, randomPermutation=False, calculateModel=False):
        iObjects = list(np.where(target == iClass)[0])
        jObjects = list(np.where(target == jClass)[0])
        objectsIdx = iObjects + jObjects

        values = np.zeros(nAttempts)
        outOfSampleValues = np.zeros(nAttempts)

        NNvalues = np.zeros(nAttempts)

        currentTime = time.time()

        twoClassObjects = np.arange(len(objectsIdx))
        ds = dataSet[objectsIdx, :]
        ds = prepareDataSet(ds)
        t = target[objectsIdx]

        enc = LabelEncoder()
        #complexityCalculator = KSComplexityCalculator(ds, t, objectsIdx)
        complexityCalculator = self.complexityCalculatorFactory.createComplexityCalculator(ds, t, objectsIdx)

        preparationTime = 0
        ksTime = 0
        NNTime = 0

        for iAttempt in range(nAttempts):
            if iAttempt % 10 == 0:
                print('Attempt #' + str(iAttempt) + ' Time: ' + str(time.time() - currentTime) + ' Preparation time: ' + str(preparationTime), ' KS time: ' + str(ksTime) + ' NN time: ' + str(NNTime))
                preparationTime = 0
                ksTime = 0
                NNTime = 0
                currentTime = time.time()

            t1 = time.time()

            iClassIdx, jClassIdx = getDataSetIndexesOfTwoClasses(currentObjects, t, iClass, jClass)
            idx = np.concatenate((iClassIdx, jClassIdx))

            tClasses = t[idx]
            dsClasses = ds[idx, :]
            preparationTime += (time.time() - t1)

            if calculateModel:
                t2 = time.time()
                tt= enc.fit_transform(np.ravel(t))
                tClasses = tt[idx]

                if randomPermutation:
                    tClasses = np.random.permutation(tClasses)

                testIdx = np.setdiff1d(twoClassObjects, idx)
                testDs = ds[testIdx, :]
                testTClasses = tt[testIdx]

                preparationTime += (time.time() - t2)
                t2 = time.time()
                #NNvalues[iAttempt] = calcNN(dsClasses, tClasses, testDs, testTClasses)
                NNvalues[iAttempt] = calcXGBoost(dsClasses, tClasses, testDs, testTClasses)
                NNTime += (time.time() - t2)

            if calculateKS:
                t2 = time.time()

                if randomPermutation:
                    tClasses = np.random.permutation(tClasses)

                nClasses, counts = np.unique(tClasses, return_counts=True)
                vt1 = GetValuedTarget(tClasses, nClasses[0], 1 / counts[0], -1 / counts[1])
                vt2 = GetValuedTarget(tClasses, nClasses[1], 1 / counts[1], -1 / counts[0])

                sds1 = getSortedSet(dsClasses, vt1)
                sds2 = getSortedSet(dsClasses, vt2)

                preparationTime += (time.time() - t2)
                t2 = time.time()
                #v, d = getMaximumDiviserFastNumbaCore(dsClasses, tClasses, vt1, sds1, vt2, sds2)
                v, d, c = self.metricCalculator.calculateMetricPro(dsClasses, tClasses, vt1, sds1, vt2, sds2)
                values[iAttempt] = v
                ksTime += (time.time() - t2)

                outOfSampleValues[iAttempt] = complexityCalculator.updateComplexity(d, c, idx)

                #outOfSampleValues[iAttempt] = complexityCalculator.calculateKSOutOfIdx(d, idx)
                #complexityCalculator.addComplexityOutOfIdx(d, idx)

        return values, NNvalues, complexityCalculator, outOfSampleValues

    def calcPValueFastNumba(self, currentObjects, dataSet, target, iClass, jClass, nAttempts, calculateKS = True, randomPermutation=False, calculateModel=False):
        iObjects = list(np.where(target == iClass)[0])
        jObjects = list(np.where(target == jClass)[0])
        objectsIdx = iObjects + jObjects

        values = np.zeros(nAttempts)
        NNvalues = np.zeros(nAttempts)

        currentTime = time.time()

        twoClassObjects = np.arange(len(objectsIdx))
        ds = dataSet[objectsIdx, :]
        ds = prepareDataSet(ds)
        t = target[objectsIdx]

        enc = LabelEncoder()

        nClasses, counts = np.unique(t, return_counts=True)
        valuedTarget1, boolValuedTarget1 = GetValuedAndBoolTarget(t, nClasses[0], 1 / counts[0], -1 / counts[1])
        valuedTarget2, boolValuedTarget2 = GetValuedAndBoolTarget(t, nClasses[1], 1 / counts[1], -1 / counts[0])

        sds1 = getSortedSet(ds, valuedTarget1)
        sds2 = getSortedSet(ds, valuedTarget2)

        sds1_device = cuda.to_device(sds1)
        sds2_device = cuda.to_device(sds2)

        preparationTime = 0
        ksTime = 0
        NNTime = 0

        for iAttempt in range(nAttempts):
            if iAttempt % 10 == 0:
                print('Attempt #' + str(iAttempt) + ' Time: ' + str(time.time() - currentTime) + ' Preparation time: ' + str(preparationTime), ' KS time: ' + str(ksTime) + ' NN time: ' + str(NNTime))
                preparationTime = 0
                ksTime = 0
                NNTime = 0
                currentTime = time.time()

            t1 = time.time()

            iClassIdx, jClassIdx = getDataSetIndexesOfTwoClasses(currentObjects, t, iClass, jClass)
            idx = np.concatenate((iClassIdx, jClassIdx))

            tClasses = t[idx]

            dsClasses = ds[idx, :]
            preparationTime += (time.time() - t1)

            if calculateModel:
                t2 = time.time()
                tt= enc.fit_transform(np.ravel(t))
                tClasses = tt[idx]

                if randomPermutation:
                    tClasses = np.random.permutation(tClasses)

                testIdx = np.setdiff1d(twoClassObjects, idx)
                testDs = ds[testIdx, :]
                testTClasses = tt[testIdx]

                preparationTime += (time.time() - t2)
                t2 = time.time()
                #NNvalues[iAttempt] = calcNN(dsClasses, tClasses, testDs, testTClasses)
                NNvalues[iAttempt] = calcXGBoost(dsClasses, tClasses, testDs, testTClasses)
                NNTime += (time.time() - t2)

            if calculateKS:
                t2 = time.time()

                if randomPermutation:
                    tClasses = np.random.permutation(tClasses)

                nClasses, counts = np.unique(tClasses, return_counts=True)
                vt1 = GetValuedTarget(tClasses, nClasses[0], 1 / counts[0], -1 / counts[1])
                vt2 = GetValuedTarget(tClasses, nClasses[1], 1 / counts[1], -1 / counts[0])
                ss1 = filterSortedSetByIndex(sds1_device, sds1.shape[0], sds1.shape[1], idx)
                ss2 = filterSortedSetByIndex(sds2_device, sds2.shape[0], sds2.shape[1], idx)

                preparationTime += (time.time() - t2)
                t2 = time.time()
                values[iAttempt] = getMaximumDiviserFastNumbaCore(dsClasses, tClasses, vt1, ss1, vt2, ss2)[0]
                ksTime += (time.time() - t2)

        return values, NNvalues

    def calcPValueFastCuda(self, currentObjects, dataSet, target, iClass, jClass, nAttempts, calculateKS = True, randomPermutation = False, calculateModel = False):
        iObjects = list(np.where(target == iClass)[0])
        jObjects = list(np.where(target == jClass)[0])
        objectsIdx = iObjects + jObjects

        values = np.zeros(nAttempts)
        NNvalues = np.zeros(nAttempts)

        currentTime = time.time()
        enc = LabelEncoder()

        objectsIdx = np.array(objectsIdx)

        twoClassObjects = np.arange(len(objectsIdx))
        ds = dataSet[objectsIdx, :]
        ds = prepareDataSet(ds)
        t = target[objectsIdx]

        nClasses, counts = np.unique(t, return_counts=True)
        valuedTarget1, boolValuedTarget1 = GetValuedAndBoolTarget(t, nClasses[0], 1 / counts[0], -1 / counts[1])
        valuedTarget2, boolValuedTarget2 = GetValuedAndBoolTarget(t, nClasses[1], 1 / counts[1], -1 / counts[0])

        sds1 = getSortedSet(ds, valuedTarget1)
        sds1_device = cuda.to_device(sds1)

        sds2 = getSortedSet(ds, valuedTarget2)
        sds2_device = cuda.to_device(sds2)

        preparationTime = 0
        nnTime = 0
        ksTime = 0

        for iAttempt in range(nAttempts):
            if iAttempt % 10 == 0:
                print('Attempt #' + str(iAttempt) + ' Time: ' + str(time.time() - currentTime) + ' Preparation time: ' + str(preparationTime) + ' KS cuda calculation: ' + str(ksTime) + ' NN time calculation: ' + str(nnTime))
                currentTime = time.time()

                nnTime = 0
                ksTime = 0
                preparationTime = 0

            #newSet, newTarget = getDataSetOfTwoClasses(currentObjects, dataSet, target, iClass, jClass)
            #values[iAttempt] = getMaximumDiviserFastCuda(newSet, newTarget)[0]
            #continue

            t1 = time.time()

            iClassIdx, jClassIdx = getDataSetIndexesOfTwoClasses(currentObjects, t, iClass, jClass)
            idx = np.concatenate((iClassIdx, jClassIdx))

            dsClasses = ds[idx, :]

            preparationTime += (time.time() - t1)

            if calculateModel:
                t2 = time.time()
                tt= enc.fit_transform(np.ravel(t))

                tClasses = tt[idx]

                if randomPermutation:
                    tClasses = np.random.permutation(tClasses)

                testIdx = np.setdiff1d(twoClassObjects, idx)
                testDs = ds[testIdx, :]
                testTClasses = tt[testIdx]

                preparationTime += (time.time() - t2)

                t2 = time.time()
                NNvalues[iAttempt] = calcNN(dsClasses, tClasses, testDs, testTClasses)
                nnTime += (time.time() - t2)

            if calculateKS:
                t2 = time.time()
                tClasses = t[idx]

                if randomPermutation:
                    tClasses = np.random.permutation(tClasses)

                nClasses, counts = np.unique(tClasses, return_counts=True)
                vt1, bvt1 = GetValuedAndBoolTarget(tClasses, nClasses[0], 1 / counts[0], -1 / counts[1])
                vt2, bvt2 = GetValuedAndBoolTarget(tClasses, nClasses[1], 1 / counts[1], -1 / counts[0])

                ss1 = filterSortedSetByIndex(sds1_device, sds1.shape[0], sds1.shape[1], idx)
                ss2 = filterSortedSetByIndex(sds2_device, sds2.shape[0], sds2.shape[1], idx)

                ss1_device = cuda.to_device(ss1)
                ss2_device = cuda.to_device(ss2)

                dsClasses_device = cuda.to_device(dsClasses)
                preparationTime += time.time() - t2

                t2 = time.time()
                values[iAttempt] = getMaximumDiviserFastCudaCore(dsClasses, dsClasses_device, tClasses, ss1, ss1_device, vt1, bvt1, ss2, ss2_device, vt2, bvt2)[0]
                ksTime += time.time() - t2

        return values, NNvalues
