import numpy as np
import torch
from numba import cuda

from CodeResearch.DataSeparationFramework.Metrics import BaseMetricCalculator
from CodeResearch.DiviserCalculation.diviserHelpers import GetValuedTarget, getSortedSet, GetValuedAndBoolTarget
from CodeResearch.Helpers.Logger import BaseLogger
from CodeResearch.LearningFramework.Learners.baseLearner import BaseLearner


class KSLearner(BaseLearner):
    def __init__(self, metricCalculator: BaseMetricCalculator, logger: BaseLogger):

        self.metricCalculator = metricCalculator
        self.logger = logger

    def trainAndTest(self, x, y, probs, xt, yt):
        model = self.train(x, y, probs)
        return self.test(model, xt, yt)

    def test(self, model, x, y):
        predictions = np.zeros(len(y))

        diviser = model[0]
        classUnderDiviser = model[1]
        oppositeClassUnderDiviser = 1 if classUnderDiviser == 0 else 0

        positiveObjects = 0
        negativeObjects = 0
        positiveObjectsCount = 0
        negativeObjectsCount = 0

        for i in range(len(y)):
            newObject = x[i, :]

            isObjectUnderDiviser = self.ifObjectIsUnder(diviser, newObject)
            objectClass = y[i]

            predictions[i] = classUnderDiviser if isObjectUnderDiviser else oppositeClassUnderDiviser

            if objectClass == classUnderDiviser:
                positiveObjectsCount += 1
                positiveObjects += (1 if isObjectUnderDiviser else 0)
            else:
                negativeObjectsCount += 1
                negativeObjects += (1 if isObjectUnderDiviser else 0)

        accuracy = positiveObjects / positiveObjectsCount - negativeObjects / negativeObjectsCount

        return accuracy, predictions

    def ifObjectIsUnder(self, diviser, object):
        for i in range(len(object)):
            if object[i] > diviser[i]:
                return False

        return True

    def update(self, model, x, y):
        raise AssertionError('KS Learner is not supposed to be updated')

    def testSingleObject(self, model, object):
        diviser = model[0]
        classUnder = model[1]

        isObjectUnderDiviser = self.ifObjectIsUnder(diviser, object)

        if isObjectUnderDiviser:
            return classUnder

        oppositeClass = 1 if classUnder == 0 else 0
        return oppositeClass

    def train(self, x, y, probs):

        if not torch.cuda.is_available():
            #self.logger.logDebug('torch cuda is not available')
            return self.calcPValuesCpuNumba(x, y)

        nFeatures = x.shape[1]
        if nFeatures < 1000:
            return self.calcPValuesCpuNumba(x, y)
        else:
            return self.calcPValueFastCuda(x, y)

    def calcPValuesCpuNumba(self, x, y):
        nClasses, counts = np.unique(y, return_counts=True)
        vt1 = GetValuedTarget(y, nClasses[0], 1 / counts[0], -1 / counts[1])
        vt2 = GetValuedTarget(y, nClasses[1], 1 / counts[1], -1 / counts[0])

        sds1 = getSortedSet(x, vt1)
        sds2 = getSortedSet(x, vt2)

        v, d, c = self.metricCalculator.calculateMetricPro(x, y, vt1, sds1, vt2, sds2)
        return (d, c)

    def calcPValueFastCuda(self, x, y):
        nClasses, counts = np.unique(y, return_counts=True)
        valuedTarget1, boolValuedTarget1 = GetValuedAndBoolTarget(y, nClasses[0], 1 / counts[0], -1 / counts[1])
        valuedTarget2, boolValuedTarget2 = GetValuedAndBoolTarget(y, nClasses[1], 1 / counts[1], -1 / counts[0])

        ss1 = getSortedSet(x, valuedTarget1)
        ss2 = getSortedSet(x, valuedTarget2)

        vt1, bvt1 = GetValuedAndBoolTarget(y, nClasses[0], 1 / counts[0], -1 / counts[1])
        vt2, bvt2 = GetValuedAndBoolTarget(y, nClasses[1], 1 / counts[1], -1 / counts[0])

        ss1_device = cuda.to_device(ss1)
        ss2_device = cuda.to_device(ss2)

        x_device = cuda.to_device(x)

        v, d, c = self.metricCalculator.calculateMetricGpu(x, x_device, y, ss1, ss1_device, vt1,
                                                           bvt1, ss2, ss2_device, vt2, bvt2)

        return (d, c)