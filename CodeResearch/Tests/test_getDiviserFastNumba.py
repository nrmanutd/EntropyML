import time

import numpy as np
from unittest import TestCase

from CodeResearch.DiviserCalculation.getDiviserFast import getMaximumDiviserFast
from CodeResearch.DiviserCalculation.getDiviserFastNumba import getMaximumDiviserFastNumba


class TestDiviserFastNumba(TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.datasets = []
        s = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [3, 2, -1], [2, 3, -1], [2, 2, 0]])
        c = np.array([1, 1, 1, -1, -1, -1])

        self.datasets.append([s, c])

        s = np.array([[1, 4], [2, 3], [3, 2], [4, 1]])
        c = np.array([1, 1, -1, -1])

        self.datasets.append([s, c])

        s = np.array([[5.1, 3.4, 1.5, 0.2],
                      [4.8, 3., 1.4, 0.3],
                      [4.6, 3.1, 1.5, 0.2],
                      [4.9, 3.1, 1.5, 0.1]])
        c = np.array([-0.5, -0.5, 0.5, 0.5])

        self.datasets.append([s, c])

        s = np.array([[5., 3., 1.6, 0.2],
                      [5.2, 3.4, 1.4, 0.2],
                      [5.1, 3.4, 1.5, 0.2],
                      [5., 3.5, 1.3, 0.3],
                      [4.4, 3.2, 1.3, 0.2],
                      [5., 3.5, 1.6, 0.6],
                      [4.8, 3., 1.4, 0.3],
                      [5., 3.3, 1.4, 0.2],
                      [5.1, 3.5, 1.4, 0.2],
                      [4.9, 3., 1.4, 0.2],
                      [4.6, 3.1, 1.5, 0.2],
                      [4.4, 2.9, 1.4, 0.2],
                      [4.3, 3., 1.1, 0.1],
                      [5.1, 3.5, 1.4, 0.3],
                      [5.1, 3.8, 1.5, 0.3],
                      [5.1, 3.7, 1.5, 0.4]])
        c = np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1])
        self.datasets.append([s, c])

        s = np.array([
            [5., 3.4, 1.6, 0.4]
            , [5.2, 3.5, 1.5, 0.2]
            , [4.7, 3.2, 1.6, 0.2]
            , [5.4, 3.4, 1.5, 0.4]
            , [5.5, 4.2, 1.4, 0.2]
            , [4.9, 3.1, 1.5, 0.1]
            , [5.5, 3.5, 1.3, 0.2]
            , [4.9, 3.1, 1.5, 0.1]
            , [5.1, 3.4, 1.5, 0.2]
            , [5., 3.5, 1.3, 0.3]
            , [4.5, 2.3, 1.3, 0.3]
            , [5., 3.5, 1.6, 0.6]
            , [5.1, 3.8, 1.9, 0.4]
            , [5.1, 3.8, 1.6, 0.2]
            , [4.6, 3.2, 1.4, 0.2]
            , [5.3, 3.7, 1.5, 0.2]
            , [5.1, 3.5, 1.4, 0.2]
            , [4.9, 3., 1.4, 0.2]
            , [5., 3.6, 1.4, 0.2]
            , [5.4, 3.9, 1.7, 0.4]
            , [4.6, 3.4, 1.4, 0.3]
            , [5., 3.4, 1.5, 0.2]
            , [4.4, 2.9, 1.4, 0.2]
            , [4.9, 3.1, 1.5, 0.1]
            , [4.8, 3.4, 1.6, 0.2]
            , [4.8, 3., 1.4, 0.1]
            , [4.3, 3., 1.1, 0.1]
            , [5.7, 3.8, 1.7, 0.3]
            , [5.1, 3.7, 1.5, 0.4]
            , [4.6, 3.6, 1., 0.2]
            , [5.1, 3.3, 1.7, 0.5]
            , [4.8, 3.4, 1.9, 0.2]])
        c = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
        self.datasets.append([s, c])

        s = np.array([[5., 3.4, 1.6, 0.4]
                         , [5.2, 3.5, 1.5, 0.2]
                         , [5.2, 3.4, 1.4, 0.2]
                         , [4.7, 3.2, 1.6, 0.2]
                         , [5.5, 4.2, 1.4, 0.2]
                         , [5., 3.2, 1.2, 0.2]
                         , [5.5, 3.5, 1.3, 0.2]
                         , [4.9, 3.1, 1.5, 0.1]
                         , [5., 3.5, 1.6, 0.6]
                         , [5.1, 3.8, 1.6, 0.2]
                         , [4.6, 3.2, 1.4, 0.2]
                         , [5.3, 3.7, 1.5, 0.2]
                         , [4.9, 3., 1.4, 0.2]
                         , [4.7, 3.2, 1.3, 0.2]
                         , [5., 3.6, 1.4, 0.2]
                         , [4.6, 3.4, 1.4, 0.3]
                         , [5., 3.4, 1.5, 0.2]
                         , [4.9, 3.1, 1.5, 0.1]
                         , [4.8, 3.4, 1.6, 0.2]
                         , [4.8, 3., 1.4, 0.1]
                         , [4.3, 3., 1.1, 0.1]
                         , [5.7, 4.4, 1.5, 0.4]
                         , [5.1, 3.5, 1.4, 0.3]
                         , [5.1, 3.7, 1.5, 0.4]])
        c = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
        self.datasets.append([s, c])

        s = np.array([[1.5, 2.9, 3.9],
                      [1.4, 2.4, 3.0],
                      [1.2, 2.2, 3.2],
                      [1.9, 2.5, 3.9],
                      [1.3, 2.3, 3.8],
                      [1.1, 1.1, 3.7],
                      [1.8, 2.8, 3.1],
                      [1.0, 1.0, 3.8]])
        c = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        self.datasets.append([s, c])

    def assertDeltaIndependently(self, dataSet, c, numbaResult):
        point = numbaResult[1]

        nObjects = len(c)
        allIdx = np.zeros(nObjects, dtype=int)
        nFeatures = dataSet.shape[1]

        for iObject in range(nObjects):
            objectOut = False
            for iFeature in range(nFeatures):
                if dataSet[iObject, iFeature] > point[iFeature]:
                    objectOut = True
                    break

            if not objectOut:
                allIdx[iObject] = 1

        cc, counts = np.unique(c, return_counts=True)
        c1 = 1 / counts[0]
        c2 = -1 / counts[1]

        curSum = 0
        for tIdx in range(nObjects):
            curSum += allIdx[tIdx] * (c1 if c[tIdx] == cc[0] else c2)

        self.assertAlmostEqual(abs(curSum), numbaResult[0])

    def template_numba_and_fast(self, s, c, shouldCheckVectors = True):
        t1 = time.time()
        numbaResult = getMaximumDiviserFastNumba(s, c)
        t2 = time.time()
        print('Time of fast numba: {:}s'.format(t2 - t1))
        self.assertDeltaIndependently(s, c, numbaResult)

        t1 = time.time()
        fastResult = getMaximumDiviserFast(s, c)
        t2 = time.time()
        print('Time of fast: {:}s'.format(t2 - t1))

        print('Fast result: ', fastResult[0])
        print('Numba result: ', numbaResult[0])

        self.assertGreaterEqual(numbaResult[0], fastResult[0])
        if shouldCheckVectors:
            self.assertEqual(fastResult[1].tolist(), numbaResult[1].tolist())

    def template_numba_random(self, nObjects, nFeatures):
        dataSet = np.random.rand(nObjects, nFeatures)
        target = np.zeros(nObjects)
        target[((int)(np.floor(nObjects/2))):nObjects] = 1

        self.template_numba_and_fast(dataSet, target, False)

    def test_numba_random(self):
        for i in range(1, 20):
            print('Attempt # ', i)
            self.template_numba_random(500, 10)

    def template_numba_result_correct_random(self, nObjects, nFeatures):
        dataSet = np.random.rand(nObjects, nFeatures)
        target = np.zeros(nObjects)
        target[((int)(np.floor(nObjects/2))):nObjects] = 1

        numbaResult = getMaximumDiviserFastNumba(dataSet, target)
        self.assertDeltaIndependently(dataSet, target, numbaResult)

    def test_numba_result_correct_random(self):
        for i in range(1, 100):
            print('Attempt #: ', i)
            self.template_numba_result_correct_random(5000, 40)

    def template_fast_result_correct_random(self, nObjects, nFeatures):
        dataSet = np.random.rand(nObjects, nFeatures)
        target = np.zeros(nObjects)
        target[((int)(np.floor(nObjects/2))):nObjects] = 1

        numbaResult = getMaximumDiviserFast(dataSet, target)
        self.assertDeltaIndependently(dataSet, target, numbaResult)

    def test_fast_result_correct_random(self):
        for i in range(1, 100):
            self.template_fast_result_correct_random(500, 4)

    def template_fast(self, d, expected):
        fastResult = getMaximumDiviserFast(d[0], d[1])
        self.assertDeltaIndependently(d[0], d[1], fastResult)

        self.assertAlmostEqual(expected, fastResult[0])

    def test_get_maximum_diviser_fast_numba_case0(self):

        d = self.datasets[0]
        self.template_numba_and_fast(d[0], d[1])

    def test_get_maximum_diviser_fast_numba_case1(self):

        d = self.datasets[1]
        self.template_numba_and_fast(d[0], d[1])

    def test_get_maximum_diviser_fast_numba_case2(self):

        d = self.datasets[2]
        self.template_numba_and_fast(d[0], d[1])

    def test_get_maximum_diviser_fast_numba_case3(self):

        d = self.datasets[3]
        self.template_numba_and_fast(d[0], d[1])

    def test_get_maximum_diviser_fast_numba_case4(self):

        d = self.datasets[4]
        self.template_numba_and_fast(d[0], d[1])

    def test_get_maximum_diviser_fast_numba_case5(self):

        d = self.datasets[5]
        self.template_numba_and_fast(d[0], d[1])

    def test_get_maximum_diviser_fast_numba_case6(self):
        d = self.datasets[6]
        self.template_numba_and_fast(d[0], d[1])

    def test_get_maximum_diviser_fast_case0(self):
        d = self.datasets[0]
        self.template_fast(d, 1.0)

    def test_get_maximum_diviser_fast_case1(self):
        d = self.datasets[1]
        self.template_fast(d, 1.0)

    def test_get_maximum_diviser_fast_case2(self):
        d = self.datasets[2]
        self.template_fast(d, 1.0)

    def test_get_maximum_diviser_fast_case3(self):
        d = self.datasets[3]
        self.template_fast(d, 0.5)

    def test_get_maximum_diviser_fast_case4(self):
        d = self.datasets[4]
        self.template_fast(d, 0.375)

    def test_get_maximum_diviser_fast_case5(self):
        d = self.datasets[5]
        self.template_fast(d, 0.5833333)

    def test_get_maximum_diviser_fast_case6(self):
        d = self.datasets[6]
        self.template_fast(d, 0.5)
