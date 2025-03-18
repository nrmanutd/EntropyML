import time
from unittest import TestCase

import numpy as np

from CodeResearch.DiviserCalculation.getDiviserFastCuda import getMaximumDiviserFastCuda
from CodeResearch.DiviserCalculation.getDiviserFastNumba import getMaximumDiviserFastNumba


class TestDiviserCuda(TestCase):
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

        s = np.array([[1.1, 1.2],
                      [1.2, 1.1],
                      [1.3, 1.4],
                      [1.2, 1.3],
                      [5.1, 5.2],
                      [6.1, 6.2],
                      [5.2, 5.1],
                      [6.2, 6.1]])
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

    def template_numba_and_cuda(self, s, c, checkVectors=True):
        t1 = time.time()
        numbaResult = getMaximumDiviserFastNumba(s, c)
        t2 = time.time()
        print('Time of numba: {:}s'.format(t2 - t1))
        self.assertDeltaIndependently(s, c, numbaResult)

        t1 = time.time()
        cudaResult = getMaximumDiviserFastCuda(s, c)
        t2 = time.time()
        print('Time of cuda: {:}s'.format(t2 - t1))
        self.assertDeltaIndependently(s, c, cudaResult)

        print('Cuda result: ', cudaResult[0])
        print('Numba result: ', numbaResult[0])

        print('Cuda vector: ', cudaResult[1].tolist())
        print('Numba vector: ', numbaResult[1].tolist())

        self.assertLessEqual(abs(cudaResult[0] - numbaResult[0]), 0.01)
        if checkVectors:
            self.assertEqual(cudaResult[1].tolist(), numbaResult[1].tolist())

    def template_cuda_random(self, nObjects, nFeatures, checkVectors = True):
        dataSet = np.random.rand(nObjects, nFeatures)
        target = np.zeros(nObjects)
        target[((int)(np.floor(nObjects/2))):nObjects] = 1

        self.template_numba_and_cuda(dataSet, target, checkVectors)

    def test_cuda_random(self):
        for i in range(1, 50):
            print('Attempt # ', i)
            self.template_cuda_random(5000, 3000, False)

    def generate_separated_dataset(self, l, alpha):
        elements = l - l % 2

        dataSet = np.zeros((elements, 3))
        target = np.zeros(elements, dtype=int)
        a = 10
        center = 2 * a * alpha

        for i in np.arange(elements):
            c = 0 if i < elements / 2 else center

            x = np.random.uniform(-a + c, c + a)
            y = np.random.uniform(-a, a)
            z = np.random.uniform(-a, a)

            dataSet[i, 0] = x
            dataSet[i, 1] = y
            dataSet[i, 2] = z

            target[i] = 1 if i < elements / 2 else -1

        return dataSet, target

    def test_get_maximum_diviser_fast_numba_check_good_separation_and_many_objects(self):
        d = self.generate_separated_dataset(2000, 5)
        self.template_numba_and_cuda(d[0], d[1])

    def test_get_maximum_diviser_numba_cuda_case0(self):
        d = self.datasets[0]
        self.template_numba_and_cuda(d[0], d[1])

    def test_get_maximum_diviser_numba_cuda_case1(self):
        d = self.datasets[1]
        self.template_numba_and_cuda(d[0], d[1])

    def test_get_maximum_diviser_numba_cuda_case2(self):
        d = self.datasets[2]
        self.template_numba_and_cuda(d[0], d[1])

    def test_get_maximum_diviser_numba_cuda_case3(self):
        d = self.datasets[3]
        self.template_numba_and_cuda(d[0], d[1])

    def test_get_maximum_diviser_numba_cuda_case4(self):
        d = self.datasets[4]
        self.template_numba_and_cuda(d[0], d[1])

    def test_get_maximum_diviser_numba_cuda_case5(self):
        d = self.datasets[5]
        self.template_numba_and_cuda(d[0], d[1])

    def test_get_maximum_diviser_numba_cuda_case6(self):
        d = self.datasets[6]
        self.template_numba_and_cuda(d[0], d[1])

    def test_get_maximum_diviser_numba_cuda_case7(self):
        d = self.datasets[7]
        self.template_numba_and_cuda(d[0], d[1])
