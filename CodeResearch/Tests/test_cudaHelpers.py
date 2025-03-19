from unittest import TestCase
import numpy as np
from numpy.testing import assert_array_equal

from CodeResearch.Cuda.cudaHelpers import getSortedSetCuda, updateSortedSetNumba, sort_matrix, \
    updateSortedSetByBucketNumba
from CodeResearch.DiviserCalculation.diviserHelpers import getSortedSet


class Test(TestCase):
    def assertIsSorted(self, dataSet, sortedIdx, target):
        self.assertEqual(dataSet.shape[0], sortedIdx.shape[0])
        self.assertEqual(dataSet.shape[1], sortedIdx.shape[1])

        for iColumn in range(dataSet.shape[1]):
            for iRow in range(1, dataSet.shape[0]):
                self.assertGreaterEqual(dataSet[sortedIdx[iRow, iColumn], iColumn], dataSet[sortedIdx[iRow - 1, iColumn], iColumn])
                if dataSet[sortedIdx[iRow, iColumn], iColumn] == dataSet[sortedIdx[iRow - 1, iColumn], iColumn]:
                    self.assertGreaterEqual(target[sortedIdx[iRow - 1, iColumn]], target[sortedIdx[iRow, iColumn]])

    def get_sorted_set_cuda_template(self, nObjects, nFeatures, randInt=True):
        dataSet = np.random.randint(0, 15, size=(nObjects, nFeatures)).astype(np.float32) if randInt else np.random.rand(nObjects, nFeatures).astype(np.float32)
        target = np.zeros(nObjects)
        target[((int)(np.floor(nObjects / 2))):nObjects] = 1

        sortedIdx = getSortedSetCuda(dataSet, target)
        self.assertIsSorted(dataSet, sortedIdx, target)

    def test_get_sorted_cuda(self):
        for i in range(50):
            print('Test number ', i)
            self.get_sorted_set_cuda_template(2000, 200)
            self.get_sorted_set_cuda_template(2000, 200, False)

    def get_sorted_set_cuda_biton_template(self, nObjects, nFeatures, randInt=True):
        dataSet = np.random.randint(0, 15, size=(nObjects, nFeatures)).astype(np.float32) if randInt else np.random.rand(nObjects, nFeatures).astype(np.float32)
        target = np.zeros(nObjects)
        target[((int)(np.floor(nObjects / 2))):nObjects] = 1

        sortedIdx = sort_matrix(dataSet, target)
        self.assertIsSorted(dataSet, sortedIdx, target)

    def test_get_sorted_cuda_biton(self):
        for i in range(50):
            print('Test number ', i)
            self.get_sorted_set_cuda_biton_template(2000, 200)
            self.get_sorted_set_cuda_biton_template(2000, 200, False)

    def test_get_sorted_set_numba_template(self, nObjects, nFeatures, randInt=True):
        dataSet = np.random.randint(0, 15, size=(nObjects, nFeatures)).astype(np.float32) if randInt else np.random.rand(nObjects, nFeatures).astype(np.float32)
        target = np.zeros(nObjects)
        target[((int)(np.floor(nObjects / 2))):nObjects] = 1

        sortedIdx = getSortedSet(dataSet, target)
        self.assertIsSorted(dataSet, sortedIdx, target)

    def test_get_sorted_numba(self):
        for i in range(50):
            self.test_get_sorted_set_numba_template(2000, 200)
            self.test_get_sorted_set_numba_template(2000, 200, False)

    def test_get_updated_numba(self):
        matrix = np.array([[0, 3], [1, 2], [2, 1], [3, 0], [4, 4], [5, 5]])

        indexes = np.array([1, 3, 5])
        expected = np.array([[0, 1], [1, 0], [2, 2]])

        result = updateSortedSetNumba(matrix, indexes)
        print(result)
        print(expected)
        assert_array_equal(result, expected)

    def test_get_updated_numba_bucket(self):
        matrix = np.array([[0, 3], [1, 2], [2, 1], [3, 0], [4, 4], [5, 5]])

        indexes = np.array([1, 3, 5])
        expected = np.array([[0, 1], [1, 0], [2, 2]])

        result = updateSortedSetByBucketNumba(matrix, indexes)
        print(result)
        print(expected)
        assert_array_equal(result, expected)
