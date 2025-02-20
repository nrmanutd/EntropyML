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


    def template_numba_and_fast(self, s, c):
        numbaResult = getMaximumDiviserFastNumba(s, c)
        fastResult = getMaximumDiviserFast(s, c)

        self.assertAlmostEqual(fastResult[0], numbaResult[0])
        self.assertEqual(fastResult[1].tolist(), numbaResult[1].tolist())

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

    def test_get_maximum_diviser_fast_case1(self):
        d = self.datasets[0]

        fastResult = getMaximumDiviserFast(d[0], d[1])[0]
        expected = 1.0

        self.assertEqual(expected, fastResult)

    def test_get_maximum_diviser_fast_case2(self):
        d = self.datasets[1]

        fastResult = getMaximumDiviserFast(d[0], d[1])[0]
        expected = 1.0

        self.assertEqual(expected, fastResult)

    def test_get_maximum_diviser_fast_case3(self):
        d = self.datasets[2]

        fastResult = getMaximumDiviserFast(d[0], d[1])[0]
        expected = 1.0

        self.assertEqual(expected, fastResult)

    def test_get_maximum_diviser_fast_case4(self):
        d = self.datasets[3]
        fastResult = getMaximumDiviserFast(d[0], d[1])[0]

        expected = 0.5

        self.assertEqual(expected, fastResult)

    def test_get_maximum_diviser_fast_case5(self):
        d = self.datasets[4]

        fastResult = getMaximumDiviserFast(d[0], d[1])[0]
        expected = 0.375

        self.assertEqual(expected, fastResult)

    def test_get_maximum_diviser_fast_case6(self):
        d = self.datasets[5]

        fastResult = getMaximumDiviserFast(d[0], d[1])[0]
        expected = 0.5833333

        self.assertAlmostEqual(expected, fastResult)
