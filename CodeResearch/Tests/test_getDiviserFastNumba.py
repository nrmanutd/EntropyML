import numpy as np
from unittest import TestCase

from CodeResearch.DiviserCalculation.getDiviserFast import getMaximumDiviserFast
from CodeResearch.DiviserCalculation.getDiviserFastNumba import getMaximumDiviserFastNumba


class TestDiviserFastNumba(TestCase):
    def test_get_maximum_diviser_fast_numba(self):
        s = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [3, 2, -1], [2, 3, -1], [2, 2, 0]])
        c = np.array([1, 1, 1, -1, -1, -1])

        numbaResult = getMaximumDiviserFastNumba(s, c)
        fastResult = getMaximumDiviserFast(s, c)

        self.assertEqual(numbaResult, fastResult)

    def test_get_maximum_diviser_fast_case1(self):
        s = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [3, 2, -1], [2, 3, -1], [2, 2, 0]])
        c = np.array([1, 1, 1, -1, -1, -1])

        fastResult = getMaximumDiviserFast(s, c)[0]
        expected = 1.0

        self.assertEqual(expected, fastResult)

    def test_get_maximum_diviser_fast_case2(self):
        s = np.array([[1, 4], [2, 3], [3, 2], [4, 1]])
        c = np.array([1, 1, -1, -1])

        fastResult = getMaximumDiviserFast(s, c)[0]
        expected = 1.0

        self.assertEqual(expected, fastResult)

    def test_get_maximum_diviser_fast_case3(self):
        s = np.array([[5.1, 3.4, 1.5, 0.2],
                       [4.8, 3., 1.4, 0.3],
                       [4.6, 3.1, 1.5, 0.2],
                       [4.9, 3.1, 1.5, 0.1]])
        c = np.array([-0.5, -0.5, 0.5, 0.5])

        fastResult = getMaximumDiviserFast(s, c)[0]
        expected = 1.0

        self.assertEqual(expected, fastResult)

    def test_get_maximum_diviser_fast_case4(self):
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
        c = [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1]

        fastResult = getMaximumDiviserFast(s, c)[0]
        expected = 0.5

        self.assertEqual(expected, fastResult)

    def test_get_maximum_diviser_fast_case5(self):
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
            c = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]

            fastResult = getMaximumDiviserFast(s, c)[0]
            expected = 0.375

            self.assertEqual(expected, fastResult)

    def test_get_maximum_diviser_fast_case6(self):
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
        c = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]

        fastResult = getMaximumDiviserFast(s, c)[0]
        expected = 0.5833333

        self.assertAlmostEqual(expected, fastResult)
