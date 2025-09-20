import numpy as np
from unittest import TestCase

from scipy.stats import entropy

from CodeResearch.ObjectComplexity.complexityCalculator import KSComplexityCalculator


class TestKSComplexityCalculator(TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        pass

    def test_two_separate_points(self):
        dataSet = np.array([[1, 1], [2, 2]])
        target = np.array([0, 1])
        cc = KSComplexityCalculator(dataSet, target)

        diviser = np.array([1.5, 1.5])
        cc.addComplexity(diviser)
        cc.addComplexity(diviser)
        cc.addComplexity(diviser)

        expectedFrequency = np.array([1, 1])
        actualFrequency = cc.getObjectsFrequences()

        self.assertEqual(expectedFrequency.tolist(), actualFrequency.tolist())

        expectedComplexity = np.array([0, 0])
        actualComplexity = cc.calculateComplexity()

        self.assertEqual(expectedComplexity.tolist(), actualComplexity.tolist())
        pass

    def test_four_separate_points(self):
        dataSet = np.array([[1, 1],[1.4, 1.4], [1.6, 1.6], [2, 2]])
        target = np.array([0, 0, 1, 1])
        cc = KSComplexityCalculator(dataSet, target)

        cc.addComplexity(np.array([1.5, 1.5]))
        cc.addComplexity(np.array([1.5, 1.5]))
        cc.addComplexity(np.array([1.3, 1.3]))
        cc.addComplexity(np.array([1.7, 1.7]))

        expectedFrequency = np.array([1, 0.75, 0.75, 1])
        actualFrequency = cc.getObjectsFrequences()

        self.assertEqual(expectedFrequency.tolist(), actualFrequency.tolist())

        expectedComplexity = np.array([0, entropy([0.75, 0.25], base=2), entropy([0.75, 0.25], base=2), 0])
        actualComplexity = cc.calculateComplexity()

        self.assertEqual(expectedComplexity.tolist(), actualComplexity.tolist())
        pass

    def test_five_separate_points(self):
        dataSet = np.array([[1, 1], [1.4, 1.4], [1.46, 1.45], [1.6, 1.6], [2, 2]])
        target = np.array([0, 0, 0, 1, 1])
        cc = KSComplexityCalculator(dataSet, target)

        cc.addComplexity(np.array([1.5, 1.5]))
        cc.addComplexity(np.array([1.5, 1.5]))
        cc.addComplexity(np.array([1.3, 1.3]))
        cc.addComplexity(np.array([1.7, 1.7]))
        cc.addComplexity(np.array([1.45, 1.7]))

        expectedFrequency = np.array([1, 0.8, 0.6, 0.8, 1])
        actualFrequency = cc.getObjectsFrequences()

        self.assertEqual(expectedFrequency.tolist(), actualFrequency.tolist())

        expectedComplexity = np.array([0, entropy([0.8, 0.2], base=2), entropy([0.6, 0.4], base=2), entropy([0.8, 0.2], base=2), 0])
        actualComplexity = cc.calculateComplexity()

        self.assertEqual(expectedComplexity.tolist(), actualComplexity.tolist())
        pass

    def test_five_separate_points_with_skip_idx(self):
        dataSet = np.array([[1, 1], [1.4, 1.4], [1.46, 1.45], [1.6, 1.6], [2, 2]])
        target = np.array([0, 0, 0, 1, 1])
        cc = KSComplexityCalculator(dataSet, target)

        cc.addComplexityOutOfIdx(np.array([1.5, 1.5]), [0, 3])
        cc.addComplexityOutOfIdx(np.array([1.5, 1.5]), [0, 3])
        cc.addComplexityOutOfIdx(np.array([1.3, 1.3]), [0, 3])
        cc.addComplexityOutOfIdx(np.array([1.7, 1.7]), [0, 3])
        cc.addComplexityOutOfIdx(np.array([1.45, 1.7]), [0, 3])

        expectedFrequency = np.array([np.nan, 0.8, 0.6, np.nan, 1])
        actualFrequency = cc.getObjectsFrequences()

        np.testing.assert_equal(expectedFrequency.tolist(), actualFrequency.tolist())

        expectedComplexity = np.array([np.nan, entropy([0.8, 0.2], base=2), entropy([0.6, 0.4], base=2), np.nan, 0])
        actualComplexity = cc.calculateComplexity()

        np.testing.assert_equal(expectedComplexity.tolist(), actualComplexity.tolist())
        pass