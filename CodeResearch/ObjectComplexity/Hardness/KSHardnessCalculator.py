import math
import numpy as np

from CodeResearch.DataSeparationFramework.Metrics.KSMetric import KSMetric
from CodeResearch.DataSeparationFramework.pValueCalculator import PValueCalculator
from CodeResearch.ObjectComplexity.Factory.ShapValuesComplexityCalculatorFactory import \
    ShapValuesComplexityCalculatorFactory
from CodeResearch.ObjectComplexity.Hardness.BaseHardnessCalculator import BaseHardnessCalculator


class KSHardnessCalculator(BaseHardnessCalculator):
    def __init__(self, attempts, fraction):

        self.fraction = fraction
        self.pValueCalculator = PValueCalculator(ShapValuesComplexityCalculatorFactory(), KSMetric(), attempts, True,
                                                 False, False)

    def calculateHardness(self, dataSet, target):
        firstClass = 0
        secondClass = 1

        result = self.pValueCalculator.calcPValueFastPro(math.ceil(len(target) * self.fraction), dataSet, target, firstClass, secondClass)

        complexityCalculator = result[2]
        importance, easiness = complexityCalculator.getShapValues()
        importance, easiness = self.reorder(importance, easiness, target, firstClass, secondClass)

        return importance, easiness

    def reorder(self, importance, easiness, target, firstClass, secondClass):
        iObjects = list(np.where(target == firstClass)[0])
        jObjects = list(np.where(target == secondClass)[0])

        allObjects = iObjects + jObjects

        rImportance = np.zeros(len(importance))
        rEasiness = np.zeros(len(easiness))

        for i in range(len(allObjects)):
            rImportance[allObjects[i]] = importance[i]
            rEasiness[allObjects[i]] = easiness[i]

        return rImportance, rEasiness