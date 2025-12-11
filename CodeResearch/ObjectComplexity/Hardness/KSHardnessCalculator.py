import math

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
        result = self.pValueCalculator.calcPValueFastPro(math.ceil(len(target) * self.fraction), dataSet, target, 0, 1)

        complexityCalculator = result[2]
        importance, easiness = complexityCalculator.getShapValues()

        return importance, easiness