import math

import numpy as np

from CodeResearch.DataSeparationFramework.Metrics.KSMetric import KSMetric
from CodeResearch.DataSeparationFramework.pValueCalculator import PValueCalculator
from CodeResearch.ObjectComplexity.Factory.ShapValuesComplexityCalculatorFactory import \
    ShapValuesComplexityCalculatorFactory
from CodeResearch.ObjectComplexity.InstancePriority.basePriorityCalculator import BasePriorityCalculator


class ShapBasedPriorityCalculator(BasePriorityCalculator):

    def __init__(self, attempts, useImportance, useHardness):
        self.useHardness = useHardness
        self.useImportance = useImportance
        self.pValueCalculator = PValueCalculator(ShapValuesComplexityCalculatorFactory(), KSMetric(), attempts,  True, False, False)

        if not useImportance and not useHardness:
            raise ValueError('At least one of importance or hardness should be set')

    def calculatePriority(self, dataSet, target):

        result = self.pValueCalculator.calcPValueFastPro(math.ceil(len(target)/2), dataSet, target, 0, 1)

        complexityCalculator = result[2]
        importance, easiness = complexityCalculator.getShapValues()

        if self.useImportance and not self.useHardness:
            idx = np.argsort(importance)[::-1]
            return [idx], [importance[idx]]

        if not self.useImportance and self.useHardness:
            hardness = 1 - easiness
            idx = np.argsort(hardness)
            return [idx], [hardness[idx]]

        minImportance = np.min(importance)
        maxImportance = np.max(importance)

        correctedImportance = (importance - minImportance) / (maxImportance - minImportance)
        values = easiness * correctedImportance
        idx = np.argsort(values)[::-1]

        return [idx], [values[idx]]
