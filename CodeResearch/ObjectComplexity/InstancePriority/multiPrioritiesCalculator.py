import math
import numpy as np

from CodeResearch.DataSeparationFramework.Metrics.KSMetric import KSMetric
from CodeResearch.DataSeparationFramework.pValueCalculator import PValueCalculator
from CodeResearch.ObjectComplexity.Factory.ShapValuesComplexityCalculatorFactory import \
    ShapValuesComplexityCalculatorFactory
from CodeResearch.ObjectComplexity.InstancePriority.basePriorityCalculator import BasePriorityCalculator


class MultiPrioritiesCalculator(BasePriorityCalculator):
    def __init__(self, attempts, useBasedPriority, useImportance, useHardness, useBoth):
        self.useBoth = useBoth
        self.useHardness = useHardness
        self.useImportance = useImportance
        self.useBasedPriority = useBasedPriority
        self.pValueCalculator = PValueCalculator(ShapValuesComplexityCalculatorFactory(), KSMetric(), attempts,  True, False, False)

    def calculatePriority(self, dataSet, target):

        print('Calculating values pro...', len(target))
        result = self.pValueCalculator.calcPValueFastPro(math.ceil(len(target)/2), dataSet, target, 0, 1)

        print('Calculated values pro...')

        complexityCalculator = result[2]
        importance, easiness = complexityCalculator.getShapValues()

        importanceIdx = np.argsort(importance)[::-1]
        hardnessIdx = np.argsort(easiness)[::-1]

        minImportance = np.min(importance)
        maxImportance = np.max(importance)

        correctedImportance = (importance - minImportance) / (maxImportance - minImportance)

        bothIdx = np.argsort(easiness * correctedImportance)[::-1]

        resultPriorities = []

        if self.useBasedPriority:
            resultPriorities.append(range(len(target)))

        if self.useImportance:
            resultPriorities.append(importanceIdx)

        if self.useHardness:
            resultPriorities.append(hardnessIdx)

        if self.useBoth:
            resultPriorities.append(bothIdx)

        return resultPriorities