import numpy as np
from typing import Callable

from CodeResearch.ObjectComplexity.Diversity.BaseObjectDiversifier import BaseObjectDiversifier
from CodeResearch.ObjectComplexity.Hardness.BaseHardnessCalculator import BaseHardnessCalculator


class CombinedDiversityBasedHardnessCalculator(BaseHardnessCalculator):
    def __init__(self, hc: BaseHardnessCalculator, dcCreator: Callable[..., BaseObjectDiversifier]):
        self.dcCreator = dcCreator
        self.hc = hc

    def calculateHardness(self, dataSet, target, baseDataSet, baseTarget, alpha):
        if baseDataSet is None or len(baseTarget) == 0:
            importance, easiness = self.hc.calculateHardness(dataSet, target, baseDataSet, baseTarget, alpha)
            return importance, easiness

        easiness = np.zeros(len(target))
        dc = self.dcCreator(easiness)
        importance = dc.calculateObjectDiversity(dataSet, target, baseDataSet, baseTarget, alpha)

        return importance, easiness