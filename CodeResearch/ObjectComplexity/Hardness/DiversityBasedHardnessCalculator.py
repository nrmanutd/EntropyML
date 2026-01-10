import numpy as np
from typing import Callable

from CodeResearch.ObjectComplexity.Diversity.BaseObjectDiversifier import BaseObjectDiversifier
from CodeResearch.ObjectComplexity.Hardness.BaseHardnessCalculator import BaseHardnessCalculator


class DiversityBasedHardnessCalculator(BaseHardnessCalculator):

    def __init__(self, hc: BaseHardnessCalculator, dcCreator: Callable[..., BaseObjectDiversifier]):
        self.dcCreator = dcCreator
        self.hc = hc

    def calculateHardness(self, dataSet, target, baseDataSet, baseTarget, alpha):
        importance, easiness = self.hc.calculateHardness(dataSet, target, baseDataSet, baseTarget, alpha)

        dc = self.dcCreator(easiness)
        importance = dc.calculateObjectDiversity(dataSet, target, baseDataSet, baseTarget, alpha)

        return importance, easiness
