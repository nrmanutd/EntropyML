from CodeResearch.ObjectComplexity.Diversity.BaseObjectDiversifier import BaseObjectDiversifier
from CodeResearch.ObjectComplexity.Hardness.BaseHardnessCalculator import BaseHardnessCalculator


class DiversityBasedHardnessCalculator(BaseHardnessCalculator):

    def __init__(self, hc: BaseHardnessCalculator, dcCreator):
        self.dcCreator = dcCreator
        self.hc = hc

    def calculateHardness(self, dataSet, target):
        importance, easiness = self.hc.calculateHardness(dataSet, target)

        importance = self.dcCreator(dataSet, target, easiness)

        return importance, easiness