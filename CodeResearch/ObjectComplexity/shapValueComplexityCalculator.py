from CodeResearch.ObjectComplexity.baseComplexityCalculator import BaseComplexityCalculator


class ShapValueComplexityCalculator(BaseComplexityCalculator):
    def __init__(self, dataSet, target, objectIdx):
        self.objectIdx = objectIdx
        self.target = target
        self.dataSet = dataSet

    def updateComplexity(self, d, idx):
        pass