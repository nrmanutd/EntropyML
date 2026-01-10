from CodeResearch.ObjectComplexity.Hardness.BaseHardnessCalculator import BaseHardnessCalculator


class EasinessInvertor(BaseHardnessCalculator):
    def __init__(self, hc: BaseHardnessCalculator):
        self.hc = hc

    def calculateHardness(self, dataSet, target, baseDataSet, baseTarget, alpha):
        i, e = self.hc.calculateHardness(dataSet, target, baseDataSet, baseTarget, alpha)

        return i, 1-e
