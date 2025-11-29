from CodeResearch.ObjectComplexity.InstancePriority.basePriorityCalculator import BasePriorityCalculator


class StandardPriorityCalculator(BasePriorityCalculator):
    def calculatePriority(self, dataSet, target):
        return [range(len(target))]