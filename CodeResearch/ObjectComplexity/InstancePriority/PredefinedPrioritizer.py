from CodeResearch.ObjectComplexity.InstancePriority.basePriorityCalculator import BasePriorityCalculator


class PredefinedPrioritizer(BasePriorityCalculator):
    def __init__(self, priorities, probs):
        self.probs = probs
        self.priorities = priorities

    def calculatePriority(self, dataSet, target):
        return self.priorities, self.probs

