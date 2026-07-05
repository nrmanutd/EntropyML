import math

from CodeResearch.ObjectComplexity.InstancePriority.basePriorityCalculator import BasePriorityCalculator


class PredefinedPrioritizer(BasePriorityCalculator):
    def __init__(self, priorities, probs, nIterations):
        self.nIterations = nIterations
        self.probs = probs
        self.priorities = priorities
        self.currentIdx = 0

    def calculatePriority(self, dataSet, target):
        bulkSize = math.ceil(len(self.priorities) / self.nIterations)

        startIdx = self.currentIdx * bulkSize
        finishIdx = (self.currentIdx + 1) * bulkSize

        self.currentIdx += 1
        self.currentIdx = self.currentIdx % self.nIterations

        return self.priorities[startIdx: finishIdx], self.probs[startIdx: finishIdx]

