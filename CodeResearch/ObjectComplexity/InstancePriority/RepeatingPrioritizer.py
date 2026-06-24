from CodeResearch.ObjectComplexity.InstancePriority.basePriorityCalculator import BasePriorityCalculator


class RepeatingPrioritizer(BasePriorityCalculator):
    def __init__(self, nRepeats: int, prioritizer: BasePriorityCalculator):
        self.prioritizer = prioritizer
        self.nRepeats = nRepeats

    def calculatePriority(self, dataSet, target):

        priorities, probs = self.prioritizer.calculatePriority(dataSet, target)

        if not isinstance(priorities, list):
            return [priorities] * self.nRepeats, [probs] * self.nRepeats

        resultPriorities = []
        resultProbs = []

        for i in range(len(priorities)):
            for _ in range(self.nRepeats):
                resultPriorities.append(priorities[i])
                resultProbs.append(probs[i])

        return resultPriorities, resultProbs