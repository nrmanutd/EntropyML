from CodeResearch.LearningFramework.Samplers.RandomWithFixedLengthSampler import RandomWithFixedLengthSampler
from CodeResearch.ObjectComplexity.InstancePriority.shapBasedPriorityCalculator import ShapBasedPriorityCalculator
from CodeResearch.ObjectComplexity.InstancePriority.standardPriorityCalculator import StandardPriorityCalculator


def calculateLosses(x, y, alpha, testAlpha, nAttempts, useImportance, useHardness, generalLearner, xgbLearner):
    if not useImportance and not useHardness:
        prioritizer = StandardPriorityCalculator()
    else:
        prioritizer = ShapBasedPriorityCalculator(nAttempts, useImportance, useHardness)

    sampler = RandomWithFixedLengthSampler(x, y, prioritizer, alpha, testAlpha)

    return generalLearner.estimateLearner(sampler, xgbLearner)