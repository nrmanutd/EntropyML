import math

import numpy as np
from sklearn.preprocessing import LabelEncoder

from CodeResearch.LearningFramework.Samplers.RandomWithFixedLengthSampler import RandomWithFixedLengthSampler
from CodeResearch.ObjectComplexity.InstancePriority.multiPrioritiesCalculator import MultiPrioritiesCalculator


def calculateLosses(x, y, alpha, testAlpha, nAttempts, generalLearner, learner):
    prioritizer = MultiPrioritiesCalculator(nAttempts, True, True, True, True)

    sampler = RandomWithFixedLengthSampler(x, y, prioritizer, alpha, testAlpha)

    result =  generalLearner.estimateLearner(sampler, learner)

    arr = np.array(result)
    res = arr.T

    return res

def filterDataSet(x, y, alpha, firstClass, secondClass):
    enc = LabelEncoder()
    target = enc.fit_transform(np.ravel(y))

    firstObjects = np.where(target == firstClass)[0]
    secondObjects = np.where(target == secondClass)[0]

    idx = list(set(firstObjects) | set(secondObjects))
    firstK = math.ceil(alpha * len(idx))
    idx = idx[:firstK]

    tt = enc.fit_transform(np.ravel(target[idx]))

    return x[idx, :], tt