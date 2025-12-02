from CodeResearch.LearningFramework.Samplers.SamplersFactories.baseSamplersFactory import BaseSamplersFactory
from CodeResearch.LearningFramework.Samplers.Batches.randomAllsetSampler import RandomAllsetSampler
from CodeResearch.ObjectComplexity.InstancePriority.PrioritizerType import PrioritizerType
from CodeResearch.ObjectComplexity.InstancePriority.RandomComplexityBasedPrioritizer import \
    RandomComplexityBasedPrioritizer
from CodeResearch.ObjectComplexity.InstancePriority.randomPrioritizer import RandomPrioritizer


class RandomAllsetSamplerFactory(BaseSamplersFactory):

    def __init__(self, batchsize, prioritizerType: PrioritizerType):
        self.batchsize = batchsize
        self.prioritizerType = prioritizerType

    def createSampler(self, x, y, probs):

        if self.prioritizerType == PrioritizerType.Random:
            prioritizer = RandomPrioritizer()
            return RandomAllsetSampler(x, y, self.batchsize, prioritizer)

        if self.prioritizerType == PrioritizerType.Probability:
            prioritizer = RandomComplexityBasedPrioritizer(probs)
            return RandomAllsetSampler(x, y, self.batchsize, prioritizer)

        raise ValueError('Unknown type of prioritizer: ', self.prioritizerType)
