from CodeResearch.LearningFramework.Samplers.SamplersFactories.baseSamplersFactory import BaseSamplersFactory
from CodeResearch.LearningFramework.Samplers.Batches.sequentialSampler import SequentialSampler


class SequentialSamplerFactory(BaseSamplersFactory):
    def __init__(self, batchsize):
        self.batchsize = batchsize

    def createSampler(self, x, y, probs):
        return SequentialSampler(x, y, self.batchsize)