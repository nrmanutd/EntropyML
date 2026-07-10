import numpy as np
import torch

from CodeResearch.CurriculumLearning.clServices.commonCLHelpers import should_stop
from CodeResearch.Helpers.Logger.BaseLogger import BaseLogger
from CodeResearch.LearningFramework.DataProcessing.BaseDataProcessor import BaseDataProcessor
from CodeResearch.LearningFramework.Learners.DataTransformationParametersLearner import \
    DataTransformationParametersLearner
from CodeResearch.LearningFramework.Learners.TorchLearner import TorchLearner
from CodeResearch.LearningFramework.NeuralNetwork.BaseScoreCalculator import BaseScoreCalculator
from CodeResearch.LearningFramework.Samplers.Batches.randomAllsetSampler import RandomAllsetSampler
from CodeResearch.ObjectComplexity.Diversity.BaseObjectDiversifier import BaseObjectDiversifier
from CodeResearch.ObjectComplexity.InstancePriority.standardPriorityCalculator import StandardPriorityCalculator


class IncrementalObjectDiversifier(BaseObjectDiversifier):
    def __init__(self, learner: TorchLearner, nAttempts: int, batchSize: int, scoreCalculator: BaseScoreCalculator, dataTransformer: BaseDataProcessor, logger: BaseLogger, minimumIterations: int = 2):
        self.scoreCalculator = scoreCalculator
        self.dataTransformer = dataTransformer
        self.minimumIterations = minimumIterations
        self.logger = logger
        self.nAttempts = nAttempts
        self.learner = learner
        self.batchSize = batchSize

    def calculateObjectDiversity(self, ds, t, baseDataSet, baseTarget, alpha):
        device = self.learner.device

        importance = np.zeros(len(t))

        if baseDataSet is None or len(baseTarget) == 0:
            raise ValueError('Incremental hardness calculator shouldnt be used with empty baseDataSet ')

        p = self.dataTransformer.estimateDataTransformationParameters(baseDataSet, baseTarget)
        learner = DataTransformationParametersLearner(self.learner, p, self.dataTransformer)

        ds, t = self.dataTransformer.applyParametersToData(ds, t, p)

        xb = torch.as_tensor(ds, dtype=torch.float32, device=device)
        yb = torch.as_tensor(t, dtype=torch.int64, device=device)

        sampler = RandomAllsetSampler(xb, yb, self.batchSize, StandardPriorityCalculator())
        scoresList = []
        currentCounter = 0

        for i in range(self.nAttempts):
            currentCounter = i
            if i%10 == 0:
                self.logger.logDebug(f'Calculating incremental step for #{i} of {self.nAttempts} attempts')

            self.logger.logDebug('Training learner...')
            model = learner.train(baseDataSet, baseTarget, np.full(len(baseTarget), 1.0 / len(baseTarget)))
            self.logger.logDebug('Learner is trained. Sampling and calculating score...')

            batches = sampler.sample()
            scores = self.scoreCalculator.calculateScore(model, batches, device)
            self.logger.logDebug('Score is calculated.')

            importance += scores
            scoresList.append(np.array(importance))

            if should_stop(scoresList, self.logger) and i >= self.minimumIterations:
                del model
                torch.cuda.empty_cache()

                self.logger.logDebug(f'Stop criteria based on rank correlation at iteration {i} of {self.nAttempts}')
                break

            del model
            torch.cuda.empty_cache()

        importance /= (currentCounter + 1)

        self.logger.logDebug(f'Finished calculating additional diversification for alpha = {alpha}')

        return importance