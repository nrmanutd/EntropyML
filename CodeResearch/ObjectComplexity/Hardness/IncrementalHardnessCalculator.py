import numpy as np
import torch

from CodeResearch.CurriculumLearning.clServices.commonCLHelpers import should_stop
from CodeResearch.Helpers.Logger.BaseLogger import BaseLogger
from CodeResearch.Helpers.permutationHelpers import stratified_split_indices_with_min
from CodeResearch.LearningFramework.DataProcessing.BaseDataProcessor import BaseDataProcessor
from CodeResearch.LearningFramework.Learners.DataTransformationParametersLearner import \
    DataTransformationParametersLearner
from CodeResearch.LearningFramework.Learners.TorchLearner import TorchLearner
from CodeResearch.LearningFramework.Samplers.Batches.randomAllsetSampler import RandomAllsetSampler
from CodeResearch.ObjectComplexity.Diversity.DiversifierHelpers import \
    centered_grad_norm_head_linear_two_pass_entropy_loss
from CodeResearch.ObjectComplexity.Hardness.BaseHardnessCalculator import BaseHardnessCalculator
from CodeResearch.ObjectComplexity.InstancePriority.standardPriorityCalculator import StandardPriorityCalculator


class IncrementalHardnessCalculator(BaseHardnessCalculator):
    def __init__(self, learner: TorchLearner, nAttempts: int, batchSize: int, dataTransformer: BaseDataProcessor, logger: BaseLogger):
        self.dataTransformer = dataTransformer
        self.logger = logger
        self.nAttempts = nAttempts
        self.learner = learner
        self.batchSize = batchSize

    def calculateHardness(self, dataSet, target, baseDataSet, baseTarget, alpha):
        importance = np.zeros(len(target))
        easiness = np.zeros(len(target))
        objectsUsed = np.zeros(len(target))

        device = self.learner.device

        ds = torch.as_tensor(dataSet, dtype=torch.float32, device=device)
        t = torch.as_tensor(target, dtype=torch.int64, device=device)

        bds = torch.as_tensor(baseDataSet, dtype=torch.float32, device=device)
        bt = torch.as_tensor(baseTarget, dtype=torch.int64, device=device)

        importancesList = []
        easinessesList = []

        for i in range(self.nAttempts):
            if i % 10 == 0:
                self.logger.logDebug(f'Calculating incremental step for #{i} of {self.nAttempts} attempts')

            shouldStopImportance = should_stop(importancesList, self.logger)
            shouldStopEasiness = should_stop(easinessesList, self.logger)

            self.logger.logDebug(f'Should stop importance: {shouldStopImportance}, should stop easiness: {shouldStopEasiness}')

            if shouldStopEasiness and shouldStopImportance:
                self.logger.logDebug(f'Stop criteria based on rank correlation at iteration {i} of {self.nAttempts} is applied')
                break

            trainIdx, testIdx = stratified_split_indices_with_min(target, alpha)
            objectsUsed[testIdx] += 1

            x = ds[trainIdx]
            y = t[trainIdx]

            xtest = ds[testIdx]
            ytest = t[testIdx]

            #split dataset on test and train at fraction alpha
            extended_x = torch.cat([x, bds]) if baseDataSet is not None and len(baseTarget) != 0 else x
            extended_y = torch.cat([y, bt]) if baseDataSet is not None and len(baseTarget) != 0 else y

            p = self.dataTransformer.estimateDataTransformationParameters(extended_x, extended_y)
            learner = DataTransformationParametersLearner(self.learner, p, self.dataTransformer)
            extended_x, extended_y = self.dataTransformer.applyParametersToData(extended_x, extended_y, p)
            xtest, ytest = self.dataTransformer.applyParametersToData(xtest, ytest, p)

            self.logger.logDebug('Start training...')
            model = learner.train(extended_x, extended_y, None)
            self.logger.logDebug('Learner trained.')

            sampler = RandomAllsetSampler(xtest, ytest, self.batchSize, StandardPriorityCalculator())
            batches = sampler.sample()

            if not shouldStopImportance:
                curImportance, curEasiness = centered_grad_norm_head_linear_two_pass_entropy_loss(model, batches, self.learner.device)

                curImportance = self.convert(curImportance, len(t), testIdx)
                importance += curImportance
                imp = self.normalize(importance, objectsUsed)
                importancesList.append(imp)

                if not shouldStopEasiness:
                    curEasiness = self.convert(curEasiness, len(t), testIdx)
                    easiness += curEasiness
                    eas = self.normalize(easiness, objectsUsed)
                    easinessesList.append(eas)

                del model
                torch.cuda.empty_cache()

                continue

            predictions = self.learner.test(model, xtest, ytest)[1]
            curEasiness = (predictions == ytest.detach().cpu().numpy()).astype(int)

            curEasiness = self.convert(curEasiness, len(t), testIdx)
            easiness += curEasiness

            eas = self.normalize(easiness, objectsUsed)
            easinessesList.append(eas)

            del model
            torch.cuda.empty_cache()

        importance = importancesList[-1]
        easiness = easinessesList[-1]

        self.logger.logDebug(f'Finished calculating additional diversification for alpha = {alpha}')

        return importance, easiness

    def convert(self, vector, n, idx):
        result = np.zeros(n)
        for i in range(len(vector)):
            result[idx[i]] = vector[i]

        return result

    def normalize(self, v1, norm):
        r = np.zeros(len(v1))

        for i in range(len(v1)):
            if norm[i] == 0:
                if v1[i] != 0 :
                    raise ValueError('Incorrect vector values')
                continue

            r[i] = v1[i] / norm[i]

        return r