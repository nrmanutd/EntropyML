from CodeResearch.LearningFramework.NeuralNetwork.BaseScoreCalculator import BaseScoreCalculator
from CodeResearch.ObjectComplexity.Diversity.DiversifierHelpers import \
    chg_shapley_to_train_mean_grad_head_linear_two_pass


class CHGScoreCalculator(BaseScoreCalculator):
    def calculateScore(self, model, batches, device):
        return self.calculateScoreDifferentBatches(model, batches, batches, device)

    def calculateScoreDifferentBatches(self, model, trainBatches, testBatches, device):
        scores = chg_shapley_to_train_mean_grad_head_linear_two_pass(model, trainBatches, testBatches, device)
        return scores