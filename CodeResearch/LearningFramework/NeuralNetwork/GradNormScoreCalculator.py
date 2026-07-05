from CodeResearch.LearningFramework.NeuralNetwork.BaseScoreCalculator import BaseScoreCalculator
from CodeResearch.ObjectComplexity.Diversity.DiversifierHelpers import grad_norm_head_linear_one_pass


class GradNormScoreCalculator(BaseScoreCalculator):
    def calculateScore(self, model, batches, device):
        scores = grad_norm_head_linear_one_pass(model, batches, device)
        return scores