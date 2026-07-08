from CodeResearch.LearningFramework.NeuralNetwork.BaseScoreCalculator import BaseScoreCalculator
from CodeResearch.ObjectComplexity.Diversity.DiversifierHelpers import centered_grad_norm_head_linear_two_pass


class GradNormCustomScoreCalculator(BaseScoreCalculator):
    def calculateScore(self, model, batches, device):
        scores = centered_grad_norm_head_linear_two_pass(model, batches, device)
        return scores