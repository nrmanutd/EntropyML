from CodeResearch.LearningFramework.NeuralNetwork.BaseScoreCalculator import BaseScoreCalculator
from CodeResearch.ObjectComplexity.Diversity.DiversifierHelpers import cosine_to_mean_grad_head_linear_two_pass


class CosDistanceScoreCalculator(BaseScoreCalculator):
    def calculateScore(self, model, batches, device):
        scores = cosine_to_mean_grad_head_linear_two_pass(model, batches, device)
        return scores