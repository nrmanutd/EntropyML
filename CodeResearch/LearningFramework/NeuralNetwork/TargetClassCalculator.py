from CodeResearch.LearningFramework.NeuralNetwork.BaseScoreCalculator import BaseScoreCalculator
from CodeResearch.ObjectComplexity.Diversity.DiversifierHelpers import prediction_correct_head_linear_one_pass


class TargetClassCalculator(BaseScoreCalculator):
    def __init__(self, easinessBetter: bool):
        self.easinessBetter = easinessBetter

    def calculateScore(self, model, batches, device):
        scores = prediction_correct_head_linear_one_pass(model, batches, device)
        if self.easinessBetter:
            return scores

        return 1 - scores
