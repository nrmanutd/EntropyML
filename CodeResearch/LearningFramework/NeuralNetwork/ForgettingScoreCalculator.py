from CodeResearch.LearningFramework.NeuralNetwork.BaseScoreCalculator import BaseScoreCalculator


class ForgettingScoreCalculator(BaseScoreCalculator):
    def calculateScore(self, model, batches, device):
        return model[1]