from CodeResearch.LearningFramework.NeuralNetwork.BaseScoreCalculator import BaseScoreCalculator
from CodeResearch.ObjectComplexity.Diversity.DiversifierHelpers import el2n_scores

class EL2NScoreCalculator(BaseScoreCalculator):
    def calculateScore(self, model, batches, device):
        scores = el2n_scores(model, batches, device)
        return scores