from CodeResearch.CurriculumLearning.clServices.commonCLHelpers import calculateProductBasedPriority
from CodeResearch.LearningFramework.NeuralNetwork.BaseScoreCalculator import BaseScoreCalculator
from CodeResearch.ObjectComplexity.Diversity.DiversifierHelpers import \
    prediction_correct_and_grad_norm_head_linear_one_pass


class GradNormAndHardnessScoreCalculator(BaseScoreCalculator):
    def calculateScore(self, model, batches, device):
        hScores, gradNormScores = prediction_correct_and_grad_norm_head_linear_one_pass(model, batches, device)
        return calculateProductBasedPriority(hScores, gradNormScores)
