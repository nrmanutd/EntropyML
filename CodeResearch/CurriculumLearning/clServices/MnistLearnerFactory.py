from CodeResearch.CurriculumLearning.clServices.BaseCLLearnersFactory import BaseCLLearnersFactory
from CodeResearch.CurriculumLearning.clServices.clHelpers import createMnistTarget, createMnistScoring
from CodeResearch.LearningFramework.Learners.NNTorchModelLearner import TorchModelLearner
from CodeResearch.LearningFramework.NeuralNetwork.PytorchHelpers import MNISTScoringNet, MNISTTargetNet


class MnistLearnerFactory(BaseCLLearnersFactory):
    def __init__(self, nClasses, targetBatchSize, scoringBatchSize):
        super().__init__(nClasses, targetBatchSize, scoringBatchSize)

    def createScoreLearner(self, epochs):
        mnist_scoring_learner = TorchModelLearner(
            model_factory=lambda: MNISTScoringNet(num_classes=self.nClasses),
            optimizer_name="adam",
            lr=1e-3,
            weight_decay=0.0,
            batch_size=self.scoringBatchSize,
            epochs=epochs,
            scheduler_name="none",
            use_amp=True,
        )
        return mnist_scoring_learner

    def createTargetLearner(self, parameters):
        mnist_target_learner = TorchModelLearner(
            model_factory=lambda: MNISTTargetNet(num_classes=self.nClasses),
            optimizer_name="adam",
            lr=1e-3,
            weight_decay=1e-4,
            batch_size=self.targetBatchSize,
            update_epochs=1,
            epochs=1,
            scheduler_name="none"
        )
        return mnist_target_learner
