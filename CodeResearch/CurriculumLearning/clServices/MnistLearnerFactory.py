from CodeResearch.CurriculumLearning.clServices.PyTorchCVCLLearnersFactory import PyTorchCVCLLearnersFactory
from CodeResearch.LearningFramework.Learners.BossModelLearner import BossModelLearner
from CodeResearch.LearningFramework.Learners.ForgettingTorchModelLearner import ForgettingTorchModelLearner
from CodeResearch.LearningFramework.Learners.NNCumulativeEpochLearner import NNCumulativeEpochLearner
from CodeResearch.LearningFramework.Learners.NNTorchModelLearner import TorchModelLearner
from CodeResearch.LearningFramework.NeuralNetwork.PytorchHelpers import MNISTScoringNet, MNISTTargetNet


class MnistLearnerFactory(PyTorchCVCLLearnersFactory):
    def __init__(self, nClasses, targetBatchSize=128, scoringBatchSize=64):
        super().__init__(nClasses, targetBatchSize, scoringBatchSize)

    def createScoreLearner_int(self, epochs):
        mnist_scoring_learner = TorchModelLearner(
            model_factory=lambda: MNISTScoringNet(num_classes=self.nClasses),
            optimizer_name="adam",
            lr=1e-3,
            weight_decay=0.0,
            batch_size=self.scoringBatchSize,
            epochs=epochs,
            update_epochs=epochs,
            scheduler_name="none",
            use_amp=True,
        )
        return mnist_scoring_learner

    def createTargetLearner_int(self, epochs):
        mnist_target_learner = TorchModelLearner(
            model_factory=lambda: MNISTTargetNet(num_classes=self.nClasses),
            optimizer_name="adam",
            lr=1e-3,
            weight_decay=1e-4,
            batch_size=self.targetBatchSize,
            update_epochs=epochs,
            epochs=epochs,
            scheduler_name="none"
        )
        return mnist_target_learner

class MnistForgettingLearnerFactory(PyTorchCVCLLearnersFactory):
    def __init__(self, nClasses, targetBatchSize=128, scoringBatchSize=64):
        super().__init__(nClasses, targetBatchSize, scoringBatchSize)

    def createScoreLearner_int(self, epochs):
        raise NotImplementedError('Method not implemented')

    def createTargetLearner_int(self, epochs):
        mnist_target_learner = ForgettingTorchModelLearner(
            model_factory=lambda: MNISTTargetNet(num_classes=self.nClasses),
            optimizer_name="adam",
            lr=1e-3,
            weight_decay=1e-4,
            batch_size=self.targetBatchSize,
            update_epochs=epochs,
            epochs=epochs,
            scheduler_name="none"
        )
        return mnist_target_learner

class MnistBossLearnerFactory(PyTorchCVCLLearnersFactory):
    def __init__(self, nClasses, targetBatchSize=128, scoringBatchSize=64):
        super().__init__(nClasses, targetBatchSize, scoringBatchSize)

    def createScoreLearner_int(self, epochs):
        raise NotImplementedError('Method not implemented')

    def createTargetLearner_int(self, epochs):
        mnist_target_learner = BossModelLearner(
            model_factory=lambda: MNISTTargetNet(num_classes=self.nClasses),
            optimizer_name="adam",
            lr=1e-3,
            weight_decay=1e-4,
            batch_size=self.targetBatchSize,
            update_epochs=epochs,
            epochs=epochs,
            scheduler_name="none"
        )
        return mnist_target_learner

class MnistCumulativeLearnerFactory(PyTorchCVCLLearnersFactory):
    def __init__(self, nClasses, targetBatchSize=128, scoringBatchSize=64):
        super().__init__(nClasses, targetBatchSize, scoringBatchSize)

    def createScoreLearner_int(self, epochs):
        mnist_scoring_learner = TorchModelLearner(
            model_factory=lambda: MNISTScoringNet(num_classes=self.nClasses),
            optimizer_name="adam",
            lr=1e-3,
            weight_decay=0.0,
            batch_size=self.scoringBatchSize,
            epochs=epochs,
            update_epochs=epochs,
            scheduler_name="none",
            use_amp=True,
        )
        return mnist_scoring_learner

    def createTargetLearner_int(self, epochs):
        mnist_target_learner = NNCumulativeEpochLearner(
            model_factory=lambda: MNISTTargetNet(num_classes=self.nClasses),
            optimizer_name="adam",
            lr=1e-3,
            weight_decay=1e-4,
            batch_size=self.targetBatchSize,
            update_epochs=15,
            epochs=epochs,
            scheduler_name="none"
        )
        return mnist_target_learner
