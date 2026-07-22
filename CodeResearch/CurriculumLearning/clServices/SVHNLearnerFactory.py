from CodeResearch.CurriculumLearning.clServices.PyTorchCVCLLearnersFactory import PyTorchCVCLLearnersFactory
from CodeResearch.LearningFramework.Learners.BossModelLearner import BossModelLearner
from CodeResearch.LearningFramework.Learners.ForgettingTorchModelLearner import ForgettingTorchModelLearner
from CodeResearch.LearningFramework.Learners.NNTorchModelLearner import TorchModelLearner
from CodeResearch.LearningFramework.NeuralNetwork.PytorchHelpers import ResNet18CIFAR, CifarResNet6n2


class SVHNLearnerFactory(PyTorchCVCLLearnersFactory):
    def __init__(self, nClasses, targetBatchSize=128, scoringBatchSize=64):
        super().__init__(nClasses, targetBatchSize, scoringBatchSize)

    def createScoreLearner_int(self, epochs):
        cifar10_scoring_learner = TorchModelLearner(
            model_factory=lambda: CifarResNet6n2(num_classes=self.nClasses, n=3, width_mult=0.5),

            optimizer_name="sgd",
            lr=0.02,  # мягче для scoring
            weight_decay=5e-4,
            momentum=0.9,
            nesterov=True,

            batch_size=self.scoringBatchSize,
            update_epochs=epochs,
            epochs=epochs,

            scheduler_name="cosine"  # важно: раз epochs=1 и эпохи снаружи — scheduler тут бессмысленен
        )
        return cifar10_scoring_learner

    def createTargetLearner_int(self, parameters):
        cifar10_target_learner = TorchModelLearner(
            model_factory=lambda: ResNet18CIFAR(num_classes=self.nClasses, width_mult=1.0),

            optimizer_name="sgd",
            lr=0.1,  # классика CIFAR при bs~128
            weight_decay=5e-4,
            momentum=0.9,
            nesterov=True,

            batch_size=self.targetBatchSize,
            update_epochs=parameters,
            epochs=parameters,

            scheduler_name="cosine"  # см. комментарий выше
        )
        return cifar10_target_learner

class SVHNForgettingLearnerFactory(PyTorchCVCLLearnersFactory):
    def __init__(self, nClasses, targetBatchSize=128, scoringBatchSize=64):
        super().__init__(nClasses, targetBatchSize, scoringBatchSize)

    def createScoreLearner_int(self, epochs):
        raise NotImplementedError('Method not implemented')

    def createTargetLearner_int(self, parameters):
        cifar10_target_learner = ForgettingTorchModelLearner(
            model_factory=lambda: ResNet18CIFAR(num_classes=self.nClasses, width_mult=1.0),

            optimizer_name="sgd",
            lr=0.1,  # классика CIFAR при bs~128
            weight_decay=5e-4,
            momentum=0.9,
            nesterov=True,

            batch_size=self.targetBatchSize,
            update_epochs=parameters,
            epochs=parameters,

            scheduler_name="cosine"  # см. комментарий выше
        )
        return cifar10_target_learner

class SVHNBossLearnerFactory(PyTorchCVCLLearnersFactory):
    def __init__(self, nClasses, targetBatchSize=128, scoringBatchSize=64):
        super().__init__(nClasses, targetBatchSize, scoringBatchSize)

    def createScoreLearner_int(self, epochs):
        raise NotImplementedError('Method not implemented')

    def createTargetLearner_int(self, parameters):
        cifar10_target_learner = BossModelLearner(
            model_factory=lambda: ResNet18CIFAR(num_classes=self.nClasses, width_mult=1.0),

            optimizer_name="sgd",
            lr=0.1,  # классика CIFAR при bs~128
            weight_decay=5e-4,
            momentum=0.9,
            nesterov=True,

            batch_size=self.targetBatchSize,
            update_epochs=parameters,
            epochs=parameters,

            scheduler_name="cosine"  # см. комментарий выше
        )
        return cifar10_target_learner