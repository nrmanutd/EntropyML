from CodeResearch.CurriculumLearning.clServices.BaseCLLearnersFactory import BaseCLLearnersFactory
from CodeResearch.CurriculumLearning.clServices.PyTorchCVCLLearnersFactory import PyTorchCVCLLearnersFactory
from CodeResearch.LearningFramework.Learners.NNTorchModelLearner import TorchModelLearner
from CodeResearch.LearningFramework.NeuralNetwork.PytorchHelpers import ResNet18CIFAR, CifarResNet6n2


class Cifar100LearnerFactory(PyTorchCVCLLearnersFactory):
    def __init__(self, nClasses, targetBatchSize=128, scoringBatchSize=64):
        super().__init__(nClasses, targetBatchSize, scoringBatchSize)

    def createScoreLearner_int(self, epochs):
        # scoring: сеть меньше, lr ниже, чтобы метрика была стабильнее
        cifar100_scoring_learner = TorchModelLearner(
            model_factory=lambda: CifarResNet6n2(num_classes=self.nClasses, n=3, width_mult=0.5),

            optimizer_name="sgd",
            lr=0.02,  # мягче, чем target (стабильность сигналов)
            weight_decay=5e-4,
            momentum=0.9,
            nesterov=True,

            batch_size=self.scoringBatchSize,
            update_epochs=epochs,
            epochs=epochs,

            scheduler_name="cosine",  # cosine annealing по эпохам/шагам (как реализовано у тебя)
        )
        return cifar100_scoring_learner

    def createTargetLearner_int(self, epochs):
        cifar100_target_learner = TorchModelLearner(
            model_factory=lambda: ResNet18CIFAR(num_classes=self.nClasses, width_mult=1.0),

            optimizer_name="sgd",
            lr=0.05,  # классика для CIFAR при bs~128
            weight_decay=5e-4,
            momentum=0.9,
            nesterov=True,

            batch_size=self.targetBatchSize,
            update_epochs=epochs,
            epochs=epochs,

            scheduler_name="cosine",
        )
        return cifar100_target_learner