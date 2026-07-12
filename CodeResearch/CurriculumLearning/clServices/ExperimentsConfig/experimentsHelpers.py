import numpy as np

from CodeResearch.CurriculumLearning.clServices.Cifar100LearnerFactory import Cifar100LearnerFactory, \
    Cifar100ForgettingLearnerFactory
from CodeResearch.CurriculumLearning.clServices.Cifar10LearnerFactory import Cifar10LearnerFactory, \
    Cifar10ForgettingLearnerFactory
from CodeResearch.CurriculumLearning.clServices.ExperimentsConfig.ExperimentConfig import ExperimentConfig
from CodeResearch.CurriculumLearning.clServices.MnistLearnerFactory import MnistLearnerFactory, \
    MnistForgettingLearnerFactory
from CodeResearch.CurriculumLearning.clServices.clHelpers import filterDataSetByFraction
from CodeResearch.dataSets import loadMnist_torch, loadCifar10_torch, loadCifar100_torch, loadSVHN_torch


def getDataset(taskName: str, datasetFraction, seed: int):
    if taskName == 'mnist_epoch':
        x, y, xtest, ytest = loadMnist_torch()
    elif taskName == 'cifar100_epoch':
        x, y, xtest, ytest = loadCifar100_torch()
    elif taskName == 'cifar_epoch':
        x, y, xtest, ytest = loadCifar10_torch()
    elif taskName == 'svhn_epoch':
        x, y, xtest, ytest = loadSVHN_torch()
    else:
        raise ValueError(f'Incorrect taskName: {taskName}')

    x, y = filterDataSetByFraction(x, y, datasetFraction, seed)
    xtest, ytest = filterDataSetByFraction(xtest, ytest, datasetFraction, seed)

    return x, y, xtest, ytest


def getLearnerFactoryAndFillConfig(taskName: str, method: str, nClasses: int, config: ExperimentConfig):
    shouldUseForgetting = ('forgetting' in method)

    if taskName == 'mnist_epoch':
        config.noincrementEpochs = 20
        config.easinessEpochs = 15
        config.diversityEpochs = 15

        if shouldUseForgetting:
            return MnistForgettingLearnerFactory(nClasses), config

        return MnistLearnerFactory(nClasses), config
    elif taskName == 'cifar100_epoch':
        config.noincrementEpochs = 40
        config.easinessEpochs = 40
        config.diversityEpochs = 40

        if shouldUseForgetting:
            return Cifar100ForgettingLearnerFactory(nClasses), config

        return Cifar100LearnerFactory(nClasses), config
    elif taskName == 'cifar_epoch':
        config.noincrementEpochs = 40
        config.easinessEpochs = 40
        config.diversityEpochs = 40

        if shouldUseForgetting:
            return Cifar10ForgettingLearnerFactory(nClasses), config

        return Cifar10LearnerFactory(nClasses), config
    else:
        raise ValueError(f'Incorrect taskName: {taskName}')

    pass


def getExperimentConfig(taskName: str, method: str, nClasses: int):
    config = ExperimentConfig()

    learnerFactory, config = getLearnerFactoryAndFillConfig(taskName, method, nClasses, config)

    config.noincrementAttempts = 10
    config.dataProcessor = learnerFactory.getDataPreprocessor()
    config.targetForScoringLearnerCreator = lambda: learnerFactory.createTargetForScoringLearner(config.noincrementEpochs)
    config.scoreHardnessLearnerBuilder = lambda e: learnerFactory.createScoreLearner(e)

    if 'k-centered' in method:
        config.scoreDiversityLearnerBuilder = lambda e: learnerFactory.createTargetForScoringLearner(e)
    else:
        config.scoreDiversityLearnerBuilder = lambda e: learnerFactory.createScoreLearner(e)

    return config
