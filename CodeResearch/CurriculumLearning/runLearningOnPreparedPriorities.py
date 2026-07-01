import numpy as np
from pathlib import Path

from CodeResearch.CurriculumLearning.clServices.Cifar100LearnerFactory import Cifar100LearnerFactory
from CodeResearch.CurriculumLearning.clServices.Cifar10LearnerFactory import Cifar10LearnerFactory
from CodeResearch.CurriculumLearning.clServices.MnistLearnerFactory import MnistLearnerFactory
from CodeResearch.CurriculumLearning.clServices.clHelpers import loadPriorities, filterDataSetByFraction, \
    createPredefinedSampler
from CodeResearch.LearningFramework.Learners.CompositeLearner import CompositeLearner
from CodeResearch.LearningFramework.Learners.epochLearner import EpochLearner
from CodeResearch.LearningFramework.Loggers.EpochLearnerLogger import EpochLearnerLogger
from CodeResearch.LearningFramework.generalLearningEstimator import GeneralLearningEstimator
from CodeResearch.dataSets import loadMnist_torch, loadCifar100_torch, loadCifar10_torch

folder = 'mnist_epoch'
repeats = 15

currentFolder = Path(folder)
files = sorted(currentFolder.glob('*.zip'))

for file in files:
    priorities, probs, prefix, taskName, betas, method = loadPriorities(file)

    if taskName == 'mnist_epoch':
        x, y, xtest, ytest = loadMnist_torch()
        x, y = filterDataSetByFraction(x, y, 1)
        xtest, ytest = filterDataSetByFraction(xtest, ytest, 1)

        nClasses = len(np.unique(y))
        learnerFactory = MnistLearnerFactory(nClasses)

        targetEpochs = 35
        easinessEpochs = 20
        diversityEpochs = 20
    elif taskName == 'cifar100_epoch':
        x, y, xtest, ytest = loadCifar100_torch()
        x, y = filterDataSetByFraction(x, y, 1)
        xtest, ytest = filterDataSetByFraction(xtest, ytest, 1)

        nClasses = len(np.unique(y))
        learnerFactory = Cifar100LearnerFactory(nClasses)

        targetEpochs = 200
        easinessEpochs = 20
        diversityEpochs = 20
    elif taskName == 'cifar_epoch':
        x, y, xtest, ytest = loadCifar10_torch()
        x, y = filterDataSetByFraction(x, y, 1)
        xtest, ytest = filterDataSetByFraction(xtest, ytest, 1)

        nClasses = len(np.unique(y))
        learnerFactory = Cifar10LearnerFactory(nClasses)

        targetEpochs = 200
        easinessEpochs = 20
        diversityEpochs = 20
    else:
        raise ValueError(f'Incorrect taskName: {taskName}')

    nFeatures = x.shape[1]
    nArrays = len(betas)
    nIterations = len(priorities)
    baseLabels = [method]

    prefix = f'{prefix}_{targetEpochs}_{easinessEpochs}_{diversityEpochs}'
    logger = EpochLearnerLogger(targetEpochs, taskName, prefix, nIterations, repeats, nArrays, betas, baseLabels)

    targetLearner = learnerFactory.createTargetLearner(targetEpochs)
    dataProcessor = learnerFactory.getDataPreprocessor()

    compositeLearner = CompositeLearner(EpochLearner(targetEpochs, targetLearner), logger)
    generalLearner = GeneralLearningEstimator(nIterations, logger)

    sampler = createPredefinedSampler(x, y, xtest, ytest, repeats, priorities, probs, logger)

    logger.logDebug(f'Starting task {prefix}')
    generalLearner.estimateLearner(sampler, compositeLearner)
    logger.logDebug(f'Finished task {prefix}')