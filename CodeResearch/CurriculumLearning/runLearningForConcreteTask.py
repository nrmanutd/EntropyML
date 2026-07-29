import math
import re

import numpy as np
from pathlib import Path

from CodeResearch.CurriculumLearning.clServices.Cifar100LearnerFactory import Cifar100LearnerFactory
from CodeResearch.CurriculumLearning.clServices.Cifar10LearnerFactory import Cifar10LearnerFactory
from CodeResearch.CurriculumLearning.clServices.MnistLearnerFactory import MnistLearnerFactory
from CodeResearch.CurriculumLearning.clServices.SVHNLearnerFactory import SVHNLearnerFactory
from CodeResearch.CurriculumLearning.clServices.TinyImageNetLearnerFactory import TinyImageNetLearnerFactory
from CodeResearch.CurriculumLearning.clServices.clHelpers import loadPriorities, filterDataSetByFraction, \
        createPredefinedSampler
from CodeResearch.Datasets.tinyImageNet import loadTinyImageNet_torch
from CodeResearch.LearningFramework.Learners.CompositeLearner import CompositeLearner
from CodeResearch.LearningFramework.Learners.epochLearner import EpochLearner
from CodeResearch.LearningFramework.Loggers.EpochLearnerLogger import EpochLearnerLogger
from CodeResearch.LearningFramework.generalLearningEstimator import GeneralLearningEstimator
from CodeResearch.Datasets.dataSets import loadMnist_torch, loadCifar100_torch, loadCifar10_torch, loadSVHN_torch

def runLearningForConcreteTask(folder: str):
    datasetFraction = 1
    repeats = 5
    currentFolder = Path(folder)

    prefixPattern = re.compile(r'^.*\.zip')

    files = sorted([
        f for f in currentFolder.glob('*.zip')
        if prefixPattern.match(f.name)
    ])

    for file in files:
        priorities, probs, prefix, taskName, betas, method = loadPriorities(file)

        if taskName == 'mnist_epoch':
            x, y, xtest, ytest = loadMnist_torch()
            x, y = filterDataSetByFraction(x, y, datasetFraction, 42)
            xtest, ytest = filterDataSetByFraction(xtest, ytest, datasetFraction, 42)

            nClasses = len(np.unique(y))
            learnerFactory = MnistLearnerFactory(nClasses)

            targetEpochs = 60
        elif taskName == 'cifar100_epoch':
            x, y, xtest, ytest = loadCifar100_torch()
            x, y = filterDataSetByFraction(x, y, datasetFraction, 42)
            xtest, ytest = filterDataSetByFraction(xtest, ytest, datasetFraction, 42)

            nClasses = len(np.unique(y))
            learnerFactory = Cifar100LearnerFactory(nClasses)

            targetEpochs = 200
        elif taskName == 'cifar_epoch':
            x, y, xtest, ytest = loadCifar10_torch()
            x, y = filterDataSetByFraction(x, y, datasetFraction, 42)
            xtest, ytest = filterDataSetByFraction(xtest, ytest, datasetFraction, 42)

            nClasses = len(np.unique(y))
            learnerFactory = Cifar10LearnerFactory(nClasses)

            targetEpochs = 200
        elif taskName == 'svhn_epoch':
            x, y, xtest, ytest = loadSVHN_torch()
            x, y = filterDataSetByFraction(x, y, datasetFraction, 42)
            xtest, ytest = filterDataSetByFraction(xtest, ytest, datasetFraction, 42)

            nClasses = len(np.unique(y))
            learnerFactory = SVHNLearnerFactory(nClasses)

            targetEpochs = 100
        elif taskName == 'tinyImageNet_epoch':
            x, y, xtest, ytest = loadTinyImageNet_torch()
            x, y = filterDataSetByFraction(x, y, datasetFraction, 42)
            xtest, ytest = filterDataSetByFraction(xtest, ytest, datasetFraction, 42)

            nClasses = len(np.unique(y))
            learnerFactory = TinyImageNetLearnerFactory(nClasses)

            targetEpochs = 200
        else:
            raise ValueError(f'Incorrect taskName: {taskName}')

        nArrays = len(betas)
        if len(priorities) % nArrays != 0:
            raise ValueError(f'Incorrect length of priorities ({len(priorities)}) and nArrays ({nArrays})')

        nIterations = math.ceil(len(priorities) / nArrays)
        baseLabels = [method]

        prefix = f'{prefix}_{targetEpochs}'
        logger = EpochLearnerLogger(targetEpochs, taskName, prefix, nIterations, repeats, nArrays, betas, baseLabels)

        targetLearner = learnerFactory.createTargetLearner(targetEpochs)

        compositeLearner = CompositeLearner(EpochLearner(targetEpochs, targetLearner), logger)
        generalLearner = GeneralLearningEstimator(nIterations, logger)

        sampler = createPredefinedSampler(x, y, xtest, ytest, repeats, nIterations, priorities, probs, logger)

        logger.logDebug(f'Starting task {prefix}')
        generalLearner.estimateLearner(sampler, compositeLearner)
        logger.logDebug(f'Finished task {prefix}')