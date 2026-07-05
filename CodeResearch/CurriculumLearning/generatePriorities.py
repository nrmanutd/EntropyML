import numpy as np

from CodeResearch.CurriculumLearning.clServices.Cifar100LearnerFactory import Cifar100LearnerFactory
from CodeResearch.CurriculumLearning.clServices.Cifar10LearnerFactory import Cifar10LearnerFactory
from CodeResearch.CurriculumLearning.clServices.MnistLearnerFactory import MnistLearnerFactory
from CodeResearch.CurriculumLearning.clServices.clHelpers import filterDataSet, \
    filterDataSetByFraction, createPrioritizer, savePriorities, createHCBuilder
from CodeResearch.Helpers.Logger.SimpleLogger import SimpleLogger
from CodeResearch.dataSets import loadCifar100_torch, loadCifar10_torch, loadMnist_torch

datasetFraction = 1
nIterations = 3

nEasinessAttempts = 50
diversityAttempts = 30

betas = [0.05, 0.1, 0.2, 0.5]
batchSize = 128

shouldEstimateForFullSet = False
loggingBetas = betas if shouldEstimateForFullSet is False else [1]

nSamples = 2000
alphas = np.array([1])
fraction = 0.5
trainAlpha = 1
testAlpha = 0.5

taskNames = ['mnist_epoch', 'cifar_epoch', 'cifar100_epoch']
#taskNames = ['mnist_epoch']

logger = SimpleLogger()

for i in range(0, len(taskNames)):
    taskName = taskNames[i]

    if taskName == 'mnist_epoch':
        x, y, xtest, ytest = loadMnist_torch()
        x, y = filterDataSetByFraction(x, y, datasetFraction)
        xtest, ytest = filterDataSetByFraction(xtest, ytest, datasetFraction)

        nClasses = len(np.unique(y))
        learnerFactory = MnistLearnerFactory(nClasses)
        noincrementEpochs = 20
    elif taskName == 'cifar100_epoch':
        x, y, xtest, ytest = loadCifar100_torch()
        x, y = filterDataSetByFraction(x, y, datasetFraction)
        xtest, ytest = filterDataSetByFraction(xtest, ytest, datasetFraction)

        nClasses = len(np.unique(y))
        learnerFactory = Cifar100LearnerFactory(nClasses)
        noincrementEpochs = 40
    elif taskName == 'cifar_epoch':
        x, y, xtest, ytest = loadCifar10_torch()
        x, y = filterDataSetByFraction(x, y, datasetFraction)
        xtest, ytest = filterDataSetByFraction(xtest, ytest, datasetFraction)

        nClasses = len(np.unique(y))
        learnerFactory = Cifar10LearnerFactory(nClasses)
        noincrementEpochs = 40
    else:
        raise ValueError(f'Incorrect taskName: {taskName}')

    nFeatures = x.shape[1]

    dataProcessor = learnerFactory.getDataPreprocessor()
    scoreLearnerBuilder = lambda e: learnerFactory.createScoreLearner(e)

    methods = ['rand', 'GradNorm', 'EL2N', 'cos', 'entropy', 'h', 'e']
    methods_inc = ['GradNorm_inc', 'EL2N_inc', 'cos_inc', 'entropy_inc', 'h_inc', 'e_inc']
    methods_addedHardness = ['h&GradNorm_inc', 'h&EL2N_inc', 'h&cos_inc', 'h&entropy_inc']

    noincrementAttempts = 10
    easinessEpochs = 20
    diversityEpochs = 20

    targetLearnerCreator = lambda: learnerFactory.createTargetLearner(noincrementEpochs)

    methods_to_iterate = methods
    trainedModelsList = []

    logger.logDebug(f'Task = {taskName}, {methods_to_iterate}')

    for method in methods_to_iterate:
        resultPriorities = []
        resultProbs = []

        logger.logDebug(f'Generating priorities for method {method}, task {taskName}...')

        prefix = f'{taskName}_{nIterations}_{nEasinessAttempts}_{fraction}_{datasetFraction}_{method}_{noincrementEpochs}_{noincrementAttempts}_NN'

        hcBuilder = createHCBuilder(method, nEasinessAttempts, logger, easinessEpochs,
                                                             diversityAttempts, diversityEpochs, batchSize, scoreLearnerBuilder, dataProcessor)

        prioritizer = createPrioritizer(hcBuilder, logger, alphas, betas, shouldEstimateForFullSet, method, noincrementAttempts, batchSize, dataProcessor, targetLearnerCreator, trainedModelsList)

        for i in range(nIterations):
            logger.logDebug(f'Calculating priorities for metric {method}, iteration # {i} ({nIterations})')
            pp, pprobs = prioritizer.calculatePriority(x, y)

            for k in range(len(pp)):
                resultPriorities.append(pp[k])
                resultProbs.append(pprobs[k])

            logger.logDebug(f'Finished calculating priorities for metric {method}, task {taskName}, iteration #{i}')
            savePriorities(resultPriorities, resultProbs, prefix, betas, taskName, method)
            logger.logDebug(f'Saved priorities')