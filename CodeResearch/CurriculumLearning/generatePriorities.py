import numpy as np

from CodeResearch.CurriculumLearning.clServices.Cifar100LearnerFactory import Cifar100LearnerFactory
from CodeResearch.CurriculumLearning.clServices.Cifar10LearnerFactory import Cifar10LearnerFactory
from CodeResearch.CurriculumLearning.clServices.MnistLearnerFactory import MnistLearnerFactory
from CodeResearch.CurriculumLearning.clServices.clHelpers import filterDataSet, \
    createLearnerHC, filterDataSetByFraction, createPrioritizer, savePriorities
from CodeResearch.Helpers.Logger.SimpleLogger import SimpleLogger
from CodeResearch.dataSets import loadCifar100_torch, loadCifar10_torch, loadMnist_torch

datasetFraction = 1
nIterations = 3

nEasinessAttempts = 50
diversityAttempts = 30

#betas = [0.05, 0.1, 0.15, 0.2, 0.25, 0.5]
betas = [0.05, 0.1, 0.2, 0.5]
#betas = [1]
baseLabels = ['l', 'h&i_inc']
#baseLabels = ['GraNd']
shouldEstimateForFullSet = False
loggingBetas = betas if shouldEstimateForFullSet is False else [1]
nArrays = len(baseLabels) * (len(loggingBetas))

nSamples = 2000
alphas = np.array([1])
fraction = 0.5
trainAlpha = 1
testAlpha = 0.5

taskNames = ['mnist_epoch', 'cifar_epoch', 'cifar100_epoch']
firstClasses = [-1, -1, -1]
secondClasses = [-1, -1, -1]

for i in range(2, len(taskNames)):
    taskName = taskNames[i]
    firstClass = firstClasses[i]
    secondClass = secondClasses[i]

    if taskName == 'mnist_epoch' and firstClass == -1:
        x, y, xtest, ytest = loadMnist_torch()
        x, y = filterDataSetByFraction(x, y, datasetFraction)
        xtest, ytest = filterDataSetByFraction(xtest, ytest, datasetFraction)

        nClasses = len(np.unique(y))
        learnerFactory = MnistLearnerFactory(nClasses)

        targetEpochs = 35
        easinessEpochs = 20
        diversityEpochs = 20
    elif taskName == 'mnist_epoch':
        x, y, xtest, ytest = loadMnist_torch()
        x, y = filterDataSetByFraction(x, y, datasetFraction)
        xtest, ytest = filterDataSet(xtest, ytest, datasetFraction, firstClass, secondClass)

        nClasses = len(np.unique(y))
        learnerFactory = MnistLearnerFactory(nClasses)
        targetEpochs = 35
        easinessEpochs = 20
        diversityEpochs = 20
    elif taskName == 'cifar100_epoch' and firstClass == -1:
        x, y, xtest, ytest = loadCifar100_torch()
        x, y = filterDataSetByFraction(x, y, datasetFraction)
        xtest, ytest = filterDataSetByFraction(xtest, ytest, datasetFraction)

        nClasses = len(np.unique(y))
        learnerFactory = Cifar100LearnerFactory(nClasses)

        targetEpochs = 200
        easinessEpochs = 20
        diversityEpochs = 20
    elif taskName == 'cifar100_epoch':
        x, y, xtest, ytest = loadCifar100_torch()
        x, y = filterDataSet(x, y, datasetFraction, firstClass, secondClass)
        xtest, ytest = filterDataSet(xtest, ytest, datasetFraction, firstClass, secondClass)

        nClasses = len(np.unique(y))
        learnerFactory = Cifar100LearnerFactory(nClasses)
        targetEpochs = 200
        easinessEpochs = 20
        diversityEpochs = 20
    elif taskName == 'cifar_epoch' and firstClass == -1:
        x, y, xtest, ytest = loadCifar10_torch()
        x, y = filterDataSetByFraction(x, y, datasetFraction)
        xtest, ytest = filterDataSetByFraction(xtest, ytest, datasetFraction)

        nClasses = len(np.unique(y))
        learnerFactory = Cifar10LearnerFactory(nClasses)

        targetEpochs = 200
        easinessEpochs = 20
        diversityEpochs = 20
    elif taskName == 'cifar_epoch':
        x, y, xtest, ytest = loadCifar10_torch()
        x, y = filterDataSet(x, y, datasetFraction, firstClass, secondClass)
        xtest, ytest = filterDataSet(xtest, ytest, datasetFraction, firstClass, secondClass)

        nClasses = len(np.unique(y))
        learnerFactory = Cifar10LearnerFactory(nClasses)
        targetEpochs = 200
        easinessEpochs = 20
        diversityEpochs = 20
    else:
        raise ValueError(f'Incorrect taskName: {taskName}')

    nFeatures = x.shape[1]

    logger = SimpleLogger()
    dataProcessor = learnerFactory.getDataPreprocessor()
    scoreLearnerBuilder = lambda e: learnerFactory.createScoreLearner(e)


    resultPriorities = []
    resultProbs = []

    methods = ['rand', 'GradNorm', 'EL2N', 'cos', 'entropy', 'h']
    methods_inc = ['GradNorm_inc', 'EL2N_inc', 'cos_inc', 'entropy_inc', 'h_inc']
    methods_addedHardness = ['h&GradNorm_inc', 'h&EL2N_inc', 'h&cos_inc', 'h&entropy_inc']

    noincrementEpochs = 40
    noincrementAttempts = 10
    targetLearnerCreator = lambda: learnerFactory.createTargetLearner(noincrementEpochs)

    methods_to_iterate = methods

    for m in methods_to_iterate:
        prefix = f'{taskName}_{nIterations}_{nEasinessAttempts}_{fraction}_{datasetFraction}_{nArrays}_{targetEpochs}_{m}_NN'
        hcBuilder = lambda shouldUseLearner: createLearnerHC(nEasinessAttempts, logger, easinessEpochs,
                                                             diversityAttempts, diversityEpochs, m, scoreLearnerBuilder,
                                                             dataProcessor)
        prioritizer = createPrioritizer(hcBuilder, logger, alphas, betas, shouldEstimateForFullSet, m, noincrementAttempts, dataProcessor, targetLearnerCreator)

        for i in range(nIterations):
            logger.logDebug(f'Calculating priorities for metric {m}, iteration # {i} ({nIterations})')
            pp, pprobs = prioritizer.calculatePriority(x, y)

            for k in range(len(pp)):
                resultPriorities.append(pp[k])
                resultProbs.append(pprobs[k])

        logger.logDebug(f'Finished calculating priorities for metric {m}')
        savePriorities(resultPriorities, resultProbs, prefix, baseLabels)
        logger.logDebug(f'Saved priorities for metric {m}, prefix {prefix}')