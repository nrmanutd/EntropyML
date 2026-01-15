import numpy as np

from CodeResearch.CurriculumLearning.clServices.Cifar100LearnerFactory import Cifar100LearnerFactory, \
    Cifar100CachedOptimizerLearnerFactory
from CodeResearch.CurriculumLearning.clServices.Cifar10LearnerFactory import Cifar10LearnerFactory
from CodeResearch.CurriculumLearning.clServices.MnistLearnerFactory import MnistLearnerFactory
from CodeResearch.CurriculumLearning.clServices.clHelpers import filterDataSet, \
    createLearnerHC, filterDataSetByFraction, createSamplerWithTest, createSampler, createLearnerHCForChain, \
    createSamplerForChain
from CodeResearch.LearningFramework.Learners.CompositeLearner import CompositeLearner
from CodeResearch.LearningFramework.Learners.epochLearner import EpochLearner
from CodeResearch.LearningFramework.Loggers.EpochLearnerLogger import EpochLearnerLogger
from CodeResearch.LearningFramework.generalLearningEstimator import GeneralLearningEstimator
from CodeResearch.dataSets import loadCifar100_torch, loadCifar10_torch, loadMnist_torch

datasetFraction = 1
nIterations = 50

nEasinessAttempts = 50
diversityAttempts = 30

repeats = 1
betas = [0.05, 0.1, 0.2, 0.5]
#baseLabels = ['l', 'h&i_inc', 'h&h_inc']
baseLabels = ['h&i_inc']
nArrays = len(baseLabels) * len(betas)

nSamples = 2000
alphas = np.array([1])
fraction = 0.5
trainAlpha = 1
testAlpha = 0.5
hidden_sizes = (16, 16)
dHidden_sizes = (16, 16)

#x, y = make_random(nSamples)
#x, y = datasets.make_blobs(n_samples=nSamples, centers=2, n_features=2, random_state=42)
#x, y = make_xor(nSamples)
#x, y = datasets.make_circles(n_samples=nSamples, factor=0.5, noise=0.1, random_state=42)
#x, y = make_spirals(nSamples)
#x, y = loadMnist()
#x, y = loadCifar()
#x, y = loadFashionMnist()
#x, y = filterDataSet(x, y, datasetFraction, firstClass, secondClass)
#x, y = load_proteins("../Data/Proteins/df_master.csv")

taskNames = ['cifar100_epoch', 'mnist_epoch', 'cifar_epoch', 'cifar100_epoch', 'cifar100_epoch', 'cifar100_epoch', 'cifar100_epoch', 'cifar100_epoch', 'cifar100_epoch', 'mnist_epoch', 'cifar_epoch', 'cifar_epoch']
firstClasses = [-1, -1, -1, 43, 47, 43, 70, 9, 23, 5, 3, 0]
secondClasses = [-1, -1, -1, 87, 52, 88, 91, 10, 33, 6, 5, 8]

for i in range(len(taskNames)):
    taskName = taskNames[i]
    firstClass = firstClasses[i]
    secondClass = secondClasses[i]

    if taskName == 'mnist_epoch' and firstClass == -1:
        x, y, xtest, ytest = loadMnist_torch()
        x, y = filterDataSetByFraction(x, y, datasetFraction)
        xtest, ytest = filterDataSetByFraction(xtest, ytest, datasetFraction)

        nClasses = len(np.unique(y))
        learnerFactory = MnistLearnerFactory(nClasses)
        targetEpochs = 20
        easinessEpochs = 5
        diversityEpochs = 5
    elif taskName == 'mnist_epoch':
        x, y, xtest, ytest = loadMnist_torch()
        x, y = filterDataSetByFraction(x, y, datasetFraction)
        xtest, ytest = filterDataSet(xtest, ytest, datasetFraction, firstClass, secondClass)

        nClasses = len(np.unique(y))
        learnerFactory = MnistLearnerFactory(nClasses)
        targetEpochs = 20
        easinessEpochs = 5
        diversityEpochs = 5
    elif taskName == 'cifar100_epoch' and firstClass == -1:
        x, y, xtest, ytest = loadCifar100_torch()
        x, y = filterDataSetByFraction(x, y, datasetFraction)
        xtest, ytest = filterDataSetByFraction(xtest, ytest, datasetFraction)

        nClasses = len(np.unique(y))
        learnerFactory = Cifar100CachedOptimizerLearnerFactory(nClasses)
        #learnerFactory = Cifar100LearnerFactory(nClasses)
        targetEpochs = 20
        easinessEpochs = 20
        diversityEpochs = 20
    elif taskName == 'cifar100_epoch':
        x, y, xtest, ytest = loadCifar100_torch()
        x, y = filterDataSet(x, y, datasetFraction, firstClass, secondClass)
        xtest, ytest = filterDataSet(xtest, ytest, datasetFraction, firstClass, secondClass)

        nClasses = len(np.unique(y))
        learnerFactory = Cifar100LearnerFactory(nClasses)
        targetEpochs = 30
        easinessEpochs = 10
        diversityEpochs = 10
    elif taskName == 'cifar_epoch' and firstClass == -1:
        x, y, xtest, ytest = loadCifar100_torch()
        x, y = filterDataSetByFraction(x, y, datasetFraction)
        xtest, ytest = filterDataSetByFraction(xtest, ytest, datasetFraction)

        nClasses = len(np.unique(y))
        learnerFactory = Cifar10LearnerFactory(nClasses)
        targetEpochs = 50
        easinessEpochs = 10
        diversityEpochs = 10
    elif taskName == 'cifar_epoch':
        x, y, xtest, ytest = loadCifar10_torch()
        x, y = filterDataSet(x, y, datasetFraction, firstClass, secondClass)
        xtest, ytest = filterDataSet(xtest, ytest, datasetFraction, firstClass, secondClass)

        nClasses = len(np.unique(y))
        learnerFactory = Cifar10LearnerFactory(nClasses)
        targetEpochs = 50
        easinessEpochs = 10
        diversityEpochs = 10
    else:
        raise ValueError(f'Incorrect taskName: {taskName}')

    nFeatures = x.shape[1]

    prefix = f'{taskName}_{nIterations}_{nEasinessAttempts}_{fraction}_{datasetFraction}_{repeats}_{nArrays}_{targetEpochs}_{firstClass}_{secondClass}_NN'
    logger = EpochLearnerLogger(targetEpochs, taskName, prefix, nEasinessAttempts, repeats, nArrays, betas, baseLabels)

    targetLearner = learnerFactory.createTargetLearner(targetEpochs)
    dataProcessor = learnerFactory.getDataPreprocessor()

    compositeLearner = CompositeLearner(EpochLearner(targetEpochs, targetLearner), logger)
    generalLearner = GeneralLearningEstimator(nIterations, logger)

    scoreLearnerBuilder = lambda e: learnerFactory.createScoreLearner(e)
    hcBuilder = lambda shouldUseLearner: createLearnerHC(nEasinessAttempts, logger, easinessEpochs, diversityAttempts, diversityEpochs, scoreLearnerBuilder, dataProcessor)
    #sampler = createSamplerWithTest(x, y, xtest, ytest, alphas, betas, trainAlpha, repeats, hcBuilder, logger)
    #sampler = createSampler(x, y, alphas, betas, testAlpha, repeats, hcBuilder, logger)

    hc = createLearnerHCForChain(nEasinessAttempts, logger, easinessEpochs, scoreLearnerBuilder, dataProcessor)
    sampler = createSamplerForChain(x, y, xtest, ytest, betas, trainAlpha, repeats, hc, lambda: learnerFactory.createScoreLearner(easinessEpochs), logger)

    logger.logDebug(f'Starting task {prefix}')
    generalLearner.estimateLearner(sampler, compositeLearner)
    logger.logDebug(f'Finished task {prefix}')