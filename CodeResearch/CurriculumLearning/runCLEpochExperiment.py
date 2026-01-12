import numpy as np

from CodeResearch.CurriculumLearning.clServices.Cifar100LearnerFactory import Cifar100LearnerFactory
from CodeResearch.CurriculumLearning.clServices.Cifar10LearnerFactory import Cifar10LearnerFactory
from CodeResearch.CurriculumLearning.clServices.MnistLearnerFactory import MnistLearnerFactory
from CodeResearch.CurriculumLearning.clServices.clHelpers import filterDataSet, \
    createSampler, createLearnerHC, filterDataSetByFraction
from CodeResearch.Helpers.commonHelpers import normalizeTarget
from CodeResearch.LearningFramework.Learners.CompositeLearner import CompositeLearner
from CodeResearch.LearningFramework.Learners.epochLearner import EpochLearner
from CodeResearch.LearningFramework.Loggers.EpochLearnerLogger import EpochLearnerLogger
from CodeResearch.LearningFramework.Samplers.SamplersFactories.RandomAllSetSamplerFactory import \
    RandomAllsetSamplerFactory
from CodeResearch.LearningFramework.generalLearningEstimator import GeneralLearningEstimator
from CodeResearch.ObjectComplexity.InstancePriority.PrioritizerType import PrioritizerType
from CodeResearch.dataSets import loadMnist_cnn, loadCifar100_torch, loadCifar10_torch

datasetFraction = 0.1
nIterations = 20

nEasinessAttempts = 50
diversityAttempts = 50

targetEpochs = 20
easinessEpochs = 10
diversityEpochs = 10

targetBatchSize = 128
scoringBatchSize = 64

repeats = 5
betas = [0.05, 0.1, 0.2, 0.5]
baseLabels = ['l', 'h&i_inc', 'h&h_inc']
nArrays = len(baseLabels) * len(betas)

nSamples = 2000
alphas = np.array([0.5])
fraction = 0.5
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

taskNames = ['mnist_epoch', 'cifar100_epoch', 'cifar100_epoch', 'cifar100_epoch', 'cifar100_epoch', 'cifar100_epoch', 'cifar100_epoch', 'cifar100_epoch', 'mnist_epoch', 'cifar_epoch', 'cifar_epoch']
firstClasses = [-1, 43, 47, 43, 70, 9, 23, 5, 3, 0]
secondClasses = [-1, 87, 52, 88, 91, 10, 33, 6, 5, 8]

for i in range(1, len(taskNames)):
    taskName = taskNames[i]
    firstClass = firstClasses[i]
    secondClass = secondClasses[i]

    if taskName == 'mnist_epoch':
        x, y = loadMnist_cnn()
        x, y = filterDataSetByFraction(x, y, datasetFraction)
        nClasses = len(np.unique(y))
        learnerFactory = MnistLearnerFactory(nClasses, targetBatchSize, scoringBatchSize)
    elif taskName == 'cifar100_epoch':
        x, y = loadCifar100_torch()
        x, y = filterDataSet(x, y, datasetFraction, firstClass, secondClass)
        y = normalizeTarget(y)
        nClasses = len(np.unique(y))
        learnerFactory = Cifar100LearnerFactory(nClasses, targetBatchSize, scoringBatchSize)
    elif taskName == 'mnist_epoch' and i > 0:
        x, y = loadMnist_cnn()
        x, y = filterDataSet(x, y, datasetFraction, firstClass, secondClass)
        y = normalizeTarget(y)
        nClasses = len(np.unique(y))
        learnerFactory = MnistLearnerFactory(nClasses, targetBatchSize, scoringBatchSize)
    elif taskName == 'cifar10_epoch':
        x, y = loadCifar10_torch()
        x, y = filterDataSet(x, y, datasetFraction, firstClass, secondClass)
        y = normalizeTarget(y)
        nClasses = len(np.unique(y))
        learnerFactory = Cifar10LearnerFactory(nClasses, targetBatchSize, scoringBatchSize)
    else:
        raise ValueError(f'Incorrect taskName: {taskName}')

    nFeatures = x.shape[1]

    prefix = f'{taskName}_{nIterations}_{nEasinessAttempts}_{fraction}_{datasetFraction}_{repeats}_{nArrays}_{targetEpochs}_{firstClass}_{secondClass}_NN'
    logger = EpochLearnerLogger(targetEpochs, taskName, prefix, nEasinessAttempts, repeats, nArrays, betas, baseLabels)

    targetLearner = learnerFactory.createTargetLearner(None)
    compositeLearner = CompositeLearner(EpochLearner(targetEpochs, targetLearner, RandomAllsetSamplerFactory(targetBatchSize, PrioritizerType.Probability)), logger)
    generalLearner = GeneralLearningEstimator(nIterations, logger)

    #hc = createLearnerBasedHardnessCalculator(nAttempts, logger, x.shape[1], nClasses, hardnessEpochs, hidden_sizes)
    #sampler = createSampler(x, y, alphas / (1 - testAlpha), betas, testAlpha, repeats, lambda shouldUseLearner: createLearnerBasedHardnessCalculator(nAttempts, logger, x.shape[1], nClasses, hardnessEpochs, hidden_sizes, dAttempts, dBatchSize, dEpochs, dHidden_sizes, shouldUseLearner), logger)
    sampler = createSampler(x, y, alphas / (1 - testAlpha), betas, testAlpha, repeats,
                            lambda shouldUseLearner: createLearnerHC(nEasinessAttempts, logger, easinessEpochs, diversityAttempts,
                                                                     scoringBatchSize, diversityEpochs, lambda e: learnerFactory.createScoreLearner(e)), logger)

    logger.logDebug(f'Starting task {prefix}')
    generalLearner.estimateLearner(sampler, compositeLearner)
    logger.logDebug(f'Finished task {prefix}')