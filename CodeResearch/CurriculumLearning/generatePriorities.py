import numpy as np

from CodeResearch.CurriculumLearning.clServices.ExperimentsConfig.experimentsHelpers import getDataset, \
    getExperimentConfig
from CodeResearch.CurriculumLearning.clServices.clHelpers import createPrioritizer, savePriorities, createHCBuilder
from CodeResearch.Helpers.Logger.SimpleLogger import SimpleLogger

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
logger = SimpleLogger()

for i in range(0, len(taskNames)):
    taskName = taskNames[i]

    x, y, xtest, ytest = getDataset(taskName, datasetFraction, 42)
    nClasses = len(np.unique(y))

    methods = ['rand', 'GradNorm', 'EL2N', 'cos', 'entropy', 'h', 'e', 'forgetting'] #todo: add rand
    methods_inc = ['GradNorm_inc', 'EL2N_inc', 'cos_inc', 'cos_to_train_inc', 'entropy_inc', 'h_inc', 'e_inc', 'k-centered_inc']
    methods_addedHardness = ['h&GradNorm_inc', 'h&EL2N_inc', 'h&cos_inc', 'h&entropy_inc', 'h&k-centered_inc']

    #restMethods = ['forgetting', 'k-centered_inc', 'h&k-centered_inc']
    restMethods = ['forgetting', 'k-centered_inc', 'h&k-centered_inc']

    methods_to_iterate = restMethods
    trainedModelsList = []

    logger.logDebug(f'Task = {taskName}, {methods_to_iterate}')

    for method in methods_to_iterate:

        resultPriorities = []
        resultProbs = []

        currentConfig = getExperimentConfig(taskName, method, nClasses)

        currentModelsList = trainedModelsList if method != 'forgetting' else []

        logger.logDebug(f'Generating priorities for method {method}, task {taskName}...')

        prefix = f'{taskName}_{nIterations}_{nEasinessAttempts}_{fraction}_{datasetFraction}_{method}_{currentConfig.noincrementEpochs}_{currentConfig.noincrementAttempts}_NN'

        hcBuilder = createHCBuilder(method, nEasinessAttempts, logger, currentConfig.easinessEpochs,
                                    diversityAttempts, currentConfig.diversityEpochs, batchSize, currentConfig.scoreHardnessLearnerBuilder, currentConfig.scoreDiversityLearnerBuilder, currentConfig.dataProcessor)

        prioritizer = createPrioritizer(hcBuilder, logger, alphas, betas, shouldEstimateForFullSet, method, currentConfig.noincrementAttempts, batchSize, currentConfig.dataProcessor, currentConfig.targetForScoringLearnerCreator, currentModelsList)

        for i in range(nIterations):
            logger.logDebug(f'Calculating priorities for metric {method}, iteration # {i} ({nIterations})')
            pp, pprobs = prioritizer.calculatePriority(x, y)

            for k in range(len(pp)):
                resultPriorities.append(pp[k])
                resultProbs.append(pprobs[k])

            logger.logDebug(f'Finished calculating priorities for metric {method}, task {taskName}, iteration #{i}')
            savePriorities(resultPriorities, resultProbs, prefix, betas, taskName, method)
            logger.logDebug(f'Saved priorities')