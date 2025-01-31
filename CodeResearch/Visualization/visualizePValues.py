import math

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

def calcErrorOfModelAdditive(pointIndx, modelValues):

    fu = modelValues[pointIndx, 0]
    fl = modelValues[0, 0]

    su = modelValues[-1, 0]
    sl = fu

    firstValue = fu - fl
    secondValue = su - sl

    firstSigmaUp = modelValues[pointIndx, 1]
    firstSigmaLow = modelValues[0, 1]

    secondSigmaUp = modelValues[-1, 1]
    secondSigmaLow = firstSigmaUp

    firstEpsilon = math.sqrt(firstSigmaUp**2 + firstSigmaLow**2)
    secondEpsilon = math.sqrt(secondSigmaUp**2 + secondSigmaLow**2)

    return {'value': firstValue, 'epsilon': firstEpsilon}, {'value': secondValue, 'epsilon': secondEpsilon}

def calcErrorOfModel(pointIndx, modelValues):

    if modelValues[pointIndx, 0] == 0:
        print('Error')

    fu = modelValues[pointIndx, 0]
    fl = modelValues[0, 0]

    su = modelValues[-1, 0]
    sl = fu

    firstValue = modelValues[pointIndx, 0] / modelValues[0, 0] - 1
    secondValue = modelValues[-1, 0] / modelValues[pointIndx, 0] - 1

    firstSigmaUp = modelValues[pointIndx, 1]
    firstSigmaLow = modelValues[0, 1]

    secondSigmaUp = modelValues[-1, 1]
    secondSigmaLow = firstSigmaUp

    firstEpsilon = (fl * firstSigmaUp + fu * firstSigmaLow) / (fl * fu)
    secondEpsilon = (sl * secondSigmaUp + su * secondSigmaLow) / (sl * su)

    return {'value': firstValue, 'epsilon': firstEpsilon}, {'value': secondValue, 'epsilon': secondEpsilon}




def makeDescription(rowName, deltaError):

    template = '{:} {:.2f} +/- {:.3f}, after {:.2f} +/- {:.3f}'

    return template.format(rowName, deltaError[0]['value'], deltaError[0]['epsilon'], deltaError[1]['value'], deltaError[1]['epsilon'])

def makeFullDescription(rowName, pointIndx, modelValues):
    delta1 = calcErrorOfModel(pointIndx, modelValues)
    delta2 = calcErrorOfModelAdditive(pointIndx, modelValues)

    d1 = makeDescription(rowName, delta1)
    d2 = makeDescription(rowName, delta2)

    return '{:}, {:}'.format(d1, d2)


def getBestComplexity(indicies, modelValues):

    bestNumber = 0
    bestInd = indicies[bestNumber]

    for i in range(1, len(indicies)):
        if indicies[i] > bestInd:
            bestInd = indicies[i]
            bestNumber = i

    #methods = ['med', 'mean', 'low', 'min']
    methods = ['M1', 'M2', 'M3', 'M4']

    description = makeFullDescription(methods[bestNumber], bestInd, modelValues)
    return bestInd, methods[bestNumber]


def visualizePValues(data):
    xSteps = data['steps']
    targetResults = data['targetResults']
    pValuesResults = data['pValuesResults']
    meanPValues = data['meanPValue']
    medianPValues = data['medianPValue']
    classesPair = data['classes']
    fast = data['fast']
    fastUp = data['fastUp']
    iStep = data['step']
    taskName = data['taskName']
    pairIdx = data['pairIndex']
    nAttempts = data['nAttempts']
    pairsNames = data['names']
    model = data['model']
    nPoints = data['nPoints']
    epsilon = data['epsilon']
    alpha = data['alpha']
    beta = data['beta']

    px = 1 / plt.rcParams['figure.dpi']
    fig, ax = plt.subplots(1, 1, sharex=True, tight_layout=True, figsize=(1920 * px, 1280 * px))

    pairedClasses = fast.shape[1]

    idx = range(0, iStep + 1)
    maxIndex, description = getBestComplexity(nPoints[pairIdx, :], model[idx, pairIdx, :])

    header = 'KS statistic estimation for {:} {:} task. Complexity: {:} ({:}), KS = {:.2f}'.format(pairsNames[pairIdx], taskName, xSteps[maxIndex, pairIdx], description, meanPValues[iStep, pairIdx])

    ax.title.set_text(header)
    ax.title.set_size(25)
    ax.grid()

    # plot values
    #for i in np.arange(pairedClasses):
    #ax[0].plot(xSteps[idx], targetResults[idx, pairIdx], label="Theoretical bound #{0}".format(pairIdx), ls=':', marker='X')
    #ax[0].plot(xSteps[idx, pairIdx], model[idx, pairIdx, 0], label="XGBoost", ls='-', marker='o')
    #ax[0].plot(xSteps[idx, pairIdx], model[idx, pairIdx, 0] - model[idx, pairIdx, 1], label="XGBoost - sigma", ls=':', marker='o', color='blue', markersize=3)
    #ax[0].plot(xSteps[idx, pairIdx], model[idx, pairIdx, 0] + model[idx, pairIdx, 1], label="XGBoost + sigma", ls=':', marker='o', color='blue', markersize=3)

    #ax[0].plot(xSteps[idx], model[idx, pairIdx, 1], label="NN", ls=':', marker='x')
    ax.plot(xSteps[idx, pairIdx], fast[idx, pairIdx], label="KS Mean ±{:}/{:}q".format(beta, 1 - beta), ls='--', color='blueviolet', linewidth=2)
    ax.plot(xSteps[idx, pairIdx], fastUp[idx, pairIdx], label="_Mean + quantile {:}".format(beta), ls='--', color='blueviolet', linewidth=2)
    ax.plot(xSteps[idx, pairIdx], np.maximum(0, meanPValues[idx, pairIdx] - np.ones(len(idx))*epsilon/np.sqrt(xSteps[idx, pairIdx])), label="Mean ±{:}/{:}q (DWK)".format(alpha, 1 - alpha), ls=':', color='red', linewidth=2)
    ax.plot(xSteps[idx, pairIdx], meanPValues[idx, pairIdx], label="KS Mean", ls='-', marker='*')
    #ax[0].plot(xSteps[idx, pairIdx], medianPValues[idx, pairIdx], label="Median pValue", ls='-', marker='*', color='blue')
    ax.plot(xSteps[idx, pairIdx], np.minimum(1, meanPValues[idx, pairIdx] + np.ones(len(idx))*epsilon/np.sqrt(xSteps[idx, pairIdx])), label="_Mean + {:}".format(alpha), ls=':', color='red', linewidth=2)

    for i in range(nAttempts):
        ax.plot(xSteps[idx, pairIdx], pValuesResults[idx, pairIdx, i], label="_FastRT #{0}".format(pairIdx), ls='', marker='x')
        #ax[0].plot(xSteps[idx], stochastic[idx, i], label="StochasticRT #{0}".format(i), marker='P')

    if iStep > 1:

        medD = makeFullDescription('med', nPoints[pairIdx, 0], model[idx, pairIdx, :])
        meanD = makeFullDescription('mean', nPoints[pairIdx, 1], model[idx, pairIdx, :])
        lowD = makeFullDescription('low', nPoints[pairIdx, 2], model[idx, pairIdx, :])
        minD = makeFullDescription('min', nPoints[pairIdx, 3], model[idx, pairIdx, :])

        ax.plot((xSteps[nPoints[pairIdx, 0], pairIdx], xSteps[nPoints[pairIdx, 0], pairIdx]), (medianPValues[nPoints[pairIdx, 0], pairIdx], medianPValues[nPoints[pairIdx, 0], pairIdx]), label='M1', color='blue', ls='', marker='X', markersize=15)
        ax.plot((xSteps[nPoints[pairIdx, 1], pairIdx], xSteps[nPoints[pairIdx, 1], pairIdx]), (meanPValues[nPoints[pairIdx, 1], pairIdx], meanPValues[nPoints[pairIdx, 1], pairIdx]), label='M2', color='red', ls='', marker='X', markersize=15)
        ax.plot((xSteps[nPoints[pairIdx, 2], pairIdx], xSteps[nPoints[pairIdx, 2], pairIdx]), (fast[nPoints[pairIdx, 2], pairIdx], fast[nPoints[pairIdx, 2], pairIdx]), label='M3', color='green', ls='', marker='X', markersize=15)
        ax.plot((xSteps[nPoints[pairIdx, 3], pairIdx], xSteps[nPoints[pairIdx, 3], pairIdx]), (fast[nPoints[pairIdx, 3], pairIdx], fast[nPoints[pairIdx, 3], pairIdx]), label='M4', color='orange', ls='', marker='X', markersize=15)

    ax.set_xlabel('Sub-sample size', fontsize=20)
    ax.set_ylabel('KS value', fontsize=20)
    ax.tick_params(axis='x', labelsize=18)
    ax.tick_params(axis='y', labelsize=18)

    ax.legend(fontsize=18)

    #ax[1].title.set_text('PValue of classes {:}'.format(pairsNames[pairIdx]))
    #ax[1].grid()

    idx = range(0, iStep + 1)
    # plot values
    #for i in np.arange(pairedClasses):
    #ax[1].plot(xSteps[idx, pairIdx], fast[idx, pairIdx], label="FastRT #{0}".format(classesPair), ls=':', marker='X')
    #ax[1].plot(xSteps[idx, pairIdx], meanPValues[idx, pairIdx] - fast[idx, pairIdx], label="Lower epsilon - {1} #{0}".format(classesPair, beta), ls=':', marker='X')
    #ax[1].plot(xSteps[idx, pairIdx], fastUp[idx, pairIdx] - fast[idx, pairIdx],
    #           label="Delta epsilon - {1} #{0}".format(classesPair, beta), ls=':', marker='X')
        # ax[0].plot(xSteps[idx], stochastic[idx, i], label="StochasticRT #{0}".format(i), marker='P')

    #ax[1].legend()

    plt.savefig('PValuesFigures\\pValues_par_{0}_{1}.png'.format(taskName, classesPair), format='png')
    plt.close(fig)

    px = 1 / plt.rcParams['figure.dpi']
    fig, ax = plt.subplots(1, 1, sharex=True, tight_layout=True, figsize=(1920 * px, 1280 * px))

    ax.title.set_text('Comparison values of classes pairs targets {:}'.format(pairIdx + 1))
    ax.grid()

    idx = range(0, iStep + 1)

    positionsBp = []
    data = []
    colors = 'bgrcmykw'

    for i in range(pairIdx + 1):
        mins = np.zeros(len(idx))
        maxs = np.zeros(len(idx))
        for k in range(len(idx)):
            mins[k] = np.min(pValuesResults[k, i, :])
            maxs[k] = np.max(pValuesResults[k, i, :])

        ax.plot(range(len(idx)), mins, label="Min #{0}".format(pairsNames[i]), ls='--', color='{:}'.format(colors[i%8]), marker='o')
        ax.plot(range(len(idx)), maxs, label="Max #{0}".format(pairsNames[i]), ls='--', color='{:}'.format(colors[i%8]), marker='o')

    ax.legend()

    plt.savefig('PValuesFigures\\pValues_par_pairs_bp_{0}.png'.format(taskName), format='png')
    plt.close(fig)

    pass