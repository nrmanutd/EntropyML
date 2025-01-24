from matplotlib import pyplot as plt
import numpy as np

def visualizePValues(data):
    xSteps = data['steps']
    targetResults = data['targetResults']
    pValuesResults = data['pValuesResults']
    classesPair = data['classes']
    fast = data['fast']
    iStep = data['step']
    taskName = data['taskName']
    pairIdx = data['pairIndex']
    nAttempts = data['nAttempts']
    pairsNames = data['names']
    model = data['model']

    px = 1 / plt.rcParams['figure.dpi']
    fig, ax = plt.subplots(2, 1, sharex=True, tight_layout=True, figsize=(1920 * px, 1280 * px))

    pairedClasses = fast.shape[1]

    ax[0].title.set_text('Comparison values of classes {:}'.format(pairIdx))
    ax[0].grid()

    idx = range(0, iStep + 1)
    # plot values
    #for i in np.arange(pairedClasses):
    ax[0].plot(xSteps[idx], targetResults[idx, pairIdx], label="FastRT #{0}".format(pairIdx), ls=':', marker='X')
    ax[0].plot(xSteps[idx], model[idx, pairIdx, 0], label="XGBoost", ls=':', marker='o')
    ax[0].plot(xSteps[idx], model[idx, pairIdx, 1], label="NN", ls=':', marker='x')
    for i in range(nAttempts):
        ax[0].plot(xSteps[idx], pValuesResults[idx, pairIdx, i], label="_FastRT #{0}".format(pairIdx), ls='', marker='x')
        #ax[0].plot(xSteps[idx], stochastic[idx, i], label="StochasticRT #{0}".format(i), marker='P')

    maxValue = np.zeros(len(idx))
    for i in range(len(idx)):
        maxValue[i] = max(pValuesResults[i, pairIdx, :])

    ax[0].plot(xSteps[idx], targetResults[idx, pairIdx] - maxValue, label="Delta #{0}".format(pairIdx), ls='-', marker='o')

    ax[0].legend()

    ax[1].title.set_text('PValue of classes {:}'.format(pairedClasses))
    ax[1].grid()

    idx = range(0, iStep + 1)
    # plot values
    #for i in np.arange(pairedClasses):
    ax[1].plot(xSteps[idx], fast[idx, pairIdx], label="FastRT #{0}".format(classesPair), ls=':', marker='X')
        # ax[0].plot(xSteps[idx], stochastic[idx, i], label="StochasticRT #{0}".format(i), marker='P')

    ax[1].legend()

    plt.savefig('PValuesFigures\\pValues_{0}_{1}.png'.format(taskName, classesPair), format='png')
    plt.close(fig)

    px = 1 / plt.rcParams['figure.dpi']
    fig, ax = plt.subplots(1, 1, sharex=True, tight_layout=True, figsize=(1920 * px, 1280 * px))

    ax.title.set_text('Comparison values of classes pairs targets {:}'.format(pairIdx + 1))
    ax.grid()

    idx = range(0, iStep + 1)
    # plot values
    for i in range(pairIdx + 1):
        ax.plot(xSteps[idx], targetResults[idx, i], label="FastRT pair #{0}".format(pairsNames[i]), ls=':',
                   marker='x')
        # ax[0].plot(xSteps[idx], stochastic[idx, i], label="StochasticRT #{0}".format(i), marker='P')

    ax.legend()

    plt.savefig('PValuesFigures\\pValues_pairs_{0}.png'.format(taskName), format='png')
    plt.close(fig)

    pass