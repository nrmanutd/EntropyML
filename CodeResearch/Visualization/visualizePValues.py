from matplotlib import pyplot as plt
import numpy as np

def visualizePValues(data):
    xSteps = data['steps']
    stochastic = data['stochastic']
    fast = data['fast']
    iStep = data['step']
    taskName = data['taskName']

    px = 1 / plt.rcParams['figure.dpi']
    fig, ax = plt.subplots(1, 1, sharex=True, tight_layout=True, figsize=(1920 * px, 1280 * px))

    pairedClasses = stochastic.shape[1]

    ax.title.set_text('Comparison values of classes (pairs {0})'.format(pairedClasses))
    ax.grid()

    idx = range(0, iStep + 1)
    # plot values
    for i in np.arange(pairedClasses):
        ax.plot(xSteps[idx], stochastic[idx, i], label="_StochasticRT #{0}".format(i))
        ax.plot(xSteps[idx], fast[idx, i], label="_FastRT #{0}".format(i), ls=':')

    ax.legend()

    plt.savefig('PValuesFigures\\pValues_{0}_{1}.png'.format(taskName, iStep), format='png')
    plt.close(fig)

    pass