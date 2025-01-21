import math

from matplotlib import pyplot as plt
import numpy as np
from numpy.linalg import linalg as LA


def VisualizeAndSaveDistributionDeltas(data, task):
    px = 1 / plt.rcParams['figure.dpi']
    fig, ax = plt.subplots(1, 1, sharex=True, tight_layout=True, figsize=(1920 * px, 1280 * px))

    xLabels = data['xLabels']
    accuracy = data['accuracy']

    rad = data['rad']
    upperRad = data['upperRad']
    modelSigma = data['modelSigma']

    mcDiarmid = data['mcDiarmid']

    pairedLines = rad.shape[0]

    ax.title.set_text('Comparison values (pairs {0})'.format(pairedLines))
    ax.grid()

    #plot values
    for i in np.arange(pairedLines):
        ax.plot(xLabels, rad[i, :], label="Rad(RT) #{0}".format(i))
        ax.plot(xLabels, upperRad[i, :], label="Upper Rad(Fast) #{0}".format(i), ls=':')

    nModels = accuracy.shape[0]
    #for i in np.arange(nModels):
    #    ax.plot(xLabels, accuracy[i, :], label="Accuracy #{0}".format(i), ls='-.')
    #    ax.plot(xLabels, modelSigma[i, :], label="ModelSigma #{0}".format(i), ls='-.')
    #    ax.plot(xLabels, 1 - accuracy[i, :], label="Loss #{0}".format(i), ls='-.')
    #    ax.plot(xLabels, accuracy[i, :] + modelSigma[i, :], label="+s #{0}".format(i))
    #    ax.plot(xLabels, accuracy[i, :] - modelSigma[i, :], label="-s #{0}".format(i))

    ax.legend()

    plt.savefig('DeltasFigures\\values_{0}_{1}_{2}_{7}_a{3}_ma{4}_rs{5}_{6}.png'.format(task['name'], task['nObjects'],
                                                                                 task['nFeatures'], task['nAttempts'],
                                                                                 task['modelAttempts'],
                                                                                 task['nRadSets'], task['totalPoints'],
                                                                                 task['nClasses']), format='png')
    plt.close(fig)

    fig, ax = plt.subplots(1, 1, sharex=True, tight_layout=True, figsize=(1920 * px, 1280 * px))

    # plot logs
    ax.title.set_text('Comparison values Logs (pairs {0})'.format(pairedLines))
    ax.grid()

    for i in np.arange(pairedLines):
        ax.plot(xLabels, np.log(rad[i, :]), label="Log Rad(RT) #{0}".format(i), linewidth=2)
        ax.plot(xLabels, np.log(upperRad[i, :]),
               label="Log Upper Rad(Fast) #{0}".format(i), linewidth=2, ls=':')

    #for i in np.arange(nModels):
        #ax.plot(xLabels, np.log(accuracy[i, :]), label="Log Accuracy #{0}".format(i), ls='-.')
        #ax.plot(xLabels, accuracy[i, :], label="Accuracy #{0}".format(i), ls='-.')
        #ax.plot(xLabels, np.log(modelSigma[i, :]), label="Log ModelSigma #{0}".format(i), ls='-.')
        #ax.plot(xLabels, np.log(1 - accuracy[i, :]), label="Log Loss #{0}".format(i), ls='-.')

    ax.legend()

    plt.savefig('DeltasFigures\\logs_{0}_{1}_{2}_{7}_a{3}_ma{4}_rs{5}_{6}.png'.format(task['name'], task['nObjects'], task['nFeatures'], task['nAttempts'], task['modelAttempts'],
                                                                       task['nRadSets'], task['totalPoints'], task['nClasses']), format='png')
    plt.close(fig)
