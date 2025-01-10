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
        ax.plot(xLabels, rad[i, :], label="Rad #{0}".format(i))
        ax.plot(xLabels, upperRad[i, :], label="Upper Rad #{0}".format(i), ls=':')

    ax.plot(xLabels, accuracy, label="Accuracy", ls='-.')
    ax.plot(xLabels, modelSigma, label="ModelSigma", ls='-.')
    ax.plot(xLabels, 1 - accuracy, label="Loss", ls='-.')
    ax.plot(xLabels, accuracy + modelSigma, label="+s")
    ax.plot(xLabels, accuracy - modelSigma, label="-s")
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
        ax.plot(xLabels, np.log(rad[i, :]), label="Log Rad #{0}".format(i), linewidth=2)
        ax.plot(xLabels, np.log(upperRad[i, :]),
               label="Log Upper Rad #{0}".format(i), linewidth=2, ls=':')

    ax.plot(xLabels, np.log(1 - accuracy), label="Log Loss", ls='-.', linewidth=2)
    ax.plot(xLabels, np.log(accuracy), label="Log Accuracy", ls='-.', linewidth=2)
    ax.plot(xLabels, accuracy, label="Accuracy", ls='-.', linewidth=2)
    ax.plot(xLabels, np.log(modelSigma), label="Log ModelSigma", ls='-.', linewidth=2)

    ax.legend()

    plt.savefig('DeltasFigures\\logs_{0}_{1}_{2}_{7}_a{3}_ma{4}_rs{5}_{6}.png'.format(task['name'], task['nObjects'], task['nFeatures'], task['nAttempts'], task['modelAttempts'],
                                                                       task['nRadSets'], task['totalPoints'], task['nClasses']), format='png')
    plt.close(fig)
