import math

from matplotlib import pyplot as plt
import numpy as np
from numpy.linalg import linalg as LA


def SaveRademacherResults(data, task):
    px = 1 / plt.rcParams['figure.dpi']
    fig, ax = plt.subplots(3, 1, sharex=True, tight_layout=True, figsize=(1920 * px, 1280 * px))

    xLabels = data['xLabels']
    accuracy = data['accuracy']
    rad = data['rad']
    upperRad = data['upperRad']
    alpha = data['alpha']

    radA = data['radA']
    upperRadA = data['upperRadA']

    modelSigma = data['modelSigma']

    mcDiarmid = data['mcDiarmid']
    lastIdx = len(upperRad) - 1

    ax[0].title.set_text('Comparison values')
    ax[0].grid()

    #plot values
    ax[0].plot(xLabels, rad, label="Rad")
    ax[0].plot(xLabels, upperRad, label="Upper Rad")
    ax[0].plot(xLabels, radA, label="RadA", ls=':')
    ax[0].plot(xLabels, upperRadA, label="Upper RadA", ls=':')
    ax[0].plot(xLabels, 1 - accuracy, label="Loss", ls='-.')
    ax[0].plot(xLabels, accuracy, label="Accuracy", ls='-.')
    ax[0].plot(xLabels, modelSigma, label="ModelSigma", ls='-.')
    ax[0].plot(xLabels, rad + mcDiarmid, label="rad+m {0}".format(task['prob']), ls=':')
    ax[0].plot(xLabels, rad - mcDiarmid, label="rad-m {0}".format(task['prob']), ls=':')
    ax[0].plot(xLabels, accuracy + modelSigma, label="+s")
    ax[0].plot(xLabels, accuracy - modelSigma, label="-s")

    # plot logs
    ax[1].title.set_text('Comparison values Logs')
    ax[1].grid()

    iStart = math.floor(0.4 * len(rad))
    idx = np.arange(iStart, len(rad))
    lrad = np.log(rad)[idx]

    deltaLogLoss = sum(np.log(1 - accuracy)[idx] - lrad) / len(lrad)
    deltaLogAccuracy = sum(np.log(1 / accuracy)[idx] - lrad) / len(lrad)
    deltaAccuracy = sum(accuracy[idx] - lrad) / len(lrad)
    deltaModelSigma = sum(np.log(modelSigma)[idx] - lrad) / len(lrad)

    ax[1].plot(xLabels, np.log(rad), label="Log Rad", linewidth=2)
    ax[1].plot(xLabels, np.log(upperRad),
               label="Log Upper Rad", linewidth=2)
    ax[1].plot(xLabels, np.log(radA), label="Log RadA", linewidth=2, ls=':')
    ax[1].plot(xLabels, np.log(upperRadA),
               label="Log Upper RadA", linewidth=2, ls=':')
    ax[1].plot(xLabels, np.log(1 - accuracy) - deltaLogLoss,
             label="Log Loss", ls='-.', linewidth=2)
    ax[1].plot(xLabels, np.log(1 / accuracy) - deltaLogAccuracy,
             label="Log Accuracy", ls='-.', linewidth=2)
    ax[1].plot(xLabels, accuracy - deltaAccuracy, label="Accuracy",
             ls='-.', linewidth=2)
    ax[1].plot(xLabels, np.log(modelSigma) - deltaModelSigma,
             label="Log ModelSigma", ls='-.', linewidth=2)
    # plt.plot(xNumberOfObjects, avgDistributions + mcDiarmids, label="+m", ls=':')
    # plt.plot(xNumberOfObjects, avgDistributions - mcDiarmids, label="-m", ls=':')
    # plt.plot(xNumberOfObjects, avgDistributions + sigmas, label="+s")
    # plt.plot(xNumberOfObjects, avgDistributions - sigmas, label="-s")

    #plot power law
    ax[2].title.set_text('Power law dependency')
    ax[2].grid()
    ax[2].plot(xLabels, alpha, label="Upper Rad alpha", linewidth=2)

    ax[2].legend()
    ax[1].legend()
    ax[0].legend()

    plt.savefig('Figures\\{0}_{1}_{2}_{7}_a{3}_ma{4}_rs{5}_{6}.png'.format(task['name'], task['nObjects'], task['nFeatures'], task['nAttempts'], task['modelAttempts'],
                                                                       task['nRadSets'], task['totalPoints'], task['nClasses']), format='png')
    plt.close(fig)

    fig, ax = plt.subplots(4, 1, sharex=True, tight_layout=True, figsize=(1920 * px, 1280 * px))
    ax[0].title.set_text('Comparison values')
    ax[0].grid()
    ax[1].grid()
    ax[2].grid()
    ax[3].grid()

    ax[0].title.set_text('Rad values')

    ax[0].plot(xLabels, rad, label="Rad", linewidth=2)
    ax[0].plot(xLabels, upperRad,
               label="Upper Rad", linewidth=2)
    ax[0].plot(xLabels, radA, label="RadA", linewidth=2, ls=':')
    ax[0].plot(xLabels, upperRadA,
               label="Upper RadA", linewidth=2, ls=':')

    ax[1].title.set_text('Log Rad values')
    ax[1].plot(xLabels, np.log(rad), label="Log Rad", linewidth=2)
    ax[1].plot(xLabels, np.log(upperRad),
               label="Log Upper Rad", linewidth=2)
    ax[1].plot(xLabels, np.log(radA), label="Log RadA", linewidth=2, ls=':')
    ax[1].plot(xLabels, np.log(upperRadA),
               label="Log Upper RadA", linewidth=2, ls=':')

    iStart = math.floor(0.4 * len(rad))
    idx = np.arange(iStart, len(rad))
    d1 = LA.norm(rad[idx] - radA[idx])
    d2 = LA.norm(upperRad[idx] - upperRadA[idx])
    d3 = LA.norm(np.log(rad[idx] / radA[idx]))
    d4 = LA.norm(np.log(upperRad[idx] / upperRadA[idx]))

    ax[2].title.set_text('Rad values deltas')
    ax[2].plot(xLabels, rad - radA, label="Rad - A delta {:.2f}".format(d1), linewidth=2)
    ax[2].plot(xLabels, upperRad - upperRadA, label="Upper Rad - A delta {:.2f}".format(d2), linewidth=2)

    ax[3].title.set_text('Rad values Logs deltas')
    ax[3].plot(xLabels, np.log(rad / radA), label="Log Rad - A delta {:.2f}".format(d3), linewidth=2)
    ax[3].plot(xLabels, np.log(upperRad / upperRadA), label="Log Upper Rad - A delta {:.2f}".format(d4), linewidth=2)

    ax[3].legend()
    ax[2].legend()
    ax[1].legend()
    ax[0].legend()

    plt.savefig(
        'Figures\\Rads_{0}_{1}_{2}_{7}_a{3}_ma{4}_rs{5}_{6}.png'.format(task['name'], task['nObjects'], task['nFeatures'],
                                                                   task['nAttempts'], task['modelAttempts'],
                                                                   task['nRadSets'], task['totalPoints'],
                                                                   task['nClasses']), format='png')
    plt.close(fig)