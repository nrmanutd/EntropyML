import math

from matplotlib import pyplot as plt
import numpy as np

def SaveRademacherResults(data, task):
    px = 1 / plt.rcParams['figure.dpi']
    fig, ax = plt.subplots(3, 1, sharex=True, tight_layout=True, figsize=(1920 * px, 1280 * px))

    xLabels = data['xLabels']
    accuracy = data['accuracy']
    rad = data['rad']
    upperRad = data['upperRad']
    modelSigma = data['modelSigma']
    mcDiarmid = data['mcDiarmid']
    lastIdx = len(upperRad) - 1

    ax[0].title.set_text('Comparison values')
    ax[0].grid()

    #plot values
    ax[0].plot(xLabels, rad, label="Rad")
    ax[0].plot(xLabels, upperRad, label="Upper Rad")
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
    ax[2].plot(xLabels, (np.log(upperRad) + np.log(xLabels) - 0.5 * np.log(np.log(2 * 2 * xLabels)))/np.log(xLabels),
               label="Log Upper Rad", linewidth=2)

    ax[2].legend()
    ax[1].legend()
    ax[0].legend()

    plt.savefig('Figures\\{0}_{1}_{2}_{7}_a{3}_ma{4}_rs{5}_{6}.png'.format(task['name'], task['nObjects'], task['nFeatures'], task['nAttempts'], task['modelAttempts'],
                                                                       task['nRadSets'], task['totalPoints'], task['nClasses']), format='png')
    plt.close(fig)

