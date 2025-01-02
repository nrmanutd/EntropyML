from matplotlib import pyplot as plt
import numpy as np

def SaveRademacherResults(data, task):
    figure = plt.figure()
    px = 1 / plt.rcParams['figure.dpi']
    fig, ax = plt.subplots(2, 1, sharex=True, tight_layout=True, figsize=(1920 * px, 1280 * px))

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
    ax[0].plot(xLabels, upperRad - upperRad[lastIdx] + rad[lastIdx], label="Upper Rad")
    # plt.plot(xNumberOfObjects, accuracy_full, label="Accuracy full", ls='-.')
    ax[0].plot(xLabels, (accuracy[lastIdx] - accuracy) + rad[lastIdx], label="Loss", ls='-.')
    ax[0].plot(xLabels, accuracy - accuracy[lastIdx] + rad[lastIdx], label="Accuracy", ls='-.')
    ax[0].plot(xLabels, modelSigma - modelSigma[lastIdx] + rad[lastIdx], label="ModelSigma", ls='-.')
    ax[0].plot(xLabels, rad + mcDiarmid, label="rad+m {0}".format(task['prob']), ls=':')
    ax[0].plot(xLabels, rad - mcDiarmid, label="rad-m {0}".format(task['prob']), ls=':')
    # plt.plot(xNumberOfObjects, avgDistributions + sigmas, label="+s")
    # plt.plot(xNumberOfObjects, avgDistributions - sigmas, label="-s")

    # plot logs
    ax[1].title.set_text('Comparison values Logs')
    ax[1].grid()

    ax[1].plot(xLabels, np.log(rad), label="Log Rad", linewidth=2)
    ax[1].plot(xLabels, np.log(upperRad / upperRad[lastIdx] * rad[lastIdx]),
             label="Log Upper Rad", linewidth=2)
    # plt.plot(xNumberOfObjects, accuracy_full, label="Accuracy full", ls='-.')
    ax[1].plot(xLabels, np.log((1 - accuracy) / (1 - accuracy[lastIdx]) * rad[lastIdx]),
             label="Log Loss", ls='-.', linewidth=2)
    ax[1].plot(xLabels, np.log(1 / accuracy * accuracy[lastIdx] * rad[lastIdx]),
             label="Log Accuracy", ls='-.', linewidth=2)
    ax[1].plot(xLabels, accuracy[lastIdx] - accuracy + np.log(rad[lastIdx]), label="Accuracy",
             ls='-.', linewidth=2)
    ax[1].plot(xLabels, np.log(modelSigma / modelSigma[lastIdx] * rad[lastIdx]),
             label="Log ModelSigma", ls='-.', linewidth=2)
    # plt.plot(xNumberOfObjects, avgDistributions + mcDiarmids, label="+m", ls=':')
    # plt.plot(xNumberOfObjects, avgDistributions - mcDiarmids, label="-m", ls=':')
    # plt.plot(xNumberOfObjects, avgDistributions + sigmas, label="+s")
    # plt.plot(xNumberOfObjects, avgDistributions - sigmas, label="-s")

    ax[1].legend()
    ax[0].legend()

    plt.savefig('Figures\\{0}_{1}_{2}_a{3}_ma{4}.png'.format(task['name'], task['nObjects'], task['nFeatures'],
                                                    task['nAttempts'], task['modelAttempts']), format='png')
    plt.close(fig)
    plt.close(figure)

