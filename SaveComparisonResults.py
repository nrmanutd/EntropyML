from matplotlib import pyplot as plt
import numpy as np

def SaveComparisonResults(bins, comparisonResults, mlResults, taskName, nObjects,
                          nFeatures):

    figure = plt.figure()
    px = 1 / plt.rcParams['figure.dpi']
    fig, ax = plt.subplots(2, 1, sharex=True, tight_layout=True, figsize=(1920 * px, 1280 * px))

    permutations = comparisonResults.shape[1]
    for iPermutation in np.arange(permutations):
        color = 'C{0}'.format(iPermutation % 10)
        ax[0].plot(bins, comparisonResults[:, iPermutation, 0], color)
        ax[1].plot(bins, comparisonResults[:, iPermutation, 1], color)

    ax[0].title.set_text('Comparison Bhattacharyya (new target to baseline)')
    ax[0].grid()

    markers = 'o*'
    mlKeys = list(mlResults.keys())
    for iKey in np.arange(len(mlKeys)):
        ax[0].plot(bins, mlResults[mlKeys[iKey]][:, 0], 'C{0}:{1}'.format((iKey + 1) % 10, markers[iKey % 2]), linewidth=2,
                   label=mlKeys[iKey])
        ax[1].plot(bins, mlResults[mlKeys[iKey]][:, 1], 'C{0}:{1}'.format((iKey + 1) % 10, markers[iKey % 2]), linewidth=2,
                   label=mlKeys[iKey])

    ax[1].title.set_text('Comparison Cross entropy (new target to baseline)')
    ax[1].grid()
    ax[1].legend()
    ax[0].legend()

    plt.savefig('{0}_{1}_{2}_comparisons.png'.format(taskName, nObjects, nFeatures), format='png')
    plt.close(fig)
    plt.close(figure)

    return