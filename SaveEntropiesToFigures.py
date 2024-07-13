from matplotlib import pyplot as plt
import numpy as np

def SaveEntropiesToFigure(bins, simple, conditional, multiEntropy, multiConditionalEntropy, efficiency, mlResults, taskName, nObjects, nFeatures, ct):

    figure = plt.figure()
    px = 1 / plt.rcParams['figure.dpi']
    fig, ax = plt.subplots(3, 1, sharex=True, tight_layout=True, figsize=(1920 * px, 1280 * px))

    permutations = efficiency.shape[1]

    ax[0].plot(bins, efficiency[:, 0], 'C0-.', linewidth=3)
    for iPermutation in np.arange(1, permutations):
        color = 'C{0}'.format(iPermutation%10)
        ax[0].plot(bins, efficiency[:, iPermutation], color)
        ax[1].plot(bins, conditional[:, iPermutation], color)
        ax[2].plot(bins, multiConditionalEntropy[:, iPermutation], color)


    ax[0].title.set_text('Efficiency (simple conditional - multi conditional)')
    ax[0].grid()

    ax[1].plot(bins, simple, 'C0--', linewidth=2)
    ax[1].plot(bins, conditional[:, 0], 'C0-.', linewidth=3)
    ax[1].title.set_text('Simple and conditional entropy')
    ax[1].text(bins[0], 2 / 3 * max(simple) + 1 / 3 * simple[0],
               'Task: {0}\nObjects: {1}\nFeatures: {2}\nUpper: {3:3.2f}'.format(taskName, nObjects, nFeatures,                                                                                 ct * nFeatures))
    ax[1].grid()

    ax[2].plot(bins, multiEntropy, 'C0--', linewidth=2)
    ax[2].plot(bins, multiConditionalEntropy[:, 0], 'C0-.', linewidth=3)

    markers = 'o*'
    mlKeys = list(mlResults.keys())
    for iKey in np.arange(len(mlKeys)):
        ax[2].plot(bins, mlResults[mlKeys[iKey]], 'C{0}:{1}'.format((iKey + 1) % 10, markers[iKey%2]), linewidth=2, label=mlKeys[iKey])

    ax[2].text(bins[0], 2 / 3 * max(multiEntropy) + 1 / 3 * multiEntropy[0], 'Upper: {0:3.2f}'.format(ct))
    ax[2].title.set_text('Simple and conditional multi entropy')
    ax[2].grid()
    ax[2].legend()

    plt.savefig('{0}_{1}_{2}_entropies.png'.format(taskName, nObjects, nFeatures), format='png')
    plt.close(fig)
    plt.close(figure)

    return