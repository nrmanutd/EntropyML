import os

import numpy as np
import matplotlib.pyplot as plt

def visualizeLearningErrors(errors_list, alphas, resultsFolder, taskName):
    errors_list = [np.asarray(e) for e in errors_list]
    alphas = np.asarray(alphas)

    if len(errors_list) != len(alphas):
        raise ValueError("Длины errors_list и alphas должны совпадать.")

    means = np.array([e.mean() for e in errors_list])
    stds = np.array([e.std(ddof=1) if len(e) > 1 else 0.0 for e in errors_list])

    fig, ax = plt.subplots()

    # График: точки + линия, с «усами» стандартного отклонения
    ax.errorbar(
        alphas,
        means,
        yerr=stds,
        fmt='o-',  # точки, соединённые линией
        capsize=5  # «шляпки» у усов
    )

    ax.set_xlabel("Training set fraction")
    ax.set_ylabel("Validation set error")

    ax.grid(True)

    plt.tight_layout()

    if not os.path.exists(resultsFolder):
        os.makedirs(resultsFolder)

    plt.savefig('{:}\\{:}.png'.format(resultsFolder, taskName), format='png')
    plt.close(fig)

def plot_multi_errors_vs_alpha(errors_nested, alphas, labels, resultsFolder, taskName):
    """
    Plot learning curves for multiple classifiers on the same figure.

    Parameters
    ----------
    errors_nested : list of list of array-like
        errors_nested[j][i] is an array of test errors for
        classifier j at training fraction alphas[i].
        Example structure:
        [
            [ [..runs..] for each alpha ],    # classifier 0
            [ [..runs..] for each alpha ],    # classifier 1
            ...
        ]

    alphas : array-like
        Training set fractions (same for all classifiers).

    labels : list of str
        Names of classifiers, used in the legend.
        Must have the same length as errors_nested.

    title : str or None
        Optional plot title.
    """
    alphas = np.asarray(alphas)

    if len(errors_nested) != len(labels):
        raise ValueError("Length of errors_nested must match length of labels.")

    fig, ax = plt.subplots()

    for clf_errors, label in zip(errors_nested, labels):
        if len(clf_errors) != len(alphas):
            raise ValueError(
                f"Each classifier must have errors for all alphas. "
                f"Got {len(clf_errors)} errors, {len(alphas)} alphas for {label}"
            )

        clf_errors = [np.asarray(e) for e in clf_errors]

        means = np.array([e.mean() for e in clf_errors])
        stds  = np.array([e.std(ddof=1) if len(e) > 1 else 0.0 for e in clf_errors])

        ax.errorbar(
            alphas,
            means,
            yerr=stds,
            fmt='o-',
            capsize=5,
            label=label
        )

    ax.set_xlabel("Training Set Fraction (alpha)")
    ax.set_ylabel("Test Error")

    ax.grid(True)
    ax.legend()
    plt.tight_layout()

    if not os.path.exists(resultsFolder):
        os.makedirs(resultsFolder)

    plt.savefig('{:}\\{:}.png'.format(resultsFolder, taskName), format='png')
    plt.close(fig)