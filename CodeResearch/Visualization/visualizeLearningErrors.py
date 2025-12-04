import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

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

def plot_multi_errors_vs_alpha(errors_nested, alphas, labels, resultsFolder, taskName, startIdx = 0):
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

    width_px = 1280
    height_px = 1024
    dpi = 300  # или 150, 200 - чем выше, тем качественнее текст
    fig, ax = plt.subplots(figsize=(width_px/dpi, height_px/dpi), dpi=dpi)

    idx = range(startIdx,len(alphas))
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
            alphas[idx],
            means[idx],
            yerr=stds[idx],
            fmt='o-',
            capsize=4,
            markersize=4,
            linewidth=1,
            capthick=1,
            elinewidth=0.8,
            label=label
        )

    plt.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=6, markerscale=0.3)
    ax.set_xlabel("Epoch number")
    ax.set_ylabel("Test Error")

    ax.grid(True)
    plt.tight_layout()

    if not os.path.exists(resultsFolder):
        os.makedirs(resultsFolder)

    plt.savefig('{:}\\{:}.png'.format(resultsFolder, taskName), format='png', dpi=500, bbox_inches='tight')
    plt.close(fig)


def plot_multi_errors_vs_alpha_std(errors_nested, alphas, labels, resultsFolder, taskName, startIdx=0):
    """
    Plot learning curves for multiple classifiers on the same figure.
    Also creates a separate plot for standard deviations.

    Parameters
    ----------
    errors_nested : list of list of array-like
        errors_nested[j][i] is an array of test errors for
        classifier j at training fraction alphas[i].
    alphas : array-like
        Training set fractions (same for all classifiers).
    labels : list of str
        Names of classifiers, used in the legend.
    resultsFolder : str
        Folder to save the plots.
    taskName : str
        Base name for the saved plots.
    startIdx : int
        Index from which to start plotting alphas.
    """
    alphas = np.asarray(alphas)

    if len(errors_nested) != len(labels):
        raise ValueError("Length of errors_nested must match length of labels.")

    # Создаем папку для результатов если её нет
    if not os.path.exists(resultsFolder):
        os.makedirs(resultsFolder)

    # ==== 1. СОЗДАЁМ ГРАФИК С ОШИБКАМИ (mean ± std) ====
    width_px = 1280
    height_px = 1024
    dpi = 300
    fig1, ax1 = plt.subplots(figsize=(width_px / dpi, height_px / dpi), dpi=dpi)

    idx = range(startIdx, len(alphas))
    colors = []  # Сохраняем цвета для использования во втором графике
    lines = []  # Сохраняем линии для легенды

    for clf_idx, (clf_errors, label) in enumerate(zip(errors_nested, labels)):
        if len(clf_errors) != len(alphas):
            raise ValueError(
                f"Each classifier must have errors for all alphas. "
                f"Got {len(clf_errors)} errors, {len(alphas)} alphas for {label}"
            )

        clf_errors = [np.asarray(e) for e in clf_errors]
        means = np.array([e.mean() for e in clf_errors])
        stds = np.array([e.std(ddof=1) if len(e) > 1 else 0.0 for e in clf_errors])

        # Рисуем график с ошибками и сохраняем цвет
        line = ax1.errorbar(
            alphas[idx],
            means[idx],
            yerr=stds[idx],
            fmt='o-',
            capsize=4,
            markersize=4,
            linewidth=1,
            capthick=1,
            elinewidth=0.8,
            label=label
        )
        colors.append(line[0].get_color())
        lines.append(Line2D([0], [0], color=line[0].get_color(), lw=2, label=label))

    # Настройки первого графика
    ax1.legend(handles=lines, loc='upper left', bbox_to_anchor=(1, 1), fontsize=6, markerscale=0.3)
    ax1.set_xlabel("Epoch number")
    ax1.set_ylabel("Test Error")
    ax1.set_title("Test Error vs Epoch (Mean ± Std)")
    ax1.grid(True)

    # Сохраняем первый график
    plt.tight_layout()
    error_plot_path = os.path.join(resultsFolder, f"{taskName}_errors.png")
    plt.savefig(error_plot_path, format='png', dpi=500, bbox_inches='tight')
    plt.close(fig1)

    # ==== 2. СОЗДАЁМ ГРАФИК СО СТАНДАРТНЫМИ ОТКЛОНЕНИЯМИ ====
    fig2, ax2 = plt.subplots(figsize=(width_px / dpi, height_px / dpi), dpi=dpi)

    for clf_idx, (clf_errors, label) in enumerate(zip(errors_nested, labels)):
        clf_errors = [np.asarray(e) for e in clf_errors]
        stds = np.array([e.std(ddof=1) if len(e) > 1 else 0.0 for e in clf_errors])

        # Используем ТОТ ЖЕ ЦВЕТ, что и в первом графике
        ax2.plot(
            alphas[idx],
            stds[idx],
            'o-',
            color=colors[clf_idx],
            markersize=4,
            linewidth=1,
            label=label
        )

    # Настройки второго графика
    ax2.legend(handles=lines, loc='upper left', bbox_to_anchor=(1, 1), fontsize=6, markerscale=0.3)
    ax2.set_xlabel("Epoch number")
    ax2.set_ylabel("Standard Deviation (Std)")
    ax2.set_title("Standard Deviation of Test Error vs Epoch")
    ax2.grid(True)

    # Сохраняем второй график
    plt.tight_layout()
    std_plot_path = os.path.join(resultsFolder, f"{taskName}_std.png")
    plt.savefig(std_plot_path, format='png', dpi=500, bbox_inches='tight')
    plt.close(fig2)