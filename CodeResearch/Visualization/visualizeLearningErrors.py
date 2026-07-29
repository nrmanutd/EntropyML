import math
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


def plot_multi_errors_vs_alpha_std(errors_nested, alphas, labels, resultsFolder, taskName, markersCount, startIdx=0, yLabel='Test Accuracy', title='Test Accuracy vs Epoch (Mean ± Std)'):
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
    width_px = 1920
    height_px = 1024
    dpi = 300
    fig1, ax1 = plt.subplots(figsize=(width_px / dpi, height_px / dpi), dpi=dpi)

    idx = range(startIdx, len(alphas))
    colors = []  # Сохраняем цвета для использования во втором графике
    lines = []  # Сохраняем линии для легенды

    bright_colors = [
        '#FF0000', '#00CC00', '#0000FF', '#FF00FF', '#FFFF00', '#00FFFF',  # RGB основные
        '#FF8000', '#FF0080', '#80FF00', '#0080FF', '#8000FF', '#00FF80',  # промежуточные
        '#FF4000', '#FF0040', '#40FF00', '#0040FF', '#4000FF', '#00FF40',  # еще больше
        '#FFA500', '#FF1493', '#32CD32', '#1E90FF', '#8A2BE2', '#00CED1',  # названия цветов
    ]

    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h', 'H', '+', 'x', 'd']

    markersCount = min(len(markers), markersCount)

    for clf_idx, (clf_errors, label) in enumerate(zip(errors_nested, labels)):
        if len(clf_errors) != len(alphas):
            raise ValueError(
                f"Each classifier must have errors for all alphas. "
                f"Got {len(clf_errors)} errors, {len(alphas)} alphas for {label}"
            )

        clf_errors = [np.asarray(e) for e in clf_errors]
        means = np.array([e.mean() for e in clf_errors])
        stds = np.array([e.std(ddof=1)/math.sqrt(len(e)) if len(e) > 1 else 0.0 for e in clf_errors])

        color = bright_colors[clf_idx % len(bright_colors)]
        marker = markers[clf_idx%markersCount]

        # Рисуем график с ошибками и сохраняем цвет
        line = ax1.errorbar(
            alphas[idx],
            means[idx],
            yerr=stds[idx],
            fmt= f'{marker}-',
            capsize=2,
            markersize=2,
            linewidth=1,
            capthick=1,
            elinewidth=0.8,
            label=label,
            color=color
        )
        #colors.append(line[0].get_color())
        colors.append(color)
        lines.append(Line2D([0], [0], color=color, lw=1, label=label, marker=marker))

    # Настройки первого графика
    ax1.legend(handles=lines, loc='upper left', bbox_to_anchor=(1, 1), fontsize=6, markerscale=0.3)
    ax1.set_xlabel("Epoch number")
    ax1.set_ylabel(yLabel)
    ax1.set_title(title)
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
    ax2.set_title("Standard Deviation of Test Accuracy vs Epoch")
    ax2.grid(True)

    # Сохраняем второй график
    plt.tight_layout()
    std_plot_path = os.path.join(resultsFolder, f"{taskName}_std.png")
    plt.savefig(std_plot_path, format='png', dpi=500, bbox_inches='tight')
    plt.close(fig2)

import os
from typing import List, Sequence, Union, Optional

import numpy as np
import matplotlib.pyplot as plt


def save_learning_curve_figure(
    method_epoch_runs: Sequence[Sequence[Sequence[float]]],
    labels: Sequence[str],
    main_method: Union[int, str],
    save_dir: str,
    file_name: str,
    title: str,
    *,
    accuracy_in_01: bool = True,
    dpi: int = 300,
    figsize=(7.0, 5.0),
    main_band_alpha: float = 0.18,
    show_legend: bool = True,
    legend_loc: str = "best",
    add_grid: bool = True,
) -> str:
    """
    Draws and saves one learning-curve figure.

    Parameters
    ----------
    method_epoch_runs :
        Data for all methods.

        Expected structure:
            method_epoch_runs[m][e] = array-like of accuracies
            for method m at epoch e across runs.

        In other words:
            - outer level: methods
            - second level: epochs
            - third level: runs

        Example shape in conceptual form:
            n_methods x n_epochs x n_runs

        For each method, the number of epochs must be the same.
        For each epoch within a method, the number of runs should also
        be the same (typically 15, but can be 10, etc.).

    labels :
        Names of methods to display in the plot.
        Must have the same length as method_epoch_runs.

    main_method :
        Either:
            - integer index of the main method, or
            - string label of the main method.

        This method will be plotted with a thicker line and with
        a shaded mean ± std band.

    save_dir :
        Directory where the figure will be saved.

    file_name :
        Name of the output image file, e.g. "svhn_20.png".

    title :
        Plot title, e.g. "SVHN, budget 20%".

    accuracy_in_01 :
        If True, assumes accuracies are in [0, 1] and converts them
        to percentages.
        If False, assumes accuracies are already in percent.

    dpi :
        Resolution of the saved image.

    figsize :
        Figure size passed to matplotlib.

    main_band_alpha :
        Transparency of the shaded std band for the main method.

    show_legend :
        Whether to display legend.

    legend_loc :
        Legend location.

    add_grid :
        Whether to draw a light grid.

    Returns
    -------
    saved_path : str
        Full path to the saved image.
    """

    if len(method_epoch_runs) != len(labels):
        raise ValueError(
            "method_epoch_runs and labels must have the same length."
        )

    if len(method_epoch_runs) == 0:
        raise ValueError("method_epoch_runs must not be empty.")

    # Resolve main method index
    if isinstance(main_method, str):
        if main_method not in labels:
            raise ValueError(
                f"main_method='{main_method}' is not present in labels."
            )
        main_idx = labels.index(main_method)
    else:
        main_idx = int(main_method)
        if not (0 <= main_idx < len(labels)):
            raise ValueError("main_method index is out of range.")

    # Convert all methods to numpy arrays and validate shapes
    processed = []
    n_epochs_expected: Optional[int] = None

    for method_idx, method_data in enumerate(method_epoch_runs):
        arr = np.asarray(method_data, dtype=np.float64)

        if arr.ndim != 2:
            raise ValueError(
                f"Method '{labels[method_idx]}' must be a 2D structure "
                f"with shape [n_epochs, n_runs], but got shape {arr.shape}."
            )

        n_epochs, n_runs = arr.shape

        if n_epochs == 0 or n_runs == 0:
            raise ValueError(
                f"Method '{labels[method_idx]}' has an empty data array."
            )

        if not np.all(np.isfinite(arr)):
            raise ValueError(
                f"Method '{labels[method_idx]}' contains NaN or inf values."
            )

        if n_epochs_expected is None:
            n_epochs_expected = n_epochs
        elif n_epochs != n_epochs_expected:
            raise ValueError(
                "All methods must have the same number of epochs. "
                f"Expected {n_epochs_expected}, got {n_epochs} for "
                f"'{labels[method_idx]}'."
            )

        if accuracy_in_01:
            arr = arr * 100.0

        processed.append(arr)

    epochs = np.arange(1, n_epochs_expected + 1)

    fig, ax = plt.subplots(figsize=figsize)

    for method_idx, (label, arr) in enumerate(zip(labels, processed)):
        # arr shape: [n_epochs, n_runs]
        mean_curve = arr.mean(axis=1)
        std_curve = arr.std(axis=1, ddof=1) if arr.shape[1] > 1 else np.zeros(arr.shape[0])

        if method_idx == main_idx:
            line, = ax.plot(
                epochs,
                mean_curve,
                label=label,
                linewidth=2.8,
            )

            ax.fill_between(
                epochs,
                mean_curve - std_curve,
                mean_curve + std_curve,
                alpha=main_band_alpha,
                color=line.get_color(),
                linewidth=0.0,
            )
        else:
            ax.plot(
                epochs,
                mean_curve,
                label=label,
                linewidth=1.5,
            )

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Test Accuracy (%)")
    ax.set_title(title)

    if add_grid:
        ax.grid(True, alpha=0.25)

    if show_legend:
        ax.legend(frameon=False, loc=legend_loc)

    fig.tight_layout()

    os.makedirs(save_dir, exist_ok=True)
    saved_path = os.path.join(save_dir, file_name)
    fig.savefig(saved_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    return saved_path