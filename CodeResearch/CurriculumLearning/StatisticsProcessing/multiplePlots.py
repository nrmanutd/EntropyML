import matplotlib.pyplot as plt
import numpy as np
import math


def plot_n_graphs(x, y_arrays, labels=None, x_label="X", y_label="Y",
                  title="Multiple Plots", figsize=(12, 8), colors=None,
                  line_styles=None, line_width=2, grid=True, legend=True,
                  share_y=False, subplots_adjust=None, show=False, fileName=None):
    """
    Рисует N графиков на одном плоте.

    Parameters:
    -----------
    x : array-like
        Массив значений по горизонтальной оси (одинаковый для всех графиков)
    y_arrays : list of array-like
        Список массивов значений по вертикальной оси
    labels : list of str or None
        Список меток для каждого графика. Если None, будут созданы автоматически
    x_label : str
        Подпись горизонтальной оси
    y_label : str
        Подпись вертикальной оси
    title : str
        Заголовок графика
    figsize : tuple
        Размер фигуры (width, height) в дюймах
    colors : list or None
        Список цветов для графиков. Если None, используется цветовая схема matplotlib
    line_styles : list or None
        Список стилей линий. Если None, используется стиль по умолчанию
    line_width : int or float
        Толщина линий
    grid : bool
        Отображать сетку
    legend : bool or str
        Отображать легенду. Если str, то положение легенды ('best', 'upper right', etc.)
    share_y : bool
        Использовать общую ось Y для всех графиков
    subplots_adjust : dict or None
        Параметры для plt.subplots_adjust (например, {'hspace': 0.3, 'wspace': 0.3})
    show : bool
        Показывать график сразу

    Returns:
    --------
    fig : matplotlib.figure.Figure
        Объект фигуры
    ax : matplotlib.axes.Axes or array of Axes
        Объект(ы) осей
    """

    # Проверка входных данных
    if not isinstance(y_arrays, (list, tuple)):
        raise TypeError("y_arrays должен быть списком или кортежем")

    n_graphs = len(y_arrays)

    # Проверка длин массивов
    for i, y in enumerate(y_arrays):
        if len(x) != len(y):
            raise ValueError(f"Массив y_arrays[{i}] имеет длину {len(y)}, "
                             f"но ожидается {len(x)} (как у массива x)")

    # Автоматические метки, если не заданы
    if labels is None:
        labels = [f'График {i + 1}' for i in range(n_graphs)]
    elif len(labels) != n_graphs:
        raise ValueError(f"Количество меток ({len(labels)}) не совпадает с "
                         f"количеством графиков ({n_graphs})")

    # Автоматические цвета, если не заданы
    if colors is None:
        colors = plt.cm.tab10(np.linspace(0, 1, n_graphs))
    elif len(colors) < n_graphs:
        # Если цветов меньше, чем графиков, циклически повторяем
        colors = [colors[i % len(colors)] for i in range(n_graphs)]

    # Автоматические стили линий, если не заданы
    if line_styles is None:
        line_styles = ['-'] * n_graphs
    elif len(line_styles) < n_graphs:
        line_styles = [line_styles[i % len(line_styles)] for i in range(n_graphs)]

    # Создаем фигуру и оси
    fig, ax = plt.subplots(figsize=figsize)

    # Рисуем все графики
    for i in range(n_graphs):
        ax.plot(x, y_arrays[i],
                label=labels[i],
                color=colors[i],
                linestyle=line_styles[i],
                linewidth=line_width,
                alpha=0.8)

    # Настраиваем график
    ax.set_xlabel(x_label, fontsize=12)
    ax.set_ylabel(y_label, fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')

    if grid:
        ax.grid(True, alpha=0.3, linestyle='--')

    if legend:
        if isinstance(legend, str):
            ax.legend(fontsize=10, loc=legend)
        else:
            ax.legend(fontsize=10, loc='best')

    # Настройка отступов, если заданы
    if subplots_adjust:
        plt.subplots_adjust(**subplots_adjust)
    else:
        plt.tight_layout()

    if show:
        plt.show()

    plt.savefig(fileName, format='png', dpi=500, bbox_inches='tight')
    plt.close(fig)

    return fig, ax