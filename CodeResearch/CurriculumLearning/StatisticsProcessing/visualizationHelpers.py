import matplotlib.pyplot as plt
import numpy as np


def plot_with_confidence_intervals(x, y1, y2,
                                   std1=None, std2=None,
                                   sigma_coeffs=None,
                                   labels=None,
                                   x_label="X",
                                   y_label="Y",
                                   title="Data with Confidence Intervals",
                                   colors=None,
                                   line_styles=None, fileName=None):
    """
    Рисует два ряда с доверительными интервалами.

    Parameters:
    -----------
    x : array-like
        Массив значений по горизонтальной оси
    y1, y2 : array-like
        Два массива значений (одинаковой длины)
    std1, std2 : array-like or None
        Стандартные отклонения для каждого элемента ряда.
        Если None, доверительные интервалы не отрисовываются.
    sigma_coeffs : list or array-like
        Коэффициенты для умножения на стандартное отклонение.
        По умолчанию [1, 2, 3] (1σ, 2σ, 3σ).
    labels : tuple or list
        Лейблы для двух рядов в формате (label1, label2)
    x_label : str
        Подпись горизонтальной оси
    y_label : str
        Подпись вертикальной оси
    title : str
        Заголовок графика
    colors : list
        Цвета для двух рядов [color1, color2]
    line_styles : list
        Стили линий для доверительных интервалов

    Returns:
    --------
    fig, ax : matplotlib figure and axes objects
    """

    # Проверка длин массивов
    if not (len(x) == len(y1) == len(y2)):
        raise ValueError("Массивы x, y1, y2 должны иметь одинаковую длину")

    if std1 is not None and len(std1) != len(y1):
        raise ValueError("std1 должен иметь ту же длину, что и y1")

    if std2 is not None and len(std2) != len(y2):
        raise ValueError("std2 должен иметь ту же длину, что и y2")

    # Значения по умолчанию
    if sigma_coeffs is None:
        sigma_coeffs = [1, 2, 3]

    if labels is None:
        labels = ("Series 1", "Series 2")

    if colors is None:
        colors = ['blue', 'red']

    if line_styles is None:
        # Стили линий: от менее прозрачных к более прозрачным
        line_styles = ['--', ':', '-.']

    # Создаем график
    fig, ax = plt.subplots(figsize=(12, 7))

    # Функция для отрисовки одного ряда с доверительными интервалами
    def plot_series(x, y, std, color, label, series_idx):
        # Основная линия
        ax.plot(x, y, color=color, linewidth=3, label=label, zorder=10)

        # Если есть стандартные отклонения, рисуем доверительные интервалы
        if std is not None:
            alpha_values = [0.4, 0.2, 0.1]  # Прозрачность для разных сигм

            for i, (coeff, alpha) in enumerate(zip(sigma_coeffs, alpha_values)):
                # Выбираем стиль линии (циклически)
                line_style = line_styles[i % len(line_styles)]

                # Верхняя граница
                y_upper = y + coeff * std
                # Нижняя граница
                y_lower = y - coeff * std

                # Рисуем границы доверительного интервала
                ax.plot(x, y_upper, color=color, linestyle=line_style,
                        alpha=alpha, linewidth=1.5,
                        label=f'{label} ±{coeff}σ' if series_idx == 0 and i == 0 else '')
                ax.plot(x, y_lower, color=color, linestyle=line_style,
                        alpha=alpha, linewidth=1.5)

                # Заливка между границами
                ax.fill_between(x, y_lower, y_upper, color=color,
                                alpha=alpha / 3, linewidth=0)

    # Рисуем оба ряда
    plot_series(x, y1, std1, colors[0], labels[0], 0)
    plot_series(x, y2, std2, colors[1], labels[1], 1)

    # Настраиваем график
    ax.set_xlabel(x_label, fontsize=12)
    ax.set_ylabel(y_label, fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(fontsize=11, loc='best')
    ax.grid(True, alpha=0.3, linestyle='--')

    # Автоматическая настройка отступов
    plt.tight_layout()

    plt.savefig(fileName, format='png', dpi=500, bbox_inches='tight')
    plt.close(fig)