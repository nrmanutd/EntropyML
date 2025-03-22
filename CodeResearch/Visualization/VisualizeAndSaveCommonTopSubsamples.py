import numpy as np
import matplotlib.pyplot as plt

# Функция для расчета квантилей
def calculate_quantiles(data):
    quantiles = {
        '1%': np.quantile(data, 0.01),
        '5%': np.quantile(data, 0.05),
        '10%': np.quantile(data, 0.10),
        '50%': np.quantile(data, 0.50),
        '90%': np.quantile(data, 0.90),
        '95%': np.quantile(data, 0.95),
        '99%': np.quantile(data, 0.99)
    }
    return quantiles

def visualizeAndSaveKSForEachPair(data, pairsNames, taskName, curPair):
    # Расчет квантилей для каждой пары классов
    quantiles_list = [calculate_quantiles(d) for d in data]

    # Расчет дельты между 99% и 1% квантилями
    deltas = [q['99%'] - q['1%'] for q in quantiles_list]

    # Сортировка пар классов по возрастанию дельты
    sorted_indices = np.argsort(deltas)
    sorted_data = [data[i] for i in sorted_indices]
    sorted_quantiles = [quantiles_list[i] for i in sorted_indices]
    sorted_deltas = [deltas[i] for i in sorted_indices]  # Отсортированные дельты

    # Построение графика
    fig = plt.figure(figsize=(15, 8))

    for i, (d, q, delta) in enumerate(zip(sorted_data, sorted_quantiles, sorted_deltas)):
        positions = np.random.normal(i + 1, 0.04, size=len(d))  # Добавляем небольшой разброс по оси X для наглядности
        plt.scatter(positions, d, alpha=0.5, s=5, marker='x')

        # Отображение квантилей с увеличенным диапазоном по горизонтали
        x_center = i + 1  # Центр для текущей пары классов
        x_min = x_center - 0.3  # Расширяем диапазон по горизонтали
        x_max = x_center + 0.3  # Расширяем диапазон по горизонтали

        if i == 0:  # Добавляем легенду только для первой пары классов
            plt.hlines(q['50%'], x_min, x_max, colors='r', linestyles='solid', label='50%', linewidth=0.8)
            plt.hlines([q['10%'], q['90%']], x_min + 0.1, x_max - 0.1, colors='g', linestyles='dashed', label='10%/90%',
                       linewidth=0.8)
            plt.hlines([q['5%'], q['95%']], x_min + 0.05, x_max - 0.05, colors='b', linestyles='dotted', label='5%/95%',
                       linewidth=0.8)
            plt.hlines([q['1%'], q['99%']], x_min, x_max, colors='purple', linestyles='dashdot', label='1%/99%',
                       linewidth=0.8)
        else:  # Для остальных пар классов не добавляем легенду
            plt.hlines(q['50%'], x_min, x_max, colors='r', linestyles='solid', linewidth=0.8)
            plt.hlines([q['10%'], q['90%']], x_min + 0.1, x_max - 0.1, colors='g', linestyles='dashed', linewidth=0.8)
            plt.hlines([q['5%'], q['95%']], x_min + 0.05, x_max - 0.05, colors='b', linestyles='dotted', linewidth=0.8)
            plt.hlines([q['1%'], q['99%']], x_min, x_max, colors='purple', linestyles='dashdot', linewidth=0.8)

        # Подпись дельты над самой верхней точкой
        max_y = np.max(d)  # Самая верхняя точка для текущей пары классов
        if max_y < 0.95:
            plt.text(x_center, max_y + 0.01, f"{delta:.2f}", ha='center', va='bottom', fontsize=8, color='black')
        else:
            min_y = np.min(d)
            plt.text(x_center, min_y - 0.01, f"{delta:.2f}", ha='center', va='bottom', fontsize=8, color='black')

    plt.xticks(range(1, len(sorted_data) + 1), [pairsNames[sorted_indices[i]] for i in range(len(sorted_data))], fontsize=6)
    plt.xlabel('Pairs of Classes')
    plt.ylabel('Statistic Values')
    plt.title('Distribution of Statistics for Each Pair of Classes')
    plt.legend()
    plt.grid(True)

    # Сохранение графика в файл
    plt.savefig('PValuesFigures\\statistics_distribution_{:}_{:}.png'.format(taskName, curPair), dpi=300, bbox_inches='tight')  # Сохраняем в файл 'statistics_distribution.png'
    plt.close(fig)