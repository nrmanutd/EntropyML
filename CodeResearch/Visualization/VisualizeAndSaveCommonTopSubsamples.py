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

def visualizeAndSaveKSForEachPairAndTwoDistributions(data1, data2, labels, taskName, curPair, folder='PValuesFigures'):
    # Расчет квантилей для каждого распределения
    quantiles_list1 = [calculate_quantiles(d) for d in data1]
    quantiles_list2 = [calculate_quantiles(d) for d in data2]

    # Расчет дельты между 99% и 1% квантилями для первого распределения
    deltas1 = [q['99%'] - q['1%'] for q in quantiles_list1]
    m = [q['50%'] for q in quantiles_list1]

    # Сортировка пар классов по возрастанию дельты (по первому распределению)
    sorted_indices = np.argsort(m)
    sorted_data1 = [data1[i] for i in sorted_indices]
    sorted_data2 = [data2[i] for i in sorted_indices]
    sorted_quantiles1 = [quantiles_list1[i] for i in sorted_indices]
    sorted_quantiles2 = [quantiles_list2[i] for i in sorted_indices]
    sorted_deltas1 = [deltas1[i] for i in sorted_indices]  # Отсортированные дельты

    # Генерация лейблов для пар классов
    sorted_labels = [labels[i] for i in sorted_indices]

    # Построение графика
    fig = plt.figure(figsize=(20, 8))

    # Вычисляем общий диапазон значений по оси Y
    all_values = np.concatenate(sorted_data1 + sorted_data2)
    y_min, y_max = np.min(all_values), np.max(all_values)
    y_range = y_max - y_min

    # Задаем смещение для подписей как 2% от диапазона значений по оси Y
    label_offset = 0.02 * y_range

    for i, (d1, d2, q1, q2, delta1) in enumerate(
            zip(sorted_data1, sorted_data2, sorted_quantiles1, sorted_quantiles2, sorted_deltas1)):
        # Смещение по горизонтали для первого и второго распределения
        x_center1 = i + 1 - 0.2  # Первое распределение смещено влево
        x_center2 = i + 1 + 0.2  # Второе распределение смещено вправо

        # Отображение точек для первого распределения
        positions1 = np.random.normal(x_center1, 0.04, size=len(d1))
        plt.scatter(positions1, d1, alpha=0.5, s=10, marker='x', label='Source distribution' if i == 0 else "")

        # Отображение точек для второго распределения
        positions2 = np.random.normal(x_center2, 0.04, size=len(d2))
        plt.scatter(positions2, d2, alpha=0.5, s=10, marker='o', label='Random permutation' if i == 0 else "")

        if i == 0:
            # Отображение квантилей для первого распределения
            plt.hlines(q1['50%'], x_center1 - 0.15, x_center1 + 0.15, colors='r', linestyles='solid', label='50%', linewidth=0.8)
            plt.hlines([q1['10%'], q1['90%']], x_center1 - 0.15, x_center1 + 0.15, colors='g', linestyles='dashed', label='10%/90%',
                       linewidth=0.8)
            plt.hlines([q1['5%'], q1['95%']], x_center1 - 0.15, x_center1 + 0.15, colors='b', linestyles='dotted', label='5%/95%',
                       linewidth=0.8)
            plt.hlines([q1['1%'], q1['99%']], x_center1 - 0.15, x_center1 + 0.15, colors='purple', linestyles='dashdot', label='1%/99%',
                       linewidth=0.8)

        else:
            # Отображение квантилей для первого распределения
            plt.hlines(q1['50%'], x_center1 - 0.15, x_center1 + 0.15, colors='r', linestyles='solid',
                       linewidth=0.8)
            plt.hlines([q1['10%'], q1['90%']], x_center1 - 0.15, x_center1 + 0.15, colors='g', linestyles='dashed',
                       linewidth=0.8)
            plt.hlines([q1['5%'], q1['95%']], x_center1 - 0.15, x_center1 + 0.15, colors='b', linestyles='dotted',
                       linewidth=0.8)
            plt.hlines([q1['1%'], q1['99%']], x_center1 - 0.15, x_center1 + 0.15, colors='purple', linestyles='dashdot',
                       linewidth=0.8)

        # Отображение квантилей для второго распределения
        plt.hlines(q2['50%'], x_center2 - 0.15, x_center2 + 0.15, colors='r', linestyles='solid', linewidth=0.8)
        plt.hlines([q2['10%'], q2['90%']], x_center2 - 0.15, x_center2 + 0.15, colors='g', linestyles='dashed',
                       linewidth=0.8)
        plt.hlines([q2['5%'], q2['95%']], x_center2 - 0.15, x_center2 + 0.15, colors='b', linestyles='dotted',
                       linewidth=0.8)
        plt.hlines([q2['1%'], q2['99%']], x_center2 - 0.15, x_center2 + 0.15, colors='purple', linestyles='dashdot',
                       linewidth=0.8)

        # Подпись дельты над самым верхним значением для первого распределения
        # Подпись дельты над самой верхней точкой
        max_y = np.max(d1)  # Самая верхняя точка для текущей пары классов
        if max_y < 0.95:
            plt.text(x_center1, max_y + 0.005, f"{delta1:.2f}", ha='center', va='bottom', fontsize=8, color='black')
        else:
            min_y = np.min(d1)
            plt.text(x_center1, min_y - 0.015, f"{delta1:.2f}", ha='center', va='bottom', fontsize=8, color='black')

    # Устанавливаем подписи по оси X с уменьшенным шрифтом
    plt.xticks(range(1, len(sorted_data1) + 1), sorted_labels, rotation=90, fontsize=8)

    plt.xlabel('Pairs of Classes', fontsize=12)
    plt.ylabel('Statistic Values', fontsize=12)
    plt.title('Distribution of Statistics for {:}'.format(taskName), fontsize=14)
    plt.legend()  # Легенда будет содержать только метки распределений
    plt.grid(True)

    # Сохранение графика в файл
    #plt.savefig('PValuesFigures\\statistics_distribution_two_distributions_{:}_{:}.png'.format(taskName, curPair), dpi=300, bbox_inches='tight')
    plt.savefig('{:}\\statistics_distribution_two_distributions_{:}_{:}.png'.format(folder, taskName, curPair),
                dpi=300, bbox_inches='tight')
    plt.close(fig)

def visualizeAndSaveKSForEachPair(data, pairsNames, taskName, nAttempts, curPair, folder='PValuesFigures'):
    # Расчет квантилей для каждой пары классов
    quantiles_list = [calculate_quantiles(d) for d in data]

    # Расчет дельты между 99% и 1% квантилями
    deltas = [q['99%'] - q['1%'] for q in quantiles_list]
    ms = [q['50%'] for q in quantiles_list]

    # Сортировка пар классов по возрастанию дельты
    sorted_indices = np.argsort(ms)
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
        #max_y = np.max(d)  # Самая верхняя точка для текущей пары классов
        #if max_y < 0.95:
        #    plt.text(x_center, max_y + 0.005, f"{delta:.2f}", ha='center', va='bottom', fontsize=8, color='black')
        #else:
        #    min_y = np.min(d)
        #    plt.text(x_center, min_y - 0.015, f"{delta:.2f}", ha='center', va='bottom', fontsize=8, color='black')

    plt.xticks(range(1, len(sorted_data) + 1), [pairsNames[sorted_indices[i]] for i in range(len(sorted_data))], fontsize=6)
    plt.xlabel('Pairs of Classes')
    plt.ylabel('Statistic Values')
    plt.title('Distribution for {:} attempts {:}'.format(taskName, nAttempts))
    plt.legend()
    plt.grid(True)

    # Сохранение графика в файл
    plt.savefig('{:}\\statistics_distribution_{:}_{:}.png'.format(folder,taskName, curPair), dpi=300, bbox_inches='tight')  # Сохраняем в файл 'statistics_distribution.png'
    plt.close(fig)