import matplotlib.pyplot as plt
import numpy as np

from CodeResearch.Visualization.saveDataForVisualization import deserialize_labeles_list_of_arrays


def plot_dual_histograms(array1, array2, l1, l2, bins=30, alpha=0.7):
    """
    Строит гистограммы двух массивов на одном графике
    """
    plt.figure(figsize=(10, 6))

    # Гистограмма первого массива
    plt.hist(array1, bins=bins, alpha=alpha, density=True,
             label=l1, color='blue')

    # Гистограмма второго массива
    plt.hist(array2, bins=bins, alpha=alpha, density=True,
             label=l2, color='red')

    plt.xlabel('Значения')
    plt.ylabel('Плотность')
    plt.title('Гистограммы KS vs KS random permutation')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

ks = deserialize_labeles_list_of_arrays('..\..\PValuesFigures\PValueLogs\KS_proteins_1000_1_0.txt')
pks = deserialize_labeles_list_of_arrays('..\..\PValuesFigures\PValueLogs\KS_permutation_proteins_1000_1_0.txt')

plot_dual_histograms(ks[0][0], pks[0][0], 'KS', 'KS (random permutation)')