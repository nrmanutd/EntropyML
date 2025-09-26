import math

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import numpy as np
import re
import os

from sklearn.preprocessing import LabelEncoder

from CodeResearch.Visualization.filesExtractor import find_files_with_regex, getLastFiles
from CodeResearch.Visualization.saveDataForVisualization import deserialize_labeles_list_of_arrays

def plotAndSaveEntropies(target, firstClass, secondClass, entropies, frequencies, folder, name):
    iObjects = list(np.where(target == firstClass)[0])

    # Создаем фигуру с двумя subplots один под другим
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))

    # Разделяем данные для entropies
    e1 = entropies[0:len(iObjects)]
    e2 = entropies[len(iObjects):len(entropies)]

    # Разделяем данные для frequencies
    f1 = frequencies[0:len(iObjects)]
    f2 = frequencies[len(iObjects):len(frequencies)]

    # Первый график - entropies
    ax1.hist(e1, bins=30, alpha=0.5, label=f'Распределение {firstClass}', color='blue', density=True)
    ax1.hist(e2, bins=30, alpha=0.5, label=f'Распределение {secondClass}', color='red', density=True)
    ax1.set_title('Гистограмма распределения энтропий', fontsize=14)
    ax1.set_xlabel('Значения энтропии', fontsize=12)
    ax1.set_ylabel('Частота', fontsize=12)
    ax1.grid(alpha=0.3)
    ax1.legend()

    # Второй график - frequencies
    ax2.hist(f1, bins=30, alpha=0.5, label=f'Распределение {firstClass}', color='green', density=True)
    ax2.hist(f2, bins=30, alpha=0.5, label=f'Распределение {secondClass}', color='orange', density=True)
    ax2.set_title('Гистограмма распределения частот', fontsize=14)
    ax2.set_xlabel('Значения частот', fontsize=12)
    ax2.set_ylabel('Частота', fontsize=12)
    ax2.grid(alpha=0.3)
    ax2.legend()

    # Настраиваем расстояние между subplots
    plt.tight_layout()

    # Сохраняем график
    plt.savefig('{:}\\{:}.png'.format(folder, name),
                dpi=300, bbox_inches='tight')
    plt.close(fig)

def extractAndSaveEdgeObjects(entropies, frequencies, x, target, firstClass, secondClass, resultFolder, name, top):
    iObjects = list(np.where(target == firstClass)[0])
    jObjects = list(np.where(target == secondClass)[0])

    e1 = entropies[0:len(iObjects)]
    e2 = entropies[len(iObjects):len(entropies)]

    f1 = frequencies[0:len(iObjects)]
    f2 = frequencies[len(iObjects):len(frequencies)]

    xx1 = x[iObjects]
    xx2 = x[jObjects]

    e1idx = np.argsort(np.array(-np.abs(e1)))
    e2idx = np.argsort(np.array(-np.abs(e2)))

    f1idx = np.argsort(np.array(f1))
    f2idx = np.argsort(np.array(f2))

    c1len = len(e1idx)
    c2len = len(e2idx)

    with open(f"{resultFolder}\\{name}_examples_{firstClass}_{secondClass}.csv", "w", encoding="utf-8") as file:
        file.write("====================================\n")
        file.write("===========Border examples==========\n")
        file.write("====================================\n")
        file.write(f"===========Class {firstClass}==========\n")
        for i in range(1, top):
            idx = e1idx[c1len - i]
            file.write("{0}:{1};{2};{3}\n".format(iObjects[idx], xx1[idx], e1[idx], firstClass))
        file.write(f"============Class {secondClass}========\n")
        for i in range(1, top):
            idx = e2idx[c2len - i]
            file.write("{0}:{1};{2};{3}\n".format(jObjects[idx], xx2[idx], e2[idx], secondClass))

        file.write("====================================\n")
        file.write("===========Complex examples=========\n")
        file.write("====================================\n")
        file.write(f"===========Class {firstClass}==========\n")
        for i in range(top):
            idx = f1idx[i]
            file.write("{0}:{1};{2};{3}\n".format(iObjects[idx], xx1[idx], f1[idx], firstClass))
        file.write(f"============Class {secondClass}========\n")
        for i in range(top):
            idx = f2idx[i]
            file.write("{0}:{1};{2};{3}\n".format(jObjects[idx], xx2[idx], f2[idx], secondClass))

        file.write("====================================\n")
        file.write("===========Simple examples==========\n")
        file.write("====================================\n")
        file.write(f"===========Class {firstClass}==========\n")
        for i in range(1, top):
            idx = f1idx[c1len - i]
            file.write("{0}:{1};{2};{3}\n".format(iObjects[idx], xx1[idx], f1[idx], firstClass))
        file.write(f"============Class {secondClass}========\n")
        for i in range(1, top):
            idx = f2idx[c2len - i]
            file.write("{0}:{1};{2};{3}\n".format(jObjects[idx], xx2[idx], f2[idx], secondClass))

def plot_with_custom_brightness(X, y, complexity, resultFolder, title="Custom Brightness Visualization"):

    X_vis = X
    x_label, y_label = 'Feature 1', 'Feature 2'

    # Кастомная цветовая карта
    colors = ['darkred', 'red', 'yellow', 'green', 'darkgreen']
    positions = [0, 0.3, 0.5, 0.7, 1]
    cmap = LinearSegmentedColormap.from_list('center_bright', list(zip(positions, colors)))

    # Нелинейная функция яркости - более резкий переход
    def calculate_alpha(comp):
        #return 1.0 - (abs(comp - 1))
        return comp**4

    fig, ax = plt.subplots(figsize=(12, 8))

    unique_classes = np.unique(y)
    markers = ['o', 's']

    for i, class_label in enumerate(unique_classes):
        class_mask = (y == class_label)
        comp_values = complexity[class_mask]

        min_v = np.min(comp_values)
        max_v = np.max(comp_values)

        cc = (comp_values - min_v) / (max_v - min_v)

        alpha_values = [calculate_alpha(c) for c in cc]

        scatter = ax.scatter(X_vis[class_mask, 0], X_vis[class_mask, 1],
                             c=comp_values,
                             cmap=cmap, vmin=min_v, vmax=max_v,
                             marker = markers[i],
                             alpha=alpha_values, s=60,
                             label=f'Class {class_label}',
                             edgecolors='black', linewidth=0.8)

    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Complexity (0=red, 1=green)\nBrightness = f(distance from 0.5)')

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(f"{title}\n(Brightest at complexity=0.5)")
    ax.legend()
    ax.grid(True, alpha=0.2)

    plt.savefig('{:}\\{:}.png'.format(resultFolder, title),
                dpi=300, bbox_inches='tight')
    plt.close(fig)

def extractData(x, target, firstClass, secondClass):
    iObjects = list(np.where(target == firstClass)[0])
    jObjects = list(np.where(target == secondClass)[0])

    allObjects = iObjects + jObjects

    xx = x[allObjects]
    tt = target[allObjects]

    return xx, tt

def transformDataTo2D(x):

    if x.shape[1] <= 2:
        return x

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(x)
    tsne = TSNE(n_components=2, random_state=42)
    X_vis = tsne.fit_transform(X_scaled)
    return X_vis

def visualizeAndSaveComplexObjects(folderWithFiles, resultsFolder, taskName, iterations, x, y, top=20):
    pattern = r"^KS_entropy_{0}_{1}_\d+_\d+_\d+.txt$".format(taskName, iterations)
    files = find_files_with_regex(folderWithFiles, pattern)
    #files = getLastFiles(files)

    if not os.path.exists(resultsFolder):
        os.makedirs(resultsFolder)

    data = {}
    pattern = r".*_(\d+)_(\d+)_(\d+)\.txt$"
    for file in files:
        print(f'Processing file {file}...')

        match = re.search(pattern, file)
        num1, num2, num3 = match.groups()
        key = "{0}_{1}".format(num1, num2)

        de = deserialize_labeles_list_of_arrays(file)
        valueEntropy = de[0][-1]

        f = file.replace('_entropy', '_frequency')
        de2 = deserialize_labeles_list_of_arrays(f)
        valueFrequecny = de2[0][-1]

        if key not in data:
            data[key] = {'frequencies': [], 'entropy':[], 'objects': []}

        data[key]['frequencies'].append(valueFrequecny)
        data[key]['entropy'].append(valueEntropy)
        data[key]['objects'].append(num3)

    enc = LabelEncoder()
    target = enc.fit_transform(np.ravel(y))

    counter = 0
    for k, v in data.items():
        counter += 1
        print('Processing for pair {0} ({1}/{2})'.format(k, counter, len(data)))
        obj = np.array(v['objects'], dtype=np.int32)
        freq = v['frequencies']
        entr = v['entropy']

        c = k.split('_')
        f = int(c[0])
        s = int(c[1])

        print(obj)

        xx, yy = extractData(x, y, f, s)
        xx = transformDataTo2D(x)

        for i in range(len(obj)):
            currentObjects = obj[i]
            ff = freq[i]
            ee = entr[i]

            plotAndSaveEntropies(target, f, s, ee, ff, resultsFolder, f'entropies_{taskName}_{iterations}_{currentObjects}_{f}_{s}')
            extractAndSaveEdgeObjects(ee, ff, x, target, f, s, resultsFolder, f'{taskName}_{iterations}_{currentObjects}', top)
            plot_with_custom_brightness(xx, yy, np.array(ff), resultsFolder, f'{taskName}_{iterations}_{currentObjects}_{f}_{s}_colored')

