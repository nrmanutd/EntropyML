import matplotlib.pyplot as plt
import numpy as np
import re
import os

from sklearn.preprocessing import LabelEncoder

from CodeResearch.Visualization.filesExtractor import find_files_with_regex
from CodeResearch.Visualization.saveDataForVisualization import deserialize_labeles_list_of_arrays

def plotAndSaveEntropies(target, firstClass, secondClass, entropies, folder, name):
    iObjects = list(np.where(target == firstClass)[0])

    # Построение гистограммы
    fig = plt.figure(figsize=(10, 6))

    e1 = entropies[0:len(iObjects)]
    e2 = entropies[len(iObjects):len(entropies)]

    plt.hist(e1, bins=30, alpha=0.5, label=f'Распределение {firstClass}', color='blue', density=True)
    plt.hist(e2, bins=30, alpha=0.5, label=f'Распределение {secondClass}', color='red', density=True)

    # Настройки графика
    plt.title('Гистограмма распределения данных', fontsize=14)
    plt.xlabel('Значения', fontsize=12)
    plt.ylabel('Частота', fontsize=12)
    plt.grid(alpha=0.3)
    plt.legend()

    plt.savefig('{:}\\{:}.png'.format(folder, name),
                dpi=300, bbox_inches='tight')
    plt.close(fig)

def extractAndSaveEdgeObjects(entropies, x, target, firstClass, secondClass, resultFolder, name, top):
    iObjects = list(np.where(target == firstClass)[0])
    jObjects = list(np.where(target == secondClass)[0])

    e1 = entropies[0:len(iObjects)]
    e2 = entropies[len(iObjects):len(entropies)]

    xx1 = x[iObjects]
    xx2 = x[jObjects]

    e1idx = np.argsort(np.array(e1))
    e2idx = np.argsort(np.array(e2))

    with open(f"{resultFolder}\\{name}_good_examples_{firstClass}_{secondClass}.csv", "w", encoding="utf-8") as file:
        for i in range(top):
            idx = e1idx[i]
            file.write("{0};{1};{2}\n".format(xx1[idx], e1[idx], firstClass))
        file.write("====================================\n")
        for i in range(top):
            idx = e2idx[i]
            file.write("{0};{1};{2}\n".format(xx2[idx], e2[idx], secondClass))

    c1len = len(e1idx)
    c2len = len(e2idx)

    with open(f"{resultFolder}\\{name}_bad_examples_{firstClass}_{secondClass}.csv", "w", encoding="utf-8") as file:
        for i in range(1, top):
            idx = e1idx[c1len - i]
            file.write("{0};{1};{2}\n".format(xx1[idx], e1[idx], firstClass))
        file.write("====================================\n")
        for i in range(1, top):
            idx = e2idx[c2len - i]
            file.write("{0};{1};{2}\n".format(xx2[idx], e2[idx], secondClass))

def getLastFiles(files):
    data = {}
    pattern = r"^.*KS_.*_.*_.*_(\d+)_(\d+)_(\d+).txt$"

    for file in files:
        match = re.search(pattern, file)
        num1, num2, num3 = match.groups()
        key = "{0}_{1}".format(num1, num2)

        num3 = int(num3)

        if key not in data:
            data[key] = {'file': file, 'objects': num3}
        else:
            if data[key]['objects'] < num3:
                data[key] = {'file': file, 'objects': num3}

    files = []
    for k, v in data.items():
        files.append(v['file'])

    return files


def visualizeAndSaveComplexObjects(folderWithFiles, resultsFolder, taskName, iterations, x, y, top=20):
    pattern = r"^KS_entropy_{0}_{1}_\d+_\d+_\d+.txt$".format(taskName, iterations)
    files = find_files_with_regex(folderWithFiles, pattern)
    files = getLastFiles(files)

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
        value = de[0][-1]

        ksFile = file.replace("_error", "")
        dks = deserialize_labeles_list_of_arrays(ksFile)
        ks = dks[0][-1]

        if key not in data:
            data[key] = {'frequencies': [], 'objects': [], 'ks': []}

        data[key]['frequencies'].append(value)
        data[key]['objects'].append(num3)
        data[key]['ks'].append(np.mean(ks))

    enc = LabelEncoder()
    target = enc.fit_transform(np.ravel(y))

    counter = 0
    for k, v in data.items():
        counter += 1
        print('Processing for pair {0} ({1}/{2})'.format(k, counter, len(data)))
        obj = np.array(v['objects'], dtype=np.int32)
        freq = v['frequencies']
        ks = np.array(v['ks'])

        idx = np.argsort(obj)
        frequencies = freq[idx[-1]]
        ks = ks[idx[-1]]

        c = k.split('_')
        f = int(c[0])
        s = int(c[1])

        plotAndSaveEntropies(target, f, s, frequencies, resultsFolder, f'entropies_{taskName}_{iterations}_{f}_{s}')
        extractAndSaveEdgeObjects(frequencies, x, target, f, s, resultsFolder, f'{taskName}_{iterations}', top)

