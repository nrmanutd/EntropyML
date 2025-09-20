import math
import os

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from CodeResearch.Visualization.VisualizeAndSaveCommonTopSubsamples import visualizeAndSaveKSForEachPairFirst
from CodeResearch.Visualization.filesExtractor import getLastFilesFromFolder
from CodeResearch.Visualization.saveDataForVisualization import deserialize_labeles_list_of_arrays_from_files


def summarizeExperiments(logsFolder, resultsFolder, y, taskName, iterations):
    if not os.path.exists(resultsFolder):
        os.makedirs(resultsFolder)

    enc = LabelEncoder()
    target = enc.fit_transform(np.ravel(y))

    utarget, tcounts = np.unique(target, return_counts=True)
    countsMap = dict(zip(utarget, tcounts/sum(tcounts)))

    pairs = set(zip(y, target))
    classesNamesMap = [""] * len(pairs)

    for a, b in pairs:
        classesNamesMap[b] = a

    files = getLastFilesFromFolder(logsFolder, f"^KS_OOS_{taskName}_{iterations}_\d+_\d+_\d+.txt$")
    result = deserialize_labeles_list_of_arrays_from_files(files, taskName, iterations)

    efiles = getLastFilesFromFolder(logsFolder, f"^KS_entropy_{taskName}_{iterations}_\d+_\d+_\d+.txt$")
    eresult = deserialize_labeles_list_of_arrays_from_files(efiles, taskName, iterations)

    ffiles = getLastFilesFromFolder(logsFolder, f"^KS_frequency_{taskName}_{iterations}_\d+_\d+_\d+.txt$")
    fresult = deserialize_labeles_list_of_arrays_from_files(ffiles, taskName, iterations)

    quantiles_list = [np.quantile(d, 0.5) for d in result[0]]
    sortedIndices = np.argsort(quantiles_list)

    firstPair = []
    secondPair = []

    firstClassAlpha = []
    secondClassAlpha = []

    for str in result[1]:
        s = str.split('_')
        firstPair.append(s[0])
        secondPair.append(s[1])

        firstClassAlpha.append(countsMap[int(s[0])])
        secondClassAlpha.append(countsMap[int(s[1])])

    entropies = np.zeros(len(result[1]))
    pairsStr = [""] * len(result[1])
    fFrequency = np.zeros(len(result[1]))
    sFrequency = np.zeros(len(result[1]))

    for i in np.arange(len(pairsStr)):
        curPair = result[1][i]
        f, s = map(int, curPair.split('_'))
        pairsStr[i] = f'{classesNamesMap[f]}/{classesNamesMap[s]}'

        iObjects = list(np.where(target == f)[0])

        curEntropy = eresult[0][i]
        curFrequency = fresult[0][i]

        fFrequency[i] = np.mean(curFrequency[0:len(iObjects)])
        sFrequency[i] = np.mean(curFrequency[len(iObjects):len(curFrequency)])

        entropies[i] = np.quantile(curEntropy, 0.5)

    arr = dict()

    sortedPairs = np.array(result[1])[sortedIndices]
    quantiles_arr = np.array(quantiles_list)[sortedIndices]
    for i in np.arange(len(sortedPairs)):
        curPair = sortedPairs[i]
        curKS = quantiles_arr[i]
        f, s = map(int, curPair.split('_'))

        if f not in arr:
            arr[f] = []

        if s not in arr:
            arr[s] = []

        arr[f].append(curKS)
        arr[s].append(curKS)

    classNames = []
    classNumbers = []
    classesRating = []
    classesSigma = []

    for k, v in arr.items():
        classNames.append(classesNamesMap[k])
        classNumbers.append(k)
        classesRating.append(np.quantile(v, 0.5))
        classesSigma.append(3 * math.sqrt(np.var(v)))

    classesIndex = np.argsort(classesRating)

    classesFrame = pd.DataFrame(
        {'Class': np.array(classNames)[classesIndex], 'Class number': np.array(classNumbers)[classesIndex],
         'Rating': np.array(classesRating)[classesIndex], '3xSigma': np.array(classesSigma)[classesIndex]})

    classesFrame.to_excel(f"{resultsFolder}\\{taskName}_{iterations}_rating.xlsx")

    firstPart = np.array(firstClassAlpha)[sortedIndices]
    secondPart = np.array(secondClassAlpha)[sortedIndices]
    minimum = np.minimum(firstPart, secondPart)

    minimumFrequency = np.minimum(fFrequency, sFrequency)

    df = pd.DataFrame(
        {"Cat 1 pairs": np.array(pairsStr)[sortedIndices], "Cat 1 pairs numbers": np.array(result[1])[sortedIndices],
         "First": np.array(firstPair)[sortedIndices], "Second": np.array(secondPair)[sortedIndices],
         "FirstPart": firstPart, "SecondPart": secondPart, "Minimum": minimum,
         "Entropy median": entropies[sortedIndices], "First_Frequency_Mean": fFrequency[sortedIndices],
         "Second_Frequency_Mean": sFrequency[sortedIndices], "Frequency_Minimum": minimumFrequency[sortedIndices],
         "KS medians": np.array(quantiles_list)[sortedIndices]})

    df.to_excel(f"{resultsFolder}\\{taskName}_{iterations}.xlsx", index=False)

    visualizeAndSaveKSForEachPairFirst(result[0], result[1], result[2], result[3], 'KS_OOS_total_pairs',
                                       resultsFolder, len(result[1]))

    files = getLastFilesFromFolder(logsFolder, f"^KS_{taskName}_{iterations}_\d+_\d+_\d+.txt$")
    res_ks = deserialize_labeles_list_of_arrays_from_files(files, taskName, iterations)

    visualizeAndSaveKSForEachPairFirst(res_ks[0], res_ks[1], res_ks[2], res_ks[3], 'KS_total_pairs',
                                       resultsFolder, len(res_ks[1]))

