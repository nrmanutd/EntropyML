import math
import os
import re

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from CodeResearch.Visualization.VisualizeAndSaveCommonTopSubsamples import visualizeAndSaveKSForEachPairFirst
from CodeResearch.Visualization.filesExtractor import find_files_with_regex
from CodeResearch.Visualization.saveDataForVisualization import deserialize_labeles_list_of_arrays

def getLastFile(logsFolder, pattern):

    files = find_files_with_regex(logsFolder, pattern)
    maxObjects = 0
    bestFile = files[0]

    p = r".*_\d+_\d+_(\d+)\.txt$"

    for file in files:
        match = re.search(p, file)
        num1 = int(match.groups()[0])

        if num1 > maxObjects:
            bestFile = file
            maxObjects = num1

    return bestFile

def summarizeExperiments(logsFolder, resultsFolder, y, taskName, iterations):
    if not os.path.exists(resultsFolder):
        os.makedirs(resultsFolder)

    enc = LabelEncoder()
    target = enc.fit_transform(np.ravel(y))

    pairs = set(zip(y, target))
    resultPairs = [""] * len(pairs)

    for a, b in pairs:
        resultPairs[b] = a

    totalClasses = len(np.unique(target))
    f = totalClasses - 1
    s = totalClasses - 2

    file_oos = getLastFile(logsFolder, f"^KS_OOS_{taskName}_{iterations}_{f}_{s}_(\d+).txt$")
    result = deserialize_labeles_list_of_arrays(file_oos)

    quantiles_list = [np.quantile(d, 0.5) for d in result[0]]
    sortedIndices = np.argsort(quantiles_list)

    firstPair = []
    secondPair = []

    for str in result[1]:
        s = str.split('_')
        firstPair.append(s[0])
        secondPair.append(s[1])

    pairsStr = [""] * len(result[1])
    for i in np.arange(len(pairsStr)):
        curPair = result[1][i]
        f, s = map(int, curPair.split('_'))
        curStr = '{0}/{1}'.format(resultPairs[f], resultPairs[s])
        pairsStr[i] = curStr

    totalClasses = len(pairs)
    arr = [[] for _ in range(totalClasses)]

    sortedPairs = np.array(result[1])[sortedIndices]
    quantiles_arr = np.array(quantiles_list)[sortedIndices]
    for i in np.arange(len(sortedPairs)):
        curPair = sortedPairs[i]
        curKS = quantiles_arr[i]
        f, s = map(int, curPair.split('_'))
        arr[f].append(curKS)
        arr[s].append(curKS)

    classesRating = [np.quantile(d, 0.5) for d in arr]
    classesSigma = 3 * [math.sqrt(np.var(d)) for d in arr]

    classesIndex = np.argsort(classesRating)
    classesFrame = pd.DataFrame(
        {'Class': np.array(resultPairs)[classesIndex], 'Class number': np.arange(totalClasses)[classesIndex],
         'Rating': np.array(classesRating)[classesIndex], '3xSigma': np.array(classesSigma)[classesIndex]})

    classesFrame.to_excel(f"{resultsFolder}\\{taskName}_rating.xlsx")

    df = pd.DataFrame(
        {"Cat 1 pairs": np.array(pairsStr)[sortedIndices], "Cat 1 pairs numbers": np.array(result[1])[sortedIndices],
         "First": np.array(firstPair)[sortedIndices], "Second": np.array(secondPair)[sortedIndices],
         "KS medians": np.array(quantiles_list)[sortedIndices]})

    df.to_excel(f"{resultsFolder}\\{taskName}.xlsx", index=False)

    visualizeAndSaveKSForEachPairFirst(result[0], result[1], result[2], result[3], 'total_pairs',
                                       resultsFolder, len(result[1]))

    file = getLastFile(logsFolder, f"^KS_{taskName}_{iterations}_{f}_{s}_(\d+).txt$")
    res_ks = deserialize_labeles_list_of_arrays(file)
    visualizeAndSaveKSForEachPairFirst(res_ks[0], res_ks[1], res_ks[2], res_ks[3], 'total_pairs',
                                       resultsFolder, len(res_ks[1]))

