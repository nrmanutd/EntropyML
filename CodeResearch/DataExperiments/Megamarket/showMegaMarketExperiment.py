import math

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from CodeResearch.Visualization.VisualizeAndSaveCommonTopSubsamples import visualizeAndSaveKSForEachPairFirst
from CodeResearch.Visualization.saveDataForVisualization import deserialize_labeles_list_of_arrays
from CodeResearch.dataSets import load_megamarket

x, y = load_megamarket("../../Data/megamarket/sampled_10k.parquet")

enc = LabelEncoder()
target = enc.fit_transform(np.ravel(y))

pairs = set(zip(y, target))
resultPairs = [""] * len(pairs)

for a,b in pairs:
    resultPairs[b] = a

result = deserialize_labeles_list_of_arrays('..\..\PValuesFigures\Megamarket\KS_megamarket_200_26_25_total.txt')

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
classesFrame = pd.DataFrame({'Class': np.array(resultPairs)[classesIndex], 'Class number': np.arange(totalClasses)[classesIndex], 'Rating': np.array(classesRating)[classesIndex], '3xSigma': np.array(classesSigma)[classesIndex]})
classesFrame.to_excel("megamarket_rating.xlsx")

df = pd.DataFrame({"Cat 1 pairs": np.array(pairsStr)[sortedIndices], "Cat 1 pairs numbers": np.array(result[1])[sortedIndices], "First": np.array(firstPair)[sortedIndices], "Second": np.array(secondPair)[sortedIndices], "KS medians" : np.array(quantiles_list)[sortedIndices]})
df.to_excel("megamarket.xlsx", index=False)

visualizeAndSaveKSForEachPairFirst(result[0], result[1], result[2], result[3], 'total_pairs', '..\..\PValuesFigures\Megamarket', len(result[1]))
