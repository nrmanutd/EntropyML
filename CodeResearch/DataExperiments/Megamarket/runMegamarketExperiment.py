import os
import pandas as pd
import numpy as np

from CodeResearch.DataExperiments.Megamarket.extractDataToStrings import extractDataToStrings
from CodeResearch.Visualization.visualizeAndSaveComplexObjects import visualizeAndSaveComplexObjects
from CodeResearch.Visualization.visualizeAndSaveKSSI import visualizeAndSaveKSSI
from CodeResearch.dataSets import load_megamarket
from CodeResearch.estimateAndVisualizeEmpiricalDistributionDelta import estimatePValuesForClassesSeparation

x, y = load_megamarket("../../Data/megamarket/sampled_10k.parquet")
iterations = 200
taskName = "megamarket"

estimatePValuesForClassesSeparation(x, y, taskName, ksAttempts=iterations, pAttempts=0, mlAttempts=0, folder='..\..\PValuesFigures')

pass

alphas = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5]
#for alpha in alphas:
#    estimatePValuesForClassesSeparation(x, y, taskName, ksAttempts=iterations, pAttempts=0, mlAttempts=0,
#                                        folder='..\..\PValuesFigures', alpha=alpha, allowedClasses=["21_6"])

iterations = 200
taskName = "megamarket"
#visualizeAndSaveKSSI("..\..\PValuesFigures\PValueLogs", os.curdir, alphas, taskName, iterations)
#visualizeAndSaveKSSI("Logs200", os.curdir, alphas, taskName, iterations)

x = pd.read_parquet("../../Data/megamarket/sampled_10k.parquet")
y = np.array(x['cat_level_1'])
x = extractDataToStrings(x)

visualizeAndSaveComplexObjects("Logs200", "ComplexObjects", taskName, iterations, x, y)