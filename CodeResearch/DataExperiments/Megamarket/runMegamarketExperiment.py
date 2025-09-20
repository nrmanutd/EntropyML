import numpy as np
import pandas as pd

from CodeResearch.DataExperiments.Megamarket.extractDataToStrings import extractDataToStrings
from CodeResearch.Visualization.summarizeExperiments import summarizeExperiments
from CodeResearch.Visualization.visualizeAndSaveComplexObjects import visualizeAndSaveComplexObjects
from CodeResearch.Visualization.visualizeAndSaveKSSI import visualizeAndSaveKSSI
from CodeResearch.dataSets import load_megamarket
from CodeResearch.estimateAndVisualizeEmpiricalDistributionDelta import estimatePValuesForClassesSeparation

x, y = load_megamarket('../../Data/megamarket/sampled_10k.parquet')

alphas = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5]
taskName = "megamarket"
iterations = 200
logsFolder = "Logs200"

#for alpha in alphas:
#    estimatePValuesForClassesSeparation(x, y, taskName, ksAttempts=iterations, pAttempts=0, mlAttempts=0, folder='..\..\PValuesFigures', alpha=alpha, allowedClasses=["12_5"])

#visualizeAndSaveKSSI(logsFolder, "KSSI", alphas, taskName, iterations)

x = pd.read_parquet("../../Data/megamarket/sampled_10k.parquet")
y = np.array(x['cat_level_1'])
x = extractDataToStrings(x)

#visualizeAndSaveComplexObjects(logsFolder, "ComplexObjects", taskName, iterations, x, y)
summarizeExperiments(logsFolder, "Summary", y, taskName, iterations)