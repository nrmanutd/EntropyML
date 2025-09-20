import numpy as np

from CodeResearch.Helpers.commonHelpers import perform_pca
from CodeResearch.Visualization.summarizeExperiments import summarizeExperiments
from CodeResearch.Visualization.visualizeAndSaveComplexObjects import visualizeAndSaveComplexObjects
from CodeResearch.Visualization.visualizeAndSaveKSSI import visualizeAndSaveKSSI
from CodeResearch.dataSets import load_proteins
from CodeResearch.estimateAndVisualizeEmpiricalDistributionDelta import estimatePValuesForClassesSeparation

x, y = load_proteins("../../Data/Proteins/df_master.csv")

median_val = np.median(y)
y = np.where(y >= median_val, 1, 0)

alphas = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5]
taskName = "proteins_full"
iterations = 200

x = np.hstack((x, -x))

for alpha in alphas:
    estimatePValuesForClassesSeparation(x, y, taskName, ksAttempts=iterations, pAttempts=0, mlAttempts=0, folder='..\..\PValuesFigures', alpha=alpha)

visualizeAndSaveKSSI("..\..\PValuesFigures\PValueLogs", "ProteinsKSSI", alphas, taskName, iterations)
visualizeAndSaveComplexObjects("..\..\PValuesFigures\PValueLogs", "ProteinsComplexObjects", taskName, iterations, x, y)
summarizeExperiments("..\..\PValuesFigures\PValueLogs", "ProteinsSummary", y, taskName, iterations)
