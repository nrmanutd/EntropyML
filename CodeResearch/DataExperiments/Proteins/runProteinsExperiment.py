import numpy as np
from CodeResearch.DataExperiments.Proteins.indexExtractor import getProteinsComplexities
from CodeResearch.Visualization.summarizeExperiments import summarizeExperiments
from CodeResearch.Visualization.visualizeAndSaveComplexObjects import visualizeAndSaveComplexObjects
from CodeResearch.Visualization.visualizeAndSaveKSSI import visualizeAndSaveKSSI
from CodeResearch.dataSets import load_proteins
from CodeResearch.estimateAndVisualizeEmpiricalDistributionDelta import estimatePValuesForClassesSeparation

x, y = load_proteins("../../Data/Proteins/df_master.csv")

#alphas = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5]
alphas = np.arange(0.01, 0.1, 0.01)
#alphas = [0.5]
taskName = "proteins_full_shap"
iterations = 1000

for alpha in alphas:
    estimatePValuesForClassesSeparation(x, y, taskName, ksAttempts=iterations, pAttempts=0, mlAttempts=0, folder='..\..\PValuesFigures', alpha=alpha, shapCalculation=True)

logsFolder = "..\..\PValuesFigures\PValueLogs"

#visualizeAndSaveKSSI(logsFolder, "ProteinsKSSI", alphas, taskName, iterations)
visualizeAndSaveComplexObjects(logsFolder, "ProteinsComplexObjects", taskName, iterations, x, y)
#summarizeExperiments(logsFolder, "ProteinsSummary", y, taskName, iterations)

ligandsIndexes = getProteinsComplexities(logsFolder, taskName, iterations)
ligandsIndexes.to_csv('ProteinsSummary\ligands.csv', index=False)
