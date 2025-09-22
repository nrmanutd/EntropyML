from CodeResearch.DataExperiments.Proteins.indexExtractor import getProteinsComplexities
from CodeResearch.Visualization.summarizeExperiments import summarizeExperiments
from CodeResearch.Visualization.visualizeAndSaveComplexObjects import visualizeAndSaveComplexObjects
from CodeResearch.Visualization.visualizeAndSaveKSSI import visualizeAndSaveKSSI
from CodeResearch.dataSets import load_proteins
from CodeResearch.estimateAndVisualizeEmpiricalDistributionDelta import estimatePValuesForClassesSeparation

x, y = load_proteins("../../Data/Proteins/df_master.csv")

taskName = "proteins_full"
iterations = 200

estimatePValuesForClassesSeparation(x, y, taskName, ksAttempts=iterations, pAttempts=0, mlAttempts=0, folder='..\..\PValuesFigures', alpha=[0.5])
logsFolder = "..\..\PValuesFigures\PValueLogs"

visualizeAndSaveComplexObjects(logsFolder, "ProteinsComplexObjects", taskName, iterations, x, y)
summarizeExperiments(logsFolder, "ProteinsSummary", y, taskName, iterations)

ligandsIndexes = getProteinsComplexities(logsFolder, taskName, iterations)
ligandsIndexes.to_csv('ProteinsSummary\ligands.csv', index=False)
