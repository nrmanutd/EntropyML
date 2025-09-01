from sklearn import datasets

from CodeResearch.Visualization.summarizeExperiments import summarizeExperiments
from CodeResearch.Visualization.visualizeAndSaveComplexObjects import visualizeAndSaveComplexObjects
from CodeResearch.Visualization.visualizeAndSaveKSSI import visualizeAndSaveKSSI
from CodeResearch.estimateAndVisualizeEmpiricalDistributionDelta import estimatePValuesForClassesSeparation

nSamples = 2000
x, y = datasets.make_blobs(n_samples=2000, centers=2, n_features=2, random_state=42)

alphas = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5]
taskName = "blobs_example"
iterations = 200

for alpha in alphas:
    estimatePValuesForClassesSeparation(x, y, taskName, ksAttempts=iterations, pAttempts=0, mlAttempts=0, folder='..\..\PValuesFigures', alpha=alpha)

visualizeAndSaveKSSI("..\..\PValuesFigures\PValueLogs", "ExampleKSSI", alphas, taskName, iterations)
visualizeAndSaveComplexObjects("..\..\PValuesFigures\PValueLogs", "ExampleComplexObjects", taskName, iterations, x, y)
summarizeExperiments("..\..\PValuesFigures\PValueLogs", "ExampleSummary", y, taskName, iterations)

