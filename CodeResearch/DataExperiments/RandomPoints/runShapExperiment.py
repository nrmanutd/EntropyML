from sklearn import datasets
import numpy as np

from CodeResearch.Visualization.summarizeExperiments import summarizeExperiments
from CodeResearch.Visualization.visualizeAndSaveComplexObjects import visualizeAndSaveComplexObjects
from CodeResearch.Visualization.visualizeAndSaveKSSI import visualizeAndSaveKSSI
from CodeResearch.dataSets import make_random, make_xor, loadMnist, make_spirals
from CodeResearch.estimateAndVisualizeEmpiricalDistributionDelta import estimatePValuesForClassesSeparation

nSamples = 2000
#x, y = make_random(nSamples)
#x, y = datasets.make_blobs(n_samples=nSamples, centers=2, n_features=2, random_state=42)
#x, y = make_xor(nSamples)
#x, y = datasets.make_circles(n_samples=nSamples, factor=0.5, noise=0.1, random_state=42)
#x, y = make_spirals(nSamples)
x, y = loadMnist()

#x = np.hstack((x, -x))
alphas = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5]
#alphas= [0.5]

allowedClasses=["8_3", "8_5", "5_3"]
#allowedClasses=[]
taskName = "mnist_shap"
iterations = 1000

#for alpha in alphas:
#    estimatePValuesForClassesSeparation(x, y, taskName, ksAttempts=iterations, pAttempts=0, mlAttempts=0, folder='..\..\PValuesFigures', alpha=alpha, shapCalculation=True, allowedClasses=allowedClasses)

visualizeAndSaveKSSI("..\..\PValuesFigures\PValueLogs", f"{taskName}KSSI", alphas, taskName, iterations)
visualizeAndSaveComplexObjects("..\..\PValuesFigures\PValueLogs", f"{taskName}_ComplexObjects", taskName, iterations, x, y)
summarizeExperiments("..\..\PValuesFigures\PValueLogs", f"{taskName}Summary", y, taskName, iterations)
