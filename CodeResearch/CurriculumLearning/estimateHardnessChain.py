import math
import time

import numpy as np

from CodeResearch.CurriculumLearning.clHelpers import filterDataSet, visualizeAndSaveComplexity, \
    plot_object_metrics, plot_distributions_kde, plot_multiple_ecdfs, \
    ecdf_advanced
from CodeResearch.Helpers.Logger.SimpleLogger import SimpleLogger
from CodeResearch.LearningFramework.Learners.TorchLearner import TorchMLPLearner
from CodeResearch.ObjectComplexity.Hardness.Factory import HardnessFactory
from CodeResearch.ObjectComplexity.Hardness.Factory.AssesorEnum import AssesorEnum
from CodeResearch.ObjectComplexity.Hardness.Factory.LearnerEnum import LearnerEnum
from CodeResearch.ObjectComplexity.Hardness.LearnerBasedHardnessCalculator import LearnerBasedHardnessCalculator
from CodeResearch.ObjectComplexity.Hardness.UsefulObjectsCalculator import UsefulObjectsCalculator
from CodeResearch.dataSets import loadCifar100

nAttempts = 50
nSamples = 2000
#x, y = make_random(nSamples)
#x, y = datasets.make_blobs(n_samples=nSamples, centers=2, n_features=2, random_state=42)
#x, y = make_xor(nSamples)
#x, y = datasets.make_circles(n_samples=nSamples, factor=0.5, noise=0.1, random_state=42)
#x, y = make_spirals(nSamples)
#x, y = loadMnist()
x, y = loadCifar100()
#x, y = loadCifar()

firstClass = 43
secondClass = 87
alpha = 0.5
epochs = 20
hidden_sizes = (16, 16)

x, y = filterDataSet(x, y, 1, firstClass, secondClass)
#x, y = load_proteins("../Data/Proteins/df_master.csv")

taskName = 'cifar100'

learner = LearnerEnum.XGBoost
assesor = AssesorEnum.ShapXGBoost

targetPrecedents = 10

logger = SimpleLogger()
nClasses = len(np.unique(y))

t = time.time()

easinesses = []
easinessDeltas = []

importances = []
importancesDeltas = []

prefix = f'{taskName}\\{taskName}_{targetPrecedents}_{firstClass}_{secondClass}'

prevEasiness = np.zeros(x.shape[0])
prevImportance = np.zeros(x.shape[0])

cumulativeEasiness = np.zeros(x.shape[0])
cumulativeImportances = np.zeros(x.shape[0])

totalCumulativeEasiness = []
totalCumulativeImportances = []

usefulObjectsCalculator = UsefulObjectsCalculator()

fractions = np.array([0.05, 0.5, 0.7])
for i in range(len(fractions)):
    fraction = fractions[i]
    currentAttempts = math.ceil(targetPrecedents / fraction)

    print(f'Calculating for fraction {fraction}, attempts: {currentAttempts}')

    l = TorchMLPLearner(input_dim=x.shape[1], num_classes=nClasses, hidden_sizes=hidden_sizes, epochs=epochs)
    a = HardnessFactory.HardnessFactory.createAssesor(AssesorEnum.ShapXGBoost)

    hc = LearnerBasedHardnessCalculator(l, a, currentAttempts, fraction, logger)
    importance, easiness = hc.calculateHardness(x, y)

    easinessDelta = easiness - prevEasiness
    importanceDelta = importance - prevImportance

    easinesses.append(easiness)
    easinessDeltas.append(easinessDelta)

    cumulativeEasiness += easiness
    cumulativeImportances += importance

    totalCumulativeEasiness.append(ecdf_advanced(easiness))
    totalCumulativeImportances.append(ecdf_advanced(importance))

    e1 = totalCumulativeEasiness[0]
    eN = totalCumulativeEasiness[-1]

    if i != 0:
        point = usefulObjectsCalculator.evaluate(easinesses[0], easiness)
        point = [point]

        print(f'first easiness: {len(np.where(easinesses[0] > point)[0])}, second easiness: {len(np.where(easiness > point)[0])} of {len(easiness)}')
    else:
        point = None

    th = [0.01, 0.02, 0.03, 0.1, 0.2, 0.5]

    print(f'-----------------{fraction}-----------------')
    for i in range(len(th)):
        t = th[i]
        if i == 0:
            print(f'Easiness: < {t}: {np.sum(easinessDelta < t)}')
            print(f'Easiness: >= {t}: {np.sum(easinessDelta >= t)}')
        else:
            print(f'Easiness: >= {t}: {np.sum(easinessDelta >= t)}')

    importances.append(importance)
    importancesDeltas.append(importanceDelta)

    prevEasiness = easiness
    prevImportance = importance

    pre = f'{prefix}_{fraction}_easiness'
    plot_distributions_kde(easinesses, title=pre, fileName=f'{pre}.png')
    plot_distributions_kde(easinessDeltas, title=f'{pre}_delta', fileName=f'{pre}_delta.png')
    plot_multiple_ecdfs(totalCumulativeEasiness, point, title=f'{pre}_ecdf', fileName=f'{pre}_ecdf.png')
    visualizeAndSaveComplexity(easiness, easinessDelta, np.ones(len(easiness)), 'easiness', 'easiness delta', f'easiness vs easiness delta {pre}', f'{pre}_vs_delta.png')
    visualizeAndSaveComplexity(easiness, importanceDelta, np.ones(len(easiness)), 'easiness', 'importance delta', f'easiness vs importance delta {pre}', f'{pre}_vs_importance_delta.png')
    visualizeAndSaveComplexity(easinessDelta, importanceDelta, np.ones(len(easiness)), 'easiness delta', 'importance delta',
                               f'easiness delta vs importance delta {pre}', f'{pre}_delta_vs_importance_delta.png')
    plot_object_metrics(easinesses, title=pre, fileName=f'{pre}_chain.png')

    pre = f'{prefix}_{fraction}_importance'
    plot_distributions_kde(importances, title=pre, fileName=f'{pre}.png')
    plot_distributions_kde(importancesDeltas, title=f'{pre}_delta', fileName=f'{pre}_delta.png')
    plot_multiple_ecdfs(totalCumulativeImportances, title=f'{pre}_ecdf', fileName=f'{pre}_ecdf.png')
    visualizeAndSaveComplexity(easiness, importance,  np.ones(len(easiness)), 'easiness', 'importance', f'easiness vs importance {pre}', f'{pre}_vs_easiness.png')
    visualizeAndSaveComplexity(importance, easinessDelta, np.ones(len(easiness)), 'importance', 'easiness delta', f'importance vs easiness delta {pre}', f'{pre}_vs_easinessDelta.png')
    plot_object_metrics(importances, title=pre, fileName=f'{pre}_chain.png')