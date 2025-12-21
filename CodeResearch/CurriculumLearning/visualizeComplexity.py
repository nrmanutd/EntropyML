import math
import time

import numpy as np

from CodeResearch.CurriculumLearning.clHelpers import filterDataSet, visualizeAndSaveComplexity, \
    plot_distributions_kde_with_metrics
from CodeResearch.ObjectComplexity.Hardness.Factory import HardnessFactory
from CodeResearch.ObjectComplexity.Hardness.Factory.AssesorEnum import AssesorEnum
from CodeResearch.ObjectComplexity.Hardness.Factory.LearnerEnum import LearnerEnum
from CodeResearch.ObjectComplexity.Hardness.HardnessCorrector import HardnessCorrector
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

firstClass = 47
secondClass = 52
alpha = 0.5

x, y = filterDataSet(x, y, 1, firstClass, secondClass)
#x, y = load_proteins("../Data/Proteins/df_master.csv")

x = np.hstack((x, -x))

taskName = 'cifar100'
fractions = [0.05, 0.1, 0.2, 0.5]

learner = LearnerEnum.KS
assesor = AssesorEnum.ShapXGBoost

targetPrecedents = 100

t = time.time()
for fraction in fractions:
    currentAttempts = math.ceil(targetPrecedents / fraction)
    print(f'Calculating for fraction {fraction}, attempts: {currentAttempts}')

    hc = HardnessFactory.HardnessFactory.createHardnessCalculator(learner, assesor, currentAttempts, fraction)
    importance, easiness = hc.calculateHardness(x, y)

    prefix = f'{taskName}\\{taskName}_{fraction}_{currentAttempts}_{fraction}_{firstClass}_{secondClass}_{learner}_{assesor}_CDF'
    plot_distributions_kde_with_metrics(easiness, importance, f'{prefix}_distribution.png')

    hc1 = HardnessCorrector(hc)
    cdfEasiness = hc1.convertToECDF(easiness)
    cdfImportance = hc1.convertToECDF(importance)

    plot_distributions_kde_with_metrics(cdfEasiness, cdfImportance, f'{prefix}_cdf_distribution.png')

    visualizeAndSaveComplexity(easiness, importance, easiness * importance, f'{prefix}_simple_prod_complexity.png')
    visualizeAndSaveComplexity(easiness, importance, cdfEasiness * cdfImportance, f'{prefix}_simple_prod_complexity_marked_cdf.png')

    n = len(importance)
    eps = 1 / (2 * n)

    score = np.exp(alpha * np.log(eps + cdfImportance) + (1 - alpha) * np.log(eps + cdfEasiness))
    visualizeAndSaveComplexity(easiness, importance, score, f'{prefix}_score_complexity_marked_cdf.png')

    easyThresholds = [0.5, 0.6, 0.7, 0.8, 0.9]
    #for easy in easyThresholds:
    #    easyIdx = np.where(easiness > easy)[0]
    #    plot_distributions_kde_with_metrics(easiness[easyIdx], importance[easyIdx], f'{prefix}_distribution_conditioned_easy_{easy}.png')

    #for important in easyThresholds:
    #    importantIdx = np.where(importance > important)[0]
    #    plot_distributions_kde_with_metrics(easiness[importantIdx], importance[importantIdx], f'{prefix}_distribution_conditioned_important_{important}.png')

    print(f'Visualization for {fraction}, time: {time.time() - t} s')