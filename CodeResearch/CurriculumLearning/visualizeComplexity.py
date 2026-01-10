import math
import time

import numpy as np

from CodeResearch.CurriculumLearning.clHelpers import filterDataSet, visualizeAndSaveComplexity, \
    plot_distributions_kde_with_metrics
from CodeResearch.Helpers.Logger.SimpleLogger import SimpleLogger
from CodeResearch.LearningFramework.Learners.TorchLearner import TorchMLPLearner
from CodeResearch.LearningFramework.Samplers.Batches.priorityBasedSampler import PriorityBasedSampler
from CodeResearch.ObjectComplexity.Diversity.NNBasedObjectDiversifier import NNBasedObjectDiversifier
from CodeResearch.ObjectComplexity.Diversity.SeparableObjectDiversifier import SeparableObjectDiversifier
from CodeResearch.ObjectComplexity.Hardness.DiversityBasedHardnessCalculator import DiversityBasedHardnessCalculator
from CodeResearch.ObjectComplexity.Hardness.Factory import HardnessFactory
from CodeResearch.ObjectComplexity.Hardness.Factory.AssesorEnum import AssesorEnum
from CodeResearch.ObjectComplexity.Hardness.Factory.LearnerEnum import LearnerEnum
from CodeResearch.ObjectComplexity.Hardness.HardnessCorrector import HardnessCorrector
from CodeResearch.ObjectComplexity.Hardness.LearnerBasedHardnessCalculator import LearnerBasedHardnessCalculator
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

dHidden_sizes = (16, 16)
dbatchSize = 10
dEpochs = 20

x, y = filterDataSet(x, y, 1, firstClass, secondClass)
#x, y = load_proteins("../Data/Proteins/df_master.csv")

taskName = 'cifar100'
fractions = [0.05, 0.1, 0.2, 0.5]

learner = LearnerEnum.XGBoost
assesor = AssesorEnum.ShapXGBoost

targetPrecedents = 100

logger = SimpleLogger()
nClasses = len(np.unique(y))
nFeatures = x.shape[1]

t = time.time()
for fraction in fractions:
    #currentAttempts = math.ceil(targetPrecedents / fraction)
    currentAttempts = targetPrecedents
    print(f'Calculating for fraction {fraction}, attempts: {currentAttempts}')

    l = TorchMLPLearner(input_dim=nFeatures, num_classes=nClasses, hidden_sizes=hidden_sizes, epochs=epochs)
    a = HardnessFactory.HardnessFactory.createAssesor(AssesorEnum.ShapXGBoost)

    hc = LearnerBasedHardnessCalculator(l, a, currentAttempts, logger)
    #dcLearner = TorchMLPLearner(input_dim=nFeatures, num_classes=nClasses, hidden_sizes=dHidden_sizes, update_epochs=1)
    #hc = DiversityBasedHardnessCalculator(hc, lambda p: NNBasedObjectDiversifier(dcLearner, lambda ds, t: PriorityBasedSampler(ds, t, dbatchSize,p), dEpochs, logger))

    dcLearner = TorchMLPLearner(input_dim=nFeatures, num_classes=nClasses, hidden_sizes=dHidden_sizes, update_epochs=1,
                                epochs=dEpochs, batch_size=dbatchSize)
    # hc = DiversityBasedHardnessCalculator(hc, lambda x: NNBasedObjectDiversifier(dcLearner, lambda ds, t: PriorityBasedSampler(ds, t, batchSize, x), dEpochs, logger))
    hc = DiversityBasedHardnessCalculator(hc, lambda x: SeparableObjectDiversifier(dcLearner, nAttempts, logger))

    #hc = HardnessCorrector(hc)


    importance, easiness = hc.calculateHardness(x, y, None, None, fraction)

    prefix = f'{taskName}\\{taskName}_{fraction}_{currentAttempts}_{targetPrecedents}_{fraction}_{firstClass}_{secondClass}_{learner}_{assesor}_hc_sep'
    plot_distributions_kde_with_metrics(easiness, importance, f'{prefix}_distribution.png')


    plot_distributions_kde_with_metrics(easiness, importance, f'{prefix}_cdf_distribution.png')

    visualizeAndSaveComplexity(easiness, importance, easiness * importance, 'easiness', 'importance', f'{prefix}_easiness_times_importance',  f'{prefix}_simple_prod_complexity.png')

    n = len(importance)
    eps = 1 / (2 * n)

    score = np.exp(alpha * np.log(eps + importance) + (1 - alpha) * np.log(eps + easiness))
    visualizeAndSaveComplexity(easiness, importance, score, 'easiness', 'importance', f'{prefix}_easiness_importance_score', f'{prefix}_score_complexity_marked_cdf.png')

    easyThresholds = [0.5, 0.6, 0.7, 0.8, 0.9]
    #for easy in easyThresholds:
    #    easyIdx = np.where(easiness > easy)[0]
    #    plot_distributions_kde_with_metrics(easiness[easyIdx], importance[easyIdx], f'{prefix}_distribution_conditioned_easy_{easy}.png')

    #for important in easyThresholds:
    #    importantIdx = np.where(importance > important)[0]
    #    plot_distributions_kde_with_metrics(easiness[importantIdx], importance[importantIdx], f'{prefix}_distribution_conditioned_important_{important}.png')

    print(f'Visualization for {fraction}, time: {time.time() - t} s')