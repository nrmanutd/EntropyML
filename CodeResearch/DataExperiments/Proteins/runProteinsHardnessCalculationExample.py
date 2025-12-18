import numpy as np
from sklearn.preprocessing import LabelEncoder

from CodeResearch.DataSeparationFramework.Metrics.KSMetric import KSMetric
from CodeResearch.Helpers.Logger.SimpleLogger import SimpleLogger
from CodeResearch.LearningFramework.Learners.KSLearner import KSLearner
from CodeResearch.ObjectComplexity.Hardness import ExpandingDatasetHardnessCalculator
from CodeResearch.ObjectComplexity.Hardness.HardnessCorrector import HardnessCorrector
from CodeResearch.ObjectComplexity.Hardness.KSHardnessCalculator import KSHardnessCalculator
from CodeResearch.ObjectComplexity.Hardness.LearnerBasedHardnessCalculator import LearnerBasedHardnessCalculator
from CodeResearch.ObjectComplexity.ObjectAssessment.StandardAssesor import StandardAssesor
from CodeResearch.ObjectComplexity.ObjectAssessment.XGBoostAssesor import XGBoostAssesor
from CodeResearch.Visualization.visualizeAndSaveComplexObjects import plot_with_custom_brightness, extractData
from CodeResearch.dataSets import load_proteins, generate_lin_reg_dataset

np.random.seed(42)

#x, y = load_proteins("../../Data/Proteins/df_master.csv")
x, y = generate_lin_reg_dataset(n_samples=2000)
logger = SimpleLogger()

nAttempts = 10000
alpha = 0.5

hc = KSHardnessCalculator(nAttempts, alpha)
assesor = XGBoostAssesor()
#assesor = StandardAssesor()
hc = LearnerBasedHardnessCalculator(KSLearner(KSMetric(), logger), assesor, nAttempts, alpha, logger)
hc = ExpandingDatasetHardnessCalculator.ExpandingDatasetHardnessCalculator(hc)
hc = HardnessCorrector(hc)

importance, easiness = hc.calculateHardness(x, y)

plot_with_custom_brightness(x, y, (1 - easiness) * importance, title='EasinessxImportance')
plot_with_custom_brightness(x, y, easiness, title='Easiness')
plot_with_custom_brightness(x, y, importance, title='Importance')