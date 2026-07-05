import numpy as np
from sklearn.preprocessing import LabelEncoder

from CodeResearch.DataSeparationFramework.Metrics.KSMetric import KSMetric
from CodeResearch.Helpers.Logger.SimpleLogger import SimpleLogger
from CodeResearch.LearningFramework.Learners.KSLearner import KSLearner
from CodeResearch.LearningFramework.Learners.XGBoostLearner import XGBoostLearner
from CodeResearch.ObjectComplexity.Hardness import ExpandingDatasetHardnessCalculator
from CodeResearch.ObjectComplexity.Hardness.HardnessCorrector import HardnessCorrector
from CodeResearch.ObjectComplexity.Hardness.KSHardnessCalculator import KSHardnessCalculator
from CodeResearch.ObjectComplexity.Hardness.LearnerBasedHardnessCalculator import LearnerBasedHardnessCalculator
from CodeResearch.ObjectComplexity.ObjectAssessment.StandardAssessor import StandardAssessor
from CodeResearch.ObjectComplexity.ObjectAssessment.XGBoostAssessor import XGBoostAssessor
from CodeResearch.Visualization.visualizeAndSaveComplexObjects import plot_with_custom_brightness, extractData
from CodeResearch.dataSets import load_proteins, generate_lin_reg_dataset

#x, y = load_proteins("../../Data/Proteins/df_master.csv")
x, y = generate_lin_reg_dataset(n_samples=1000, noise=0.1)
logger = SimpleLogger()

nAttempts = 10000
alpha = 0.1

hc1 = KSHardnessCalculator(nAttempts, alpha)
#hc1 = ExpandingDatasetHardnessCalculator.ExpandingDatasetHardnessCalculator(hc1)

np.random.seed(42)
#i1, e1 = hc1.calculateHardness(x, y)

#plot_with_custom_brightness(x, y, (1 - e1) * i1, title='EasinessxImportance1')
#plot_with_custom_brightness(x, y, e1, title='Easiness1')
#plot_with_custom_brightness(x, y, i1, title='Importance1')

assesor = StandardAssessor()
#assesor = XGBoostAssesor()
#hc2 = LearnerBasedHardnessCalculator(KSLearner(KSMetric(), logger), assesor, nAttempts, alpha, logger)
hc2 = LearnerBasedHardnessCalculator(XGBoostLearner(), assesor, nAttempts, alpha, logger)
#hc2 = ExpandingDatasetHardnessCalculator.ExpandingDatasetHardnessCalculator(hc2)
np.random.seed(42)

i2, e2 = hc2.calculateHardness(x, y)

plot_with_custom_brightness(x, y, (1 - e2) * i2, title='EasinessxImportance3')
plot_with_custom_brightness(x, y, e2, title='Easiness3')
plot_with_custom_brightness(x, y, i2, title='Importance3')