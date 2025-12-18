import numpy as np
from sklearn.preprocessing import LabelEncoder

from CodeResearch.ObjectComplexity.Hardness import ExpandingDatasetHardnessCalculator
from CodeResearch.ObjectComplexity.Hardness.HardnessCorrector import HardnessCorrector
from CodeResearch.ObjectComplexity.Hardness.KSHardnessCalculator import KSHardnessCalculator
from CodeResearch.Visualization.visualizeAndSaveComplexObjects import plot_with_custom_brightness, extractData
from CodeResearch.dataSets import load_proteins, generate_lin_reg_dataset

#x, y = load_proteins("../../Data/Proteins/df_master.csv")
x, y = generate_lin_reg_dataset(n_samples=2000)

nAttempts = 1000
hardnessCalculator = KSHardnessCalculator(nAttempts, 0.5)

importance, easiness = hardnessCalculator.calculateHardness(x, y)

plot_with_custom_brightness(x, y, (1 - easiness) * importance, title='EasinessxImportance')
plot_with_custom_brightness(x, y, 1 - easiness, title='Easiness')
plot_with_custom_brightness(x, y, importance, title='Importance')