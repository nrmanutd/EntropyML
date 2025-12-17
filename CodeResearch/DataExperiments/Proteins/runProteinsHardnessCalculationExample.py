import numpy as np
from sklearn.preprocessing import LabelEncoder

from CodeResearch.ObjectComplexity.Hardness.KSHardnessCalculator import KSHardnessCalculator
from CodeResearch.Visualization.visualizeAndSaveComplexObjects import plot_with_custom_brightness, extractData
from CodeResearch.dataSets import load_proteins, generate_lin_reg_dataset

#x, y = load_proteins("../../Data/Proteins/df_master.csv")
x, y = generate_lin_reg_dataset(n_samples=1000)

hardnessCalculator = KSHardnessCalculator(100, 0.5)
shaps, easiness = hardnessCalculator.calculateHardness(x, y)

plot_with_custom_brightness(x, y, easiness)