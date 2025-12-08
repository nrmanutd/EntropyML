from CodeResearch.ObjectComplexity.Hardness.KSHardnessCalculator import KSHardnessCalculator
from CodeResearch.dataSets import load_proteins

x, y = load_proteins("../../Data/Proteins/df_master.csv")

hardnessCalculator = KSHardnessCalculator(100, 0.5)
shaps, easiness = hardnessCalculator.calculateHardness(x, y)

print(shaps)
print(easiness)