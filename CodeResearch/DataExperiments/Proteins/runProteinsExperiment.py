import numpy as np

from CodeResearch.Helpers.commonHelpers import perform_pca
from CodeResearch.dataSets import load_proteins
from CodeResearch.estimateAndVisualizeEmpiricalDistributionDelta import estimatePValuesForClassesSeparation

x, y = load_proteins("../../Data/Proteins/df_master.csv")
xx = perform_pca(x, 10)

median_val = np.median(y)
yy = np.where(y >= median_val, 1, 0)

estimatePValuesForClassesSeparation(xx, yy, 'proteins', ksAttempts=100, pAttempts=100, mlAttempts=100, folder='..\..\PValuesFigures')