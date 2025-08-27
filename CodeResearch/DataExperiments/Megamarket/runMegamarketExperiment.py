from CodeResearch.dataSets import load_megamarket
from CodeResearch.estimateAndVisualizeEmpiricalDistributionDelta import estimatePValuesForClassesSeparation

x, y = load_megamarket("../../Data/megamarket/sampled_10k.parquet")

estimatePValuesForClassesSeparation(x, y, 'megamarket', ksAttempts=200, pAttempts=0, mlAttempts=0, folder='..\..\PValuesFigures')