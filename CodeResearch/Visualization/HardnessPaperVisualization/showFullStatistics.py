from CodeResearch.Visualization.HardnessPaperVisualization.LatexConstants import constants
from CodeResearch.Visualization.HardnessPaperVisualization.Services.visualizationHelpers import evaluateAndSaveAccuracyTable

folder = '..\\..\\CurriculumLearning\\gold_final'
task = ['mnist', 'cifar', 'cifar100', 'svhn']
fraction = ['0.05', '0.1', '0.2', '0.5']
method = ['rand', 'EL2N', 'GradNorm', 'chg_inc', 'forgetting', 'k-centered_inc', 'boss', 'h&GradNorm_inc']
fileName = 'LatexResults/statistics_aaai_2027.txt'

evaluateAndSaveAccuracyTable(folder, task, fraction, method, fileName, constants.accuracyTableBegin, constants.accuracyTableEnd)
