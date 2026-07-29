from CodeResearch.Visualization.HardnessPaperVisualization.LatexConstants import constants
from CodeResearch.Visualization.HardnessPaperVisualization.Services.visualizationHelpers import evaluateAndSaveAccuracyTable

#folder = '../../CurriculumLearning/gold_final'
folder = ''

task = ['mnist', 'cifar', 'cifar100', 'svhn']
fraction = ['0.05', '0.1', '0.2', '0.5']
method = ['h', 'GradNorm', 'hGradNorm', 'h_inc', 'GradNorm_inc', 'h&GradNorm_inc']
fileName = 'LatexResults/ablation_statistics_aaai_2027.txt'

evaluateAndSaveAccuracyTable(folder, task, fraction, method, fileName, constants.ablationAccuracyTableBegin, constants.ablationAccuracyTableEnd)
