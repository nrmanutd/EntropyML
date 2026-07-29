from CodeResearch.Visualization.HardnessPaperVisualization.LatexConstants import constants
from CodeResearch.Visualization.HardnessPaperVisualization.Services.visualizationHelpers import \
    evaluateAndSaveCITableAndRankingsTable

#folder = '..\\..\\CurriculumLearning\\gold_final'
folder = ''

task = ['mnist', 'cifar', 'cifar100', 'svhn']
#task = ['cifar']
fraction = ['0.05', '0.1', '0.2', '0.5']

method = ['h', 'GradNorm', 'hGradNorm', 'h_inc', 'GradNorm_inc']
targetMethod = 'h&GradNorm_inc'
fileName = 'LatexResults/ablation_ci_statistics_aaai_2027.txt'
ranksFileName = 'LatexResults/ablation_ranks_table_aaai_2027.txt'

latexMarkup = {'ciBegin': constants.ablationCITableBegin, 'ciEnd': constants.ablationCITableEnd, 'rankBegin': constants.ablationRankingTableBegin, 'rankEnd': constants.ablationRankingTableEnd}
evaluateAndSaveCITableAndRankingsTable(folder, task, fraction, method, targetMethod, fileName, ranksFileName, latexMarkup)
