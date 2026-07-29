from CodeResearch.Visualization.HardnessPaperVisualization.LatexConstants import constants
from CodeResearch.Visualization.HardnessPaperVisualization.Services.visualizationHelpers import \
    evaluateAndSaveCITableAndRankingsTable

#folder = '..\\..\\CurriculumLearning\\gold_final'
folder = ''

task = ['mnist', 'cifar', 'cifar100', 'svhn']
fraction = ['0.05', '0.1', '0.2', '0.5']

method = ['rand', 'EL2N', 'GradNorm', 'chg_inc', 'forgetting', 'k-centered_inc', 'boss']
targetMethod = 'h&GradNorm_inc'
fileName = 'LatexResults/ci_statistics_aaai_2027.txt'
ranksFileName = 'LatexResults/ranks_ci_statistics_aaai_2027.txt'

latexMarkup = {'ciBegin': constants.ciTableBegin, 'ciEnd': constants.ciTableEnd, 'rankBegin': constants.rankingTableBegin, 'rankEnd': constants.rankingTableEnd}
evaluateAndSaveCITableAndRankingsTable(folder, task, fraction, method, targetMethod, fileName, ranksFileName, latexMarkup)
