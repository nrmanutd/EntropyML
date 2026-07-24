from CodeResearch.Visualization.HardnessPaperVisualization.visualizationHelpers import \
    evaluateAndSaveCITableAndRankingsTable

folder = '..\\..\\CurriculumLearning\\gold_final'
task = ['mnist', 'cifar', 'cifar100', 'svhn']
#task = ['cifar']
fraction = ['0.05', '0.1', '0.2', '0.5']

method = ['rand', 'EL2N', 'GradNorm', 'chg_inc', 'forgetting', 'k-centered_inc', 'boss']
targetMethod = 'h&GradNorm_inc'
fileName = 'ci_statistics_aaai_2027.txt'
ranksFileName = 'ranks_ci_statistics_aaai_2027.txt'

evaluateAndSaveCITableAndRankingsTable(folder, task, fraction, method, targetMethod, fileName, ranksFileName)
