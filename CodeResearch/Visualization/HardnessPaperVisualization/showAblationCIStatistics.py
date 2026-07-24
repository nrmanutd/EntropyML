from CodeResearch.Visualization.HardnessPaperVisualization.visualizationHelpers import \
    evaluateAndSaveCITableAndRankingsTable

folder = '..\\..\\CurriculumLearning\\gold_final'
task = ['mnist', 'cifar', 'cifar100', 'svhn']
#task = ['cifar']
fraction = ['0.05', '0.1', '0.2', '0.5']

method = ['h', 'GradNorm', 'h_inc', 'GradNorm_inc']
targetMethod = 'h&GradNorm_inc'
fileName = 'ablation_ci_statistics_aaai_2027.txt'
ranksFileName = 'ablation_ranks_table_aaai_2027.txt'

evaluateAndSaveCITableAndRankingsTable(folder, task, fraction, method, targetMethod, fileName, ranksFileName)
