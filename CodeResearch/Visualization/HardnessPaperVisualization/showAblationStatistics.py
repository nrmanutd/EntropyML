from CodeResearch.Visualization.HardnessPaperVisualization.visualizationHelpers import evaluateAndSaveAccuracyTable

folder = '..\\..\\CurriculumLearning\\gold_final'
task = ['mnist', 'cifar', 'cifar100', 'svhn']
fraction = ['0.05', '0.1', '0.2', '0.5']
method = ['h', 'GradNorm', 'h_inc', 'GradNorm_inc', 'h&GradNorm_inc']
fileName = 'ablation_statistics_aaai_2027.txt'

evaluateAndSaveAccuracyTable(folder, task, fraction, method, fileName)
