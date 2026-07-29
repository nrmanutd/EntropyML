from CodeResearch.Visualization.HardnessPaperVisualization.Services.extractData import make_grid, \
    extractDataAndSavePng

#folder = '..\\..\\CurriculumLearning\\gold_final'
folder = ''

task = ['cifar', 'cifar100', 'svhn', 'mnist']
fraction = ['0.05', '0.1', '0.2', '0.5']
methods = ['h&GradNorm_inc', 'boss', 'rand', 'EL2N', 'GradNorm', 'k-centered_inc']

for i in range(len(task)):
    files = []
    for j in range(len(fraction)):
        t = task[i]
        f = fraction[j]

        file = extractDataAndSavePng(folder, t, f, methods, 'LatexResults')
        files.append(file)

    make_grid(files, f'LatexResults/{task[i]}_all_budgets.png', nrows=2, ncols=2    )