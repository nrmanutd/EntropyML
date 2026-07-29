from CodeResearch.Visualization.HardnessPaperVisualization.Services.extractData import make_grid, \
    extractDataAndSavePng

#folder = '..\\..\\CurriculumLearning\\gold_final'
folder = ''

task = ['cifar100', 'svhn']
fraction = ['0.2', '0.2']
methods = ['h&GradNorm_inc', 'boss', 'rand', 'EL2N', 'GradNorm', 'k-centered_inc']

files = []
for i in range(len(task)):
    t = task[i]
    f = fraction[i]
    file = extractDataAndSavePng(folder, t, f, methods, 'LatexResults')
    files.append(file)

make_grid(files, f'LatexResults/paper_grid.png', nrows=1, ncols=2)