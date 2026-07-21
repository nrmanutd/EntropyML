from CodeResearch.Visualization.HardnessPaperVisualization.extractData import extractAndSave, make_grid, \
    extractDataAndSave

folder = '..\\..\\CurriculumLearning\\gold_v3'
#task = ['cifar', 'cifar100', 'svhn', 'mnist']
task=['mnist']
fraction = ['0.05', '0.1', '0.2', '0.5']

for it in range(len(task)):
    for f in fraction:
        t = task[it]
        extractDataAndSave(folder, t, f)

for t in task:
    files = []
    for f in fraction:
        curFile = f"{folder}\\{t}_epoch\\{t.lower()}_{f}_errors.png"
        files.append(curFile)

    print(files)
    make_grid(files, f'{folder}\\{t}_epoch\\grid_pictures.png', nrows=1, ncols=4)
