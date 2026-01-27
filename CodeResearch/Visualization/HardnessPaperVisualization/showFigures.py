from CodeResearch.Visualization.HardnessPaperVisualization.extractData import extractAndSave, make_grid

folder = '..\\..\\CurriculumLearning\\gold'
task = ['MNIST', 'CIFAR 10', 'CIFAR 100']
targetLength = [80, 40, 40]
fraction = ['0.1', '0.2', '0.5']
protocol = ['fixed test', 'random subset']

for it in range(len(task)):
    for f in fraction:
        for p in protocol:
            t = task[it]
            tl = targetLength[it]
            extractAndSave(folder, t, tl, f, p)

for t in task:
    files = []
    for p in protocol:
        for f in fraction:
            curFile = f"{t.lower()}\\{t.lower()}_{p}_{f}_errors.png"
            files.append(curFile)

    make_grid(files, t)
