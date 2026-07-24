from CodeResearch.Visualization.HardnessPaperVisualization.Services.extractData import extractAndSave, make_grid

folder = '..\\..\\CurriculumLearning\\extragold'
task = ['CIFAR 10']
targetLength = [40, 40, 40]
fraction = ['0.1', '0.2', '0.5']
protocol = ['fixed test']

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

    make_grid(files[:2], f'{t}_additional', nrows=1, ncols=2)
