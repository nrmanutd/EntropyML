from CodeResearch.CurriculumLearning.runLearningForConcreteTask import runLearningForConcreteTask

folders = ['mnist_epoch', 'cifar_epoch', 'cifar100_epoch', 'svhn_epoch']

for folder in folders:
    runLearningForConcreteTask(folder)