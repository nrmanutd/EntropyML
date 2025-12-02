from CodeResearch.CurriculumLearning.clHelpers import processEpochLosses, processLosses
from CodeResearch.Helpers.Logger.BaseLogger import BaseLogger
from CodeResearch.Visualization.saveDataForVisualization import serialize_labeled_list_of_arrays
from CodeResearch.Visualization.visualizeLearningErrors import plot_multi_errors_vs_alpha


class EpochLearnerLogger(BaseLogger):
    def __init__(self, epochs, taskName, prefix, nAttempts):
        self.nAttempts = nAttempts
        self.prefix = prefix
        self.taskName = taskName
        self.epochs = epochs
        self.counter = 0

    def logConcreteObject(self, object):

        tripleLosses = object
        tripleLosses = processEpochLosses(tripleLosses)
        tripleLosses = processLosses(tripleLosses)

        losses = []
        lossesHardness = []
        lossesImportant = []
        lossesHardAndImportant = []

        for i in range(self.epochs):
            curShift = i
            losses.append(tripleLosses[curShift])
            lossesImportant.append(tripleLosses[curShift + self.epochs])
            lossesHardness.append(tripleLosses[curShift + 2 * self.epochs])
            lossesHardAndImportant.append(tripleLosses[curShift + 3 * self.epochs])

        xAxis = range(self.epochs)
        labels = ['l', 'hard', 'important', 'both']
        errors = [losses, lossesHardness, lossesImportant, lossesHardAndImportant]
        for i in range(len(errors)):
            err = errors[i]
            serialize_labeled_list_of_arrays(err, [f'{k}_{labels[i]}' for k in range(len(err))], self.prefix, self.nAttempts,
                                             f'{self.taskName}\\{self.prefix}_{labels[i]}_{self.counter}_data.txt')

        plot_multi_errors_vs_alpha(errors, xAxis, labels, self.taskName, f'{self.prefix}_{self.counter}')

        self.counter += 1

        pass