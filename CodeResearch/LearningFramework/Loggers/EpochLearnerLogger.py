from CodeResearch.CurriculumLearning.clHelpers import processEpochLosses, processLosses
from CodeResearch.Helpers.Logger.BaseLogger import BaseLogger
from CodeResearch.Visualization.saveDataForVisualization import serialize_labeled_list_of_arrays
from CodeResearch.Visualization.visualizeLearningErrors import plot_multi_errors_vs_alpha


class EpochLearnerLogger(BaseLogger):
    def __init__(self, epochs, taskName, prefix, nAttempts):
        self.nAttempts = nAttempts
        self.prefix = f'{prefix}_lmh'
        self.taskName = taskName
        self.epochs = epochs
        self.counter = 0

    def logConcreteObject(self, object):

        tripleLosses = object
        tripleLosses = processEpochLosses(tripleLosses)
        tripleLosses = processLosses(tripleLosses)

        losses = []
        lossesl = []
        lossesm = []

        lossesHardAndImportant = []
        lossesHardAndImportantl = []
        lossesHardAndImportantm = []

        for i in range(self.epochs):
            curShift = i
            lossesl.append(tripleLosses[curShift])
            lossesm.append(tripleLosses[curShift + self.epochs])
            losses.append(tripleLosses[curShift + 2 * self.epochs])

            lossesHardAndImportantl.append(tripleLosses[curShift + 3 * self.epochs])
            lossesHardAndImportantm.append(tripleLosses[curShift + 4 * self.epochs])
            lossesHardAndImportant.append(tripleLosses[curShift + 5 * self.epochs])

        xAxis = range(self.epochs)
        labels = ['l (0.1)', 'l (0.5)', 'l (1)',  'h&i (0.5)', 'h&i (0.1)', 'h&i (1)']
        errors = [lossesl, lossesm, losses, lossesHardAndImportantl, lossesHardAndImportantm, lossesHardAndImportant]
        for i in range(len(errors)):
            err = errors[i]
            serialize_labeled_list_of_arrays(err, [f'{k}_{labels[i]}' for k in range(len(err))], self.prefix, self.nAttempts,
                                             f'{self.taskName}\\{self.prefix}_{labels[i]}_{self.counter}_data.txt')

        plot_multi_errors_vs_alpha(errors, xAxis, labels, self.taskName, f'{self.prefix}_{self.counter}')

        self.counter += 1

        pass