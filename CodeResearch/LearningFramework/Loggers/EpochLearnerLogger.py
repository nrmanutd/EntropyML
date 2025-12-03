from CodeResearch.CurriculumLearning.clHelpers import processEpochLosses, processLosses
from CodeResearch.Helpers.Logger.BaseLogger import BaseLogger
from CodeResearch.Visualization.saveDataForVisualization import serialize_labeled_list_of_arrays
from CodeResearch.Visualization.visualizeLearningErrors import plot_multi_errors_vs_alpha


class EpochLearnerLogger(BaseLogger):
    def __init__(self, epochs, taskName, prefix, nAttempts, nRepeats, nArrays):
        super().__init__()
        self.nArrays = nArrays
        self.nRepeats = nRepeats
        self.nAttempts = nAttempts
        self.prefix = f'{prefix}_llmh'
        self.taskName = taskName
        self.epochs = epochs
        self.counter = 0
        self.errors = []

    def logConcreteObject(self, object):

        tripleLosses = object
        tripleLosses = processEpochLosses(tripleLosses)
        tripleLosses = processLosses(tripleLosses)

        errors = [[] for _ in range(self.nArrays)]

        for i in range(self.epochs):
            curShift = i

            for arr in errors:
                curError = []

                for k in range(self.nRepeats):
                    curError = curError + tripleLosses[curShift + k * self.epochs].tolist()

                arr.append(curError)
                curShift += self.epochs * self.nRepeats

        xAxis = range(self.epochs)
        labels = ['l (0.05)', 'l (0.1)', 'l (0.5)', 'l (1)', 'h&i (0.05)', 'h&i (0.1)', 'h&i (0.5)', 'h&i (1)']
        for i in range(len(errors)):
            err = errors[i]
            serialize_labeled_list_of_arrays(err, [f'{k}_{labels[i]}' for k in range(len(err))], self.prefix, self.nAttempts,
                                             f'{self.taskName}\\{self.prefix}_{labels[i]}_{self.counter}_data.txt')

        plot_multi_errors_vs_alpha(errors, xAxis, labels, self.taskName, f'{self.prefix}_{self.counter}')
        plot_multi_errors_vs_alpha(errors, xAxis, labels, self.taskName, f'{self.prefix}_5_{self.counter}', 5)

        self.counter += 1

        return