from CodeResearch.CurriculumLearning.clServices.clHelpers import processEpochLosses, processLosses
from CodeResearch.Helpers.Logger.BaseLogger import BaseLogger
from CodeResearch.Visualization.saveDataForVisualization import serialize_labeled_list_of_arrays
from CodeResearch.Visualization.visualizeLearningErrors import plot_multi_errors_vs_alpha_std


class EpochLearnerLogger(BaseLogger):
    def __init__(self, epochs, taskName, prefix, nAttempts, nRepeats, nArrays, betas, baseLabels):
        super().__init__()
        self.baseLabels = baseLabels
        self.betas = betas
        self.nArrays = nArrays
        self.nRepeats = nRepeats
        self.nAttempts = nAttempts
        self.prefix = f'{prefix}_chain_main_standard_incremental_top_fraction_sampled'
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

        if len(self.baseLabels) != int(self.nArrays / len(self.betas)):
            raise ValueError('Incorrect number of labels in baseLabels array')

        labels = []
        for l in self.baseLabels:
            labels = labels + [f'{l} ({beta})' for beta in self.betas]

        for i in range(len(errors)):
            err = errors[i]
            serialize_labeled_list_of_arrays(err, [f'{k}_{labels[i]}' for k in range(len(err))], self.prefix, self.nAttempts,
                                             f'{self.taskName}\\{self.prefix}_{labels[i]}_{self.counter}_data.txt')

        plot_multi_errors_vs_alpha_std(errors, xAxis, labels, self.taskName, f'{self.prefix}_{self.counter}', len(self.betas))
        plot_multi_errors_vs_alpha_std(errors, xAxis, labels, self.taskName, f'{self.prefix}_5_{self.counter}', len(self.betas), 5)

        self.counter += 1

        return