from abc import abstractmethod

from CodeResearch.DataSeparationFramework.BaseDataSeparationCalculator import BaseDataSeparationCalculator
from CodeResearch.Visualization.VisualizeAndSaveCommonTopSubsamples import visualizeAndSaveKSForEachPair
from CodeResearch.Visualization.saveDataForVisualization import serialize_labeled_list_of_arrays


class SimpleDataSeparationCalculator(BaseDataSeparationCalculator):
    def __init__(self, dataSet, target, attempts, name, taskName, folder, logsFolder):

        self.logsFolder = logsFolder
        self.dataSet = dataSet
        self.target = target
        self.attempts = attempts
        self.name = name
        self.taskName = taskName
        self.folder = folder

        self.objectsCount = []
        self.commonPermutationPairs = []
        self.labels = []

    @abstractmethod
    def calculateMetric(self, objects, iClass, jClass):
        pass

    def processCalculatedMetric(self, data):
        pass

    def serializeCalculatedData(self):
        pass

    def calculateDataSeparability(self, objects, iClass, jClass):
        curPair = f'{iClass}_{jClass}'

        metricValues = self.calculateMetric(objects, iClass, jClass)

        if len(metricValues[0]) > 0:
            self.commonPermutationPairs.append(metricValues[0])
            self.processCalculatedMetric(metricValues)

        self.labels.append(curPair)
        self.objectsCount.append(objects)

    def serializeResults(self):
        curPair = self.labels[-1]
        currentObjects = self.objectsCount[-1]

        if len(self.commonPermutationPairs) > 0:
            visualizeAndSaveKSForEachPair(self.commonPermutationPairs, self.labels,
                                      f'{self.taskName}_{self.name}', self.attempts,
                                      curPair, self.folder)


            serialize_labeled_list_of_arrays(self.commonPermutationPairs, self.labels,
                                         f'{self.taskName}_{self.name}', self.attempts,
                                         f'{self.logsFolder}\\{self.name}_{self.taskName}_{self.attempts}_{curPair}_{currentObjects}.txt')

            self.serializeCalculatedData()


