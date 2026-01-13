from CodeResearch.LearningFramework.DataProcessing.BaseDataProcessor import BaseDataProcessor


class StubDataProcessor(BaseDataProcessor):
    def applyParametersToData(self, dataSet, target, parameters):
        return dataSet, target

    def estimateDataTransformationParameters(self, dataSet, target):
        return None