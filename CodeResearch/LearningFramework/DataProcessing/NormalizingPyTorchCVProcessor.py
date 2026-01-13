from CodeResearch.LearningFramework.DataProcessing.BaseDataProcessor import BaseDataProcessor


class NormalizingPyTorchCVProcessor(BaseDataProcessor):
    def applyParametersToData(self, dataSet, target, parameters):
        mean = parameters[0]
        std = parameters[1]

        X = (dataSet - mean) / (std + 1e-8)
        return X, target

    def estimateDataTransformationParameters(self, dataSet, target):
        print(f'Number of elements to normalize: {len(target)}')
        if len(target) <= 1:
            raise ValueError()

        mean = dataSet.mean(dim=(0, 2, 3), keepdim=True)
        std = dataSet.std(dim=(0, 2, 3), keepdim=True, unbiased=False)

        return (mean, std)
