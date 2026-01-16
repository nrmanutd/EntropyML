import torch
import numpy as np

from CodeResearch.LearningFramework.DataProcessing.BaseDataProcessor import BaseDataProcessor


class NormalizingPyTorchCVProcessor(BaseDataProcessor):
    def applyParametersToData(self, dataSet, target, parameters):
        mean = parameters[0]
        std = parameters[1]

        X = (dataSet - mean) / (std + 1e-8)
        return X, target

    def estimateDataTransformationParameters(self, dataSet, target):
        if len(target) <= 1:
            raise ValueError(f'Number of elements to normalize: {len(target)}')

        if isinstance(dataSet, torch.Tensor):
            mean = dataSet.mean(dim=(0, 2, 3), keepdim=True)
            std = dataSet.std(dim=(0, 2, 3), keepdim=True, unbiased=False)
            return mean, std
        elif isinstance(dataSet, np.ndarray):
            mean = np.mean(dataSet, axis=(0, 2, 3), keepdims=True)
            std = np.std(dataSet, axis=(0, 2, 3), keepdims=True, ddof=0)

            return mean, std
        else:
            raise TypeError(f'Unsupported data type: {type(dataSet)}. Expected torch.Tensor or np.ndarray')
