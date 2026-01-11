from typing import Callable, Optional, Union, Any, Tuple

from abc import abstractmethod

import torch
import torch.nn as nn
from CodeResearch.LearningFramework.Learners.baseLearner import BaseLearner


class TorchLearner(BaseLearner):

    def __init__(self, device: Optional[Union[str, torch.device]] = None):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

    @abstractmethod
    def build_model(self) -> nn.Module:
        pass
