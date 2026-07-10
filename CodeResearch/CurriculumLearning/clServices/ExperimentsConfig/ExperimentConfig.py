from dataclasses import dataclass

from CodeResearch.CurriculumLearning.clServices.PyTorchCVCLLearnersFactory import PyTorchCVCLLearnersFactory
from CodeResearch.LearningFramework.DataProcessing.BaseDataProcessor import BaseDataProcessor
from CodeResearch.LearningFramework.Learners.NNTorchModelLearner import TorchModelLearner
from typing import Callable, Optional

@dataclass
class ExperimentConfig:
    learnerFactory: Optional[PyTorchCVCLLearnersFactory] = None

    noincrementEpochs: Optional[int] = None
    easinessEpochs: Optional[int] = None
    diversityEpochs: Optional[int] = None

    noincrementAttempts: Optional[int] = None

    scoreDiversityLearnerBuilder: Optional[Callable[[int], TorchModelLearner]] = None
    scoreHardnessLearnerBuilder: Optional[Callable[[int], TorchModelLearner]] = None

    targetLearnerCreator: Optional[Callable[[int], TorchModelLearner]] = None
    dataProcessor: Optional[BaseDataProcessor] = None