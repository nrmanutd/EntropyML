from CodeResearch.DataSeparationFramework.Metrics.KSMetric import KSMetric
from CodeResearch.Helpers.Logger import BaseLogger
from CodeResearch.Helpers.Logger.SimpleLogger import SimpleLogger
from CodeResearch.LearningFramework.Learners.KSLearner import KSLearner
from CodeResearch.LearningFramework.Learners.XGBoostLearner import XGBoostLearner
from CodeResearch.LearningFramework.Learners.baseLearner import BaseLearner
from CodeResearch.ObjectComplexity.Hardness import BaseHardnessCalculator
from CodeResearch.ObjectComplexity.Hardness.ExpandingDatasetHardnessCalculator import ExpandingDatasetHardnessCalculator
from CodeResearch.ObjectComplexity.Hardness.Factory import LearnerEnum, AssesorEnum
from CodeResearch.ObjectComplexity.Hardness.LearnerBasedHardnessCalculator import LearnerBasedHardnessCalculator
from CodeResearch.ObjectComplexity.ObjectAssessment import BaseObjectAssessor
from CodeResearch.ObjectComplexity.ObjectAssessment.EasinessOnlyAssessor import EasinessOnlyAssessor
from CodeResearch.ObjectComplexity.ObjectAssessment.StandardAssessor import StandardAssessor
from CodeResearch.ObjectComplexity.ObjectAssessment.XGBoostAssessor import XGBoostAssessor


class HardnessFactory:

    @staticmethod
    def createHardnessCalculatorWithLogger(learner: LearnerEnum.LearnerEnum, assesor: AssesorEnum.AssesorEnum, nAttempts,
                                 fraction, logger: BaseLogger) -> BaseHardnessCalculator:
        l = HardnessFactory.createLearner(learner, logger)
        a = HardnessFactory.createAssesor(assesor)

        hc = LearnerBasedHardnessCalculator(l, a, nAttempts, fraction, logger)
        hc = HardnessFactory.updateForKS(learner, hc)
        return hc

    @staticmethod
    def updateForKS(learner: LearnerEnum.LearnerEnum, hc: BaseHardnessCalculator) -> BaseHardnessCalculator:
        if learner == LearnerEnum.LearnerEnum.KS:
            hc = ExpandingDatasetHardnessCalculator(hc)

        return hc

    @staticmethod
    def createHardnessCalculator(learner: LearnerEnum.LearnerEnum, assesor: AssesorEnum.AssesorEnum,  nAttempts, fraction) -> BaseHardnessCalculator:
        logger = SimpleLogger()

        return HardnessFactory.createHardnessCalculatorWithLogger(learner, assesor, nAttempts, fraction, logger)

    @staticmethod
    def createLearner(learner: LearnerEnum.LearnerEnum, logger: BaseLogger.BaseLogger) -> BaseLearner:
        if learner == LearnerEnum.LearnerEnum.KS:
            return KSLearner(KSMetric(), logger)

        if learner == LearnerEnum.LearnerEnum.XGBoost:
            return XGBoostLearner()

        raise ValueError(f'Unsupported type of learner: {learner}')

    @staticmethod
    def createAssesor(assesor: AssesorEnum.AssesorEnum) -> BaseObjectAssessor.BaseObjectAssessor:
        if assesor == AssesorEnum.AssesorEnum.Standard:
            return StandardAssessor()

        if assesor == AssesorEnum.AssesorEnum.ShapXGBoost:
            return XGBoostAssessor()

        if assesor == AssesorEnum.AssesorEnum.EasinessOnly:
            return EasinessOnlyAssessor()

        raise ValueError(f'Unsupported type of learner: {assesor}')