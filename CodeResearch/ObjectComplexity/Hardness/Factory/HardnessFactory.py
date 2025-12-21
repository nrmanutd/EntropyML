from CodeResearch.DataSeparationFramework.Metrics.KSMetric import KSMetric
from CodeResearch.Helpers.Logger import BaseLogger
from CodeResearch.Helpers.Logger.SimpleLogger import SimpleLogger
from CodeResearch.LearningFramework.Learners.KSLearner import KSLearner
from CodeResearch.LearningFramework.Learners.XGBoostLearner import XGBoostLearner
from CodeResearch.LearningFramework.Learners.baseLearner import BaseLearner
from CodeResearch.ObjectComplexity.Hardness import BaseHardnessCalculator
from CodeResearch.ObjectComplexity.Hardness.Factory import LearnerEnum, AssesorEnum
from CodeResearch.ObjectComplexity.Hardness.LearnerBasedHardnessCalculator import LearnerBasedHardnessCalculator
from CodeResearch.ObjectComplexity.ObjectAssessment import BaseObjectAssesor
from CodeResearch.ObjectComplexity.ObjectAssessment.StandardAssesor import StandardAssesor
from CodeResearch.ObjectComplexity.ObjectAssessment.XGBoostAssesor import XGBoostAssesor


class HardnessFactory:

    @staticmethod
    def createHardnessCalculator(learner: LearnerEnum.LearnerEnum, assesor: AssesorEnum.AssesorEnum,  nAttempts, fraction) -> BaseHardnessCalculator:
        logger = SimpleLogger()
        l = HardnessFactory.createLearner(learner, logger)
        a = HardnessFactory.createAssesor(assesor)

        hc = LearnerBasedHardnessCalculator(l, a, nAttempts, fraction, logger)
        return hc

    @staticmethod
    def createLearner(learner: LearnerEnum.LearnerEnum, logger: BaseLogger.BaseLogger) -> BaseLearner:
        if learner == LearnerEnum.LearnerEnum.KS:
            return KSLearner(KSMetric(), logger)

        if learner == LearnerEnum.LearnerEnum.XGBoost:
            return XGBoostLearner()

        raise ValueError(f'Unsupported type of learner: {learner}')

    @staticmethod
    def createAssesor(assesor: AssesorEnum.AssesorEnum) -> BaseObjectAssesor.BaseObjectAssesor:
        if assesor == AssesorEnum.AssesorEnum.Standard:
            return StandardAssesor()

        if assesor == AssesorEnum.AssesorEnum.ShapXGBoost:
            return XGBoostAssesor()

        raise ValueError(f'Unsupported type of learner: {assesor}')