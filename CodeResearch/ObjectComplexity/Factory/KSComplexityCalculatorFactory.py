from CodeResearch.ObjectComplexity.Factory.BaseComplexityCalculatorFactory import BaseComplexityCalculatorFactory
from CodeResearch.ObjectComplexity.complexityCalculator import KSComplexityCalculator


class KSComplexityCalculatorFactory(BaseComplexityCalculatorFactory):
    def createComplexityCalculator(self, ds, target, idx):
        return KSComplexityCalculator(ds, target, idx)
