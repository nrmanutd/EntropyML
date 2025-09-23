from CodeResearch.ObjectComplexity.Factory.BaseComplexityCalculatorFactory import BaseComplexityCalculatorFactory
from CodeResearch.ObjectComplexity.shapValueComplexityCalculator import ShapValueComplexityCalculator

class ShapValuesComplexityCalculatorFactory(BaseComplexityCalculatorFactory):
    def createComplexityCalculator(self, ds, target, idx):
        return ShapValueComplexityCalculator(ds, target, idx)