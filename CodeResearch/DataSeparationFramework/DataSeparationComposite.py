from CodeResearch.DataSeparationFramework.dataSeparationBaseCalculator import BaseDataSeparationCalculator
from typing import List

class DataSeparationComposite(BaseDataSeparationCalculator):
    def __init__(self, components: List[BaseDataSeparationCalculator]):
        self.components = components

    def calculateDataSeparability(self, objects, iClass, jClass):
        for component in self.components:
            component.calculateDataSeparability(objects, iClass, jClass)

    def serializeResults(self):
        for component in self.components:
            component.serializeResults()