from CodeResearch.ObjectComplexity.Diversity.BaseObjectDiversifier import BaseObjectDiversifier


class NNBasedObjectDiversifier(BaseObjectDiversifier):
    def __init__(self, priority):
        self.priority = priority

    def calculateObjectDiversity(self, dataSet, target):
        pass