from CodeResearch.ObjectComplexity.ObjectAssessment.StandardAssessor import StandardAssessor


class EasinessOnlyAssessor(StandardAssessor):
    def estimateImportance(self, trainIdxes, testIdxes, testResponds, target):
        return None