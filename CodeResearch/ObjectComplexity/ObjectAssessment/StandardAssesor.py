import numpy as np
from CodeResearch.ObjectComplexity.ObjectAssessment.BaseObjectAssesor import BaseObjectAssesor


class StandardAssesor(BaseObjectAssesor):
    def estimate(self, trainIdxes, testIdxes, testResponds, target):
        easyness = self.estimateEasyness(trainIdxes, testIdxes, testResponds, target)
        importance = self.estimateImportance(trainIdxes, testIdxes, testResponds, target)

        importance = self.convertToECDF(importance)
        return easyness, importance

    def estimateEasyness(self, trainIdxes, testIdxes, testResponds, target):
        totalObjects = len(target)
        totalAttempts = len(testIdxes)

        objectsCorrect = np.zeros(totalObjects)
        objectsUsed = np.zeros(totalObjects)

        for i in range(totalAttempts):
            curTestIdx = testIdxes[i]

            objectsUsed[curTestIdx] += 1
            curResponds = testResponds[i][1]  # todo: check how to process testResponds value

            for j in range(len(curTestIdx)):
                curObjectRespond = curResponds[j]
                curObjectIdxOriginal = curTestIdx[j]
                trueObjectRespond = target[curObjectIdxOriginal]

                if curObjectRespond == trueObjectRespond:
                    objectsCorrect[curObjectIdxOriginal] += 1

        easyness = np.array([objectsCorrect[i] / objectsUsed[i] for i in range(len(target))])

        return easyness

    def estimateImportance(self, trainIdxes, testIdxes, testResponds, target):
        totalObjects = len(target)
        totalAttempts = len(trainIdxes)

        shapValues = np.zeros(totalObjects)
        accuracy = np.array([testResponds[k][0] for k in range(len(testResponds))])
        usedObjects = []

        for i in range(totalAttempts):
            curObjects = np.zeros(totalObjects)
            curObjects[trainIdxes[i]] = 1
            usedObjects.append(curObjects)

        for i in range(totalObjects):
            withObjectIdx = []
            noObjectIdx = []

            for j in range(totalAttempts):
                if usedObjects[j][i] == 1:
                    withObjectIdx.append(j)
                else:
                    noObjectIdx.append(j)

            shapValues[i] = np.mean(accuracy[withObjectIdx]) - np.mean(accuracy[noObjectIdx])

        shapValues = self.convertToECDF(shapValues)
        return shapValues

    def convertToECDF(self, importance):
        idx = np.argsort(importance)

        result = np.zeros(len(importance))
        totalElements = len(idx)

        for i in range(len(idx)):
            originalIdx = idx[i]
            result[originalIdx] = i / totalElements

        return result





