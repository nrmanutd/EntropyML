import numpy as np
import shap
from xgboost import XGBRegressor

from CodeResearch.ObjectComplexity.ObjectAssessment.StandardAssessor import StandardAssessor


class XGBoostAssessor(StandardAssessor):
    def estimateImportance(self, trainIdxes, testIdxes, testResponds, target):
        totalObjects = len(target)
        totalAttempts = len(trainIdxes)

        accuracy = np.array([testResponds[k][0] for k in range(len(testResponds))])
        usedObjects = np.zeros((totalAttempts, totalObjects))

        for i in range(totalAttempts):
            for j in range(len(trainIdxes[i])):
                usedObjects[i, trainIdxes[i][j]] = 1

        model = XGBRegressor().fit(usedObjects, accuracy)
        explainer = shap.TreeExplainer(model)

        shap_values = explainer.shap_values(usedObjects)

        #return np.abs(shap_values).mean(axis=0)
        return shap_values.mean(axis=0)

