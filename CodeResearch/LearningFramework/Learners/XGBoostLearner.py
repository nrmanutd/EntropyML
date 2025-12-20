from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier

from CodeResearch.LearningFramework.Learners.baseLearner import BaseLearner


class XGBoostLearner(BaseLearner):

    def train(self, x, y, probs):
        model = XGBClassifier().fit(x, y)
        return model

    def test(self, model, x, y):
        predict = model.predict(x)
        accuracy = accuracy_score(y, predict)

        return accuracy, predict

    def trainAndTest(self, x, y, probs, xt, yt):
        model = self.train(x, y, probs)
        return self.test(model, xt, yt)

    def update(self, model, x, y):
        raise NotImplementedError('update for XGBoost shouldnt be called')