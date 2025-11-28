from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from CodeResearch.LearningFramework.Learners.baseLearner import BaseLearner


class LogitLearner(BaseLearner):
    def test(self, model, x, y):
        predict = model.predict(x)
        accuracy = accuracy_score(y, predict)

        return accuracy

    def train(self, x, y):
        clf = LogisticRegression(
            max_iter=1000,  # увеличиваем число итераций, чтобы точно сошлось
            solver="lbfgs"  # стандартный солвер (подходит для L2-регуляризации)
        )

        clf.fit(x, y)
        return clf

