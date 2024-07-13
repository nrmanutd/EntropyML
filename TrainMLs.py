from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import numpy as np

def TrainLogit(dataSet, target):
    seed = 7
    test_size = 0.25
    X_train, X_test, y_train, y_test = train_test_split(dataSet, target, test_size=test_size, random_state=seed)
    logit = LogisticRegression(random_state=0, max_iter=1000).fit(X_train, np.ravel(y_train))

    return logit.predict(dataSet)

def TrainXGBoost(dataSet, target):
    seed = 7
    test_size = 0.25
    X_train, X_test, y_train, y_test = train_test_split(dataSet, target, test_size=test_size, random_state=seed)
    model = XGBClassifier().fit(X_train, np.ravel(y_train))

    return model.predict(dataSet)