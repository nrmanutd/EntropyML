import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier


def calcConcreteModel(dataSet, nObjects, target):
    enc = LabelEncoder()
    target = enc.fit_transform(np.ravel(target))

    totalObjects = dataSet.shape[0]
    mask = np.zeros(totalObjects)
    mask[np.arange(nObjects)] = 1

    mask = np.random.permutation(mask)
    idx = np.where(mask > 0)[0]

    testIdx = np.where(mask == 0)[0]

    u, indexes = np.unique(target, return_index=True)
    s = set(np.concatenate((idx, indexes)))
    idx = list(s)

    subSet = dataSet[idx, :]
    subTarget = target[idx]

    train_idx = np.arange(nObjects, dtype=int)

    u, indexes = np.unique(subTarget, return_index=True)
    s = set(np.concatenate((train_idx, indexes)))
    train_idx = list(s)

    X_train = subSet[train_idx]
    Y_train = subTarget[train_idx]

    X_test = dataSet[testIdx]
    Y_test = target[testIdx]

    model = XGBClassifier().fit(X_train, Y_train)

    predict = model.predict(X_test)
    accuracy = accuracy_score(Y_test, predict)

    return accuracy

def calcModel(dataSet, nObjects, nAttempts, target):
    accuracy = np.zeros(nAttempts)

    for i in np.arange(nAttempts):
        accuracy[i] = calcConcreteModel(dataSet, nObjects, target)

    acc = np.mean(accuracy)
    sigma = np.std(accuracy)

    return {'accuracy': acc, 'modelSigma': sigma}