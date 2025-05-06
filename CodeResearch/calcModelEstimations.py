import random

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.utils import to_categorical
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report


def calcXGBoost(X_train, Y_train, X_test, Y_test):
    model = XGBClassifier().fit(X_train, Y_train)

    predict = model.predict(X_test)
    accuracy = accuracy_score(Y_test, predict)

    return accuracy


def calcLinRegression(X_train, Y_train, X_test, Y_test):
    """
    Train a linear (logistic) regression classifier and evaluate on test data.

    Parameters:
        X_train (array-like): Training features.
        Y_train (array-like): Training labels.
        X_test  (array-like): Test features.
        Y_test  (array-like): Test labels.

    Returns:
        dict: {
            'model': trained LogisticRegression model,
            'predictions': array of predicted labels on X_test,
            'accuracy': accuracy score on test set,
            'report': classification report as dict
        }
    """
    # Initialize and train logistic regression (linear classifier)
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, Y_train)

    # Predict and evaluate
    preds = model.predict(X_test)
    acc = accuracy_score(Y_test, preds)
    report = classification_report(Y_test, preds, output_dict=True)

    return acc


def calcSVM(X_train, Y_train, X_test, Y_test):
    """
    Train a linear SVM classifier and evaluate on test data.

    Parameters:
        X_train (array-like): Training features.
        Y_train (array-like): Training labels.
        X_test  (array-like): Test features.
        Y_test  (array-like): Test labels.

    Returns:
        dict: {
            'model': trained SVC model,
            'predictions': array of predicted labels on X_test,
            'accuracy': accuracy score on test set,
            'report': classification report as dict
        }
    """
    # Initialize and train linear SVM
    model = SVC(kernel='linear', probability=False)
    model.fit(X_train, Y_train)

    # Predict and evaluate
    preds = model.predict(X_test)
    acc = accuracy_score(Y_test, preds)
    report = classification_report(Y_test, preds, output_dict=True)

    return acc

def calcNN(X_train, Y_train, X_test, Y_test):

    #return 0

    nFeatures = X_train.shape[1]
    nClasses = len(np.unique(Y_train))

    Y_train = to_categorical(Y_train, nClasses)  # One-hot encode the labels
    Y_test = to_categorical(Y_test, nClasses)  # One-hot encode the labels

    model = define_model(nFeatures, nClasses)
    # fit model
    model.fit(X_train, Y_train, epochs=10, batch_size=32, verbose=0)
    # save model
    #model.save('digit_model.h5')
    _, acc = model.evaluate(X_test, Y_test, verbose=0)

    return acc

def define_model(nFeatures, nClasses):
    model = Sequential()
    model.add(Input(shape=(nFeatures,)))

    dense = 512 if nFeatures > 20 else 16

    model.add(Dense(dense, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(dense, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(nClasses, activation='softmax'))
    # compile model
    opt = SGD(learning_rate=0.01, momentum=0.9)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def fitAndPredict(X_train, Y_train, X_test, Y_test):
    nModels = 2

    accuracy = np.zeros(nModels)
    accuracy[0] = calcXGBoost(X_train, Y_train, X_test, Y_test)
    accuracy[1] = calcNN(X_train, Y_train, X_test, Y_test)

    return accuracy

def calcConcreteModel(dataSet, nObjects, target):
    totalObjects = dataSet.shape[0]
    mask = np.zeros(totalObjects)
    mask[0:nObjects] = 1

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

    accuracy = fitAndPredict(X_train, Y_train, X_test, Y_test)

    return accuracy

def calcModel(dataSet, nObjects, nAttempts, target):
    enc = LabelEncoder()
    target = enc.fit_transform(np.ravel(target))

    nModels = 2
    accuracy = np.zeros((nModels, nAttempts))

    for i in np.arange(nAttempts):
        accuracy[:, i] = calcConcreteModel(dataSet, nObjects, target)

    acc = np.zeros(nModels)
    sigma = np.zeros(nModels)

    for iModel in np.arange(nModels):
        acc[iModel] = np.mean(accuracy[iModel, :])
        sigma[iModel] = np.std(accuracy[iModel, :], ddof=1)

    return {'accuracy': acc, 'modelSigma': sigma}