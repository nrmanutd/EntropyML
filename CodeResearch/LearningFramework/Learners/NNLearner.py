import numpy as np
from keras import Sequential, Input
from keras.src.layers import Dense
from keras.src.utils import to_categorical

from CodeResearch.LearningFramework.Learners.baseLearner import BaseLearner


class NNLearner(BaseLearner):
    def __init__(self, dense=16, nLayers=2):
        self.nLayers = nLayers
        self.dense = dense

    def test(self, model, x, y):

        nClasses = len(np.unique(y))
        y_test = to_categorical(y, nClasses)
        _, acc = model.evaluate(x, y_test, verbose=0)

        y_pred_proba = model.predict(x, verbose=0)
        y_pred = np.argmax(y_pred_proba, axis=1)

        return acc, y_pred

    def train(self, x, y, probs):
        nFeatures = x.shape[1]
        nClasses = len(np.unique(y))

        model = self.define_model(nFeatures, nClasses)
        # fit model
        y_train = to_categorical(y, nClasses)
        model.fit(x, y_train, epochs=10, batch_size=128, validation_split=0.1, verbose=0)#todo: check if validation split is necessary here
        return model

    def define_model(self, nFeatures, nClasses):
        model = Sequential()
        model.add(Input(shape=(nFeatures,)))

        for k in range(self.nLayers):
            model.add(Dense(self.dense, activation='relu', kernel_initializer='he_uniform'))
            model.add(Dense(self.dense, activation='relu', kernel_initializer='he_uniform'))

        model.add(Dense(nClasses, activation='softmax'))
        # compile model

        model.compile(optimizer="adam", loss='categorical_crossentropy', metrics=['accuracy'])
        return model

    def trainAndTest(self, x, y, probs, xt, yt):
        model = self.train(x, y, probs)
        accuracy, prediction = self.test(model, xt, yt)
        return accuracy, prediction