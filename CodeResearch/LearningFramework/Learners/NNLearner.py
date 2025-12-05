import numpy as np
from keras import Sequential, Input
from keras.src.layers import Dense
from keras.src.utils import to_categorical

from CodeResearch.LearningFramework.Learners.baseLearner import BaseLearner


class NNLearner(BaseLearner):
    def test(self, model, x, y):

        nClasses = len(np.unique(y))
        y_test = to_categorical(y, nClasses)
        _, acc = model.evaluate(x, y_test, verbose=0)
        return acc

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

        dense = 512 if nFeatures > 20 else 16

        model.add(Dense(dense, activation='relu', kernel_initializer='he_uniform'))
        model.add(Dense(dense, activation='relu', kernel_initializer='he_uniform'))
        model.add(Dense(nClasses, activation='softmax'))
        # compile model

        model.compile(optimizer="adam", loss='categorical_crossentropy', metrics=['accuracy'])
        return model

    def trainAndTest(self, x, y, probs, xt, yt):
        model = self.train(x, y, probs)
        accuracy = self.test(model, xt, yt)
        return accuracy