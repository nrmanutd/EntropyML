import numpy as np
import tensorflow as tf
from keras import Sequential, Input
from keras.src.layers import Dense
from keras.src.losses import losses
from keras.src.utils import to_categorical
from tensorflow.keras import optimizers

from CodeResearch.LearningFramework.Learners.baseLearner import BaseLearner

class NNEpochLearner(BaseLearner):

    def train(self, x, y, probs):
        nFeatures = x.shape[1]
        nClasses = len(np.unique(y))
        self.optimizer = optimizers.Adam(learning_rate=1e-3)
        self.loss_fn = losses.CategoricalCrossentropy()

        model = self.define_model(nFeatures, nClasses)
        model = self.update(model, x, y)
        return model

    def update(self, model, x, y):
        nClasses = len(np.unique(y))

        with tf.GradientTape() as tape:
            predictions = model(x, training=True)
            y_onehot = tf.one_hot(y, depth=nClasses)
            loss_value = self.loss_fn(y_onehot, predictions)

        # считаем и применяем градиенты
        grads = tape.gradient(loss_value, model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, model.trainable_variables))

        return model

    def test(self, model, x, y):
        nClasses = len(np.unique(y))
        y_test = to_categorical(y, nClasses)
        _, acc = model.evaluate(x, y_test, verbose=0)
        return acc

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
        return self.test(model, xt, yt)