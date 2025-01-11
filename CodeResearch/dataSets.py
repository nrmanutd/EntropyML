from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical


def loadMnist():
    num_train = 60000  # there are 60000 training examples in MNIST
    num_test = 10000  # there are 10000 test examples in MNIST

    height, width, depth = 28, 28, 1  # MNIST images are 28x28 and greyscale
    num_classes = 10  # there are 10 classes (1 per digit)

    # load dataset
    (trainX, trainY), (testX, testY) = mnist.load_data()
    # reshape dataset to have a single channel

    trainX = trainX.reshape(num_train, height * width)  # Flatten data to 1D
    testX = testX.reshape(num_test, height * width)  # Flatten data to 1D
    trainX = trainX.astype('float32')
    testX = testX.astype('float32')
    trainX /= 255  # Normalise data to [0, 1] range
    testX /= 255  # Normalise data to [0, 1] range

    trainY = to_categorical(trainY, num_classes)  # One-hot encode the labels
    testY = to_categorical(testY, num_classes)  # One-hot encode the labels

    return trainX, trainY, testX, testY