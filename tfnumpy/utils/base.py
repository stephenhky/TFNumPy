

class NotImplementedError(Exception):
    def __init__(self):
        self.message = 'Not Implemented!'


class SupervisedClassifier:
    def train(self, trainX, trainY):
        raise NotImplementedError()

    def predict(self, testX):
        raise NotImplementedError()
