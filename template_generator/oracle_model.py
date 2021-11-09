import numpy as np
from .instances import Prediction

class OracleModel:
    def __init__(self, models):
        self.__models = models

    # Predict an instance with a set of oracle models
    def predict(self, text):
        models = self.__models
        predictions = []

        for model in models:
            label, proba = model.predict(text)
            predictions.append(Prediction(label[0], proba[0]))

        return predictions

    # Predict a list of inputs with the set of models
    def predict_all(self, inputs):
        print('Predicting inputs...')
        return [self.predict(input.original_text) for input in inputs]
