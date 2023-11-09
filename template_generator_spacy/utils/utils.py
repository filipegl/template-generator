from template_generator.instances import Prediction


def make_prediction(input, model):
    label, proba = model.predict(input)

    return Prediction(label[0], proba[0])