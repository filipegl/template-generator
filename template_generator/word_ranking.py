from os import error
from lime.lime_text import LimeTextExplainer
from abc import ABC, abstractmethod
from .instances import Prediction

class WordRank(ABC):
    
    @staticmethod
    @abstractmethod
    def rank(sentences, model):
        pass


class WordRankR1S(WordRank):

    @staticmethod
    def rank(inputs, model):
        print('Ranking words using Replace-1 Score...')

        for input in inputs:
            # Perform instance prediction
            if input.prediction == None:
                label, proba = model.predict(input.original_text)
                input.prediction = Prediction(label[0], proba[0])
            tokens_str = input.tokenized
            tokens = input.tokens

            label = input.prediction.label
            for i in range(len(tokens_str)):
                modified_input = ' '.join(tokens_str[:i] + tokens_str[i+1:])
                modified_input_proba = model.predict_proba(modified_input)[0]
                tokens[i].rank_score = modified_input_proba[label] - input.prediction.proba[label]

        return inputs


class WordRankLime(WordRank):

    @staticmethod
    def rank(inputs, model):
        print('Ranking words using Lime...')
        explainer = LimeTextExplainer()
        count = len(inputs)

        for input, i in zip(inputs, range(count)):
            print(f':: Explaining instance: {i+1} of {count}\r')

            exp = explainer.explain_instance(' '.join(input.tokenized), model.predict_proba, num_features=input.length)
            ranks = [*exp.as_map().values()][0]
            for rank in ranks:
                index, score = rank
                print(index, rank)
                input.tokens[index].rank_score = score

        return inputs
