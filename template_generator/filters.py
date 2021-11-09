from abc import ABC, abstractclassmethod

from .instances import Prediction

class Filter(ABC):
    @abstractclassmethod
    def apply(cls):
        pass


class UnanimousClassificationFilter(Filter):

    ''' Filter inputs by unanimous classification.

    Given a list of inputs previously classified by a set of models, 
    filter all inputs unanimously classified in predictions.
    '''
    @classmethod
    def __filter_unanimous(cls, predictions):
        labels = [prediction.label for prediction in predictions]
        return all(labels[i] == labels[i + 1] for i in range(len(labels) - 1))

    @classmethod
    def apply(cls, inputs, predictions):
        print('Filtering instances classified unanimously...')
        filtered = list(filter(lambda x: cls.__filter_unanimous(x[1]), zip(inputs, predictions)))

        return [x[0] for x in filtered], [x[1] for x in filtered]


class HighClassificationScoreFilter(Filter):
    
    ''' Filter inputs by high classification.

    Given a list of inputs previously classified by a set of models,
    filter all the inputs with high score average in predictions.
    '''
    @classmethod
    def __filter_high_score(cls, predictions, min_score):
        pred_scores = sum([prediction.proba for prediction in predictions]) / len(predictions)
        return max(pred_scores) >= min_score

    @classmethod
    def apply(cls, inputs, predictions, min_score=0.9):
        print(f'Filtering instances by classification score greater than {min_score}')
        filtered = list(filter(lambda x: cls.__filter_high_score(x[1], min_score), zip(inputs, predictions)))

        return [x[0] for x in filtered], [x[1] for x in filtered]


class HighClassificationScoreWordFilter(Filter):
    
    ''' Filter inputs by high classification score for relevant words.

    Given a list of inputs previously ranked by a WordRanker, filter
    all inputs that have high score in its most relevant words in input 
    prediction.
    '''
    @classmethod
    def __filter_high_score_words(cls, input, model, n_words, min_score):
        
        tokens = input.sorted_tokens
        if len(tokens) == 0:
            return False
        
        for token in tokens[:n_words]:
            if not token.is_predicted:
                label, proba = model.predict(token.word)
                token.prediction = Prediction(label[0], proba[0])
            
            if max(token.prediction.proba) < min_score:
                return False

        return True

    @classmethod
    def apply(cls, inputs, model, n_words=2, min_score=0.95):
        print(f'Filtering instances by relevant words classification score greater than {min_score}')
        return list(filter(lambda x: cls.__filter_high_score_words(x, model, n_words, min_score), inputs))


class RelevantWordsFilter(Filter):
    
    ''' Filter inputs by relevant words.

    Given a list of inputs previously ranked by a WordRanker, filter
    all inputs that have only relevant_tags as most relevant
    words in input prediction.
    '''
    @classmethod
    def __filter_by_relevance(cls, input, relevant_tags, n_words=2):
        tokens = input.sorted_tokens
        if len(tokens) == 0:
            return False

        for token in tokens[:n_words]:
            if not token.tag in relevant_tags:
                return False

        return True

    @classmethod
    def apply(cls, inputs, relevant_tags, n_words=2):
        print(f'Filtering instances by relevant words...')
        return list(filter(lambda x: cls.__filter_by_relevance(x, relevant_tags, n_words), inputs))


class RankedWordsFilter(Filter):
    
    ''' Filter sentences by ranked words.

    Given a list of sentences and an entire instance, filter
    all sentences containing one of most relevant words in instance prediction.
    '''
    @classmethod
    def __filter_by_containing_ranked_words(cls, sentence, sent_counts):
        n_ranked_words = sent_counts[sentence.original_instance]
        wr_indexes = [token.index for token in sentence.original_instance.sorted_tokens[:n_ranked_words]]
        most_ranked_index = sentence.sorted_tokens[0].index

        return most_ranked_index in wr_indexes


    @classmethod
    def apply(cls, sentences):
        print(f'Filtering instances by contaning ranked words...')
        sent_counts = { }
        for sentence in sentences:
            key = sentence.original_instance
            sent_counts[key] = 0 if sent_counts.get(key) is None else  sent_counts[key] + 1

        return list(filter(lambda x: cls.__filter_by_containing_ranked_words(x, sent_counts), sentences))


class MinimmumInputSizeFilter(Filter):
    
    ''' Filter inputs by its size.

    Given a list of inputs, filter all the inputs that contains 
    a minimun ammount of words.
    Default is 5 words.
    '''
    @classmethod
    def __filter_by_minimum_length(cls, input, min_words):
        return input.length >= min_words


    @classmethod
    def apply(cls, inputs, min_words=5):
        print(f'Filtering instances by contaning a minimmum of words: {min_words}...')
        return list(filter(lambda x: cls.__filter_by_minimum_length(x, min_words), inputs))
