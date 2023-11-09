from abc import ABC, abstractclassmethod
from template_generator.utils.utils import make_prediction

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
    def __filter_unanimous(cls, input):
        labels = [prediction.label for prediction in input.predictions]
        return all(labels[i] == labels[i + 1] for i in range(len(labels) - 1))

    @classmethod
    def apply(cls, inputs):
        print('Filtering instances classified unanimously...')
        return list(filter(lambda x: cls.__filter_unanimous(x), inputs))


class HighClassificationScoreFilter(Filter):
    
    ''' Filter inputs by high classification.

    Given a list of inputs previously classified by a set of models,
    filter all the inputs with high score average in predictions.
    '''
    @classmethod
    def __filter_high_score(cls, input, min_score):
        return max(input.prediction.proba) >= min_score

    @classmethod
    def apply(cls, inputs, min_score=0.9):
        print(f'Filtering instances by classification score greater than {min_score}')
        return list(filter(lambda x: cls.__filter_high_score(x, min_score), inputs))


class HighClassificationScoreWordFilter(Filter):
    
    ''' Filter inputs by high classification score for relevant words.

    Given a list of inputs previously ranked by a WordRanker, filter
    all inputs that have high score in its most relevant words in input 
    prediction.
    '''
    @classmethod
    def __filter_high_score_words(cls, input, model, relevant_tags, n_words, ranked_words_count, min_score):
        tokens = input.sorted_tokens[:ranked_words_count]
        
        count = 0
        for token in tokens:
            if not token.is_predicted:
                token.prediction = make_prediction(token.word, model)

            if token.tag in relevant_tags and max(token.prediction.proba) >= min_score:
                count += 1
        
        return count >= n_words

    @classmethod
    def apply(cls, inputs, model, relevant_tags, n_words=2, ranked_words_count=2, min_score=0.95):
        print(f'Filtering instances by relevant words classification score greater than {min_score}')
        return list(filter(lambda x: cls.__filter_high_score_words(x, model, relevant_tags, n_words, ranked_words_count, min_score), inputs))


class RelevantWordsFilter(Filter):
    
    ''' Filter inputs by relevant words.

    Given a list of inputs previously ranked by a WordRanker, filter
    all inputs that have only relevant_tags as most relevant
    words in input prediction.
    '''
    @classmethod
    def __filter_by_relevance(cls, input, relevant_tags, n_words=2, ranked_words_count=2):
        tokens = input.sorted_tokens[:ranked_words_count]
        if len(tokens) == 0:
            return False

        return len([token for token in tokens if token.tag in relevant_tags]) >= n_words


    @classmethod
    def apply(cls, inputs, relevant_tags, n_words=2, ranked_words_count=2):
        print(f'Filtering instances by relevant words...')
        return list(filter(lambda x: cls.__filter_by_relevance(x, relevant_tags, n_words, ranked_words_count), inputs))


class ContainingRankedWordsFilter(Filter):
    
    ''' Filter sentences by ranked words.

    Given a list of sentences and an entire instance, filter
    all sentences containing one of most relevant words in instance prediction.
    '''
    @classmethod
    def __filter_by_containing_ranked_words(cls, sentence, sent_counts):
        ranked_words_count = sent_counts[sentence.original_instance]
        wr_indexes = [token.index for token in sentence.original_instance.sorted_tokens[:ranked_words_count]]
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
