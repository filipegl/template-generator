import collections
import spacy

from numpy.core.fromnumeric import argmax

class Prediction:

    def __init__(self, label, proba):
        self.label = label
        self.proba = proba


class Token:
    
    def __init__(self, word, index, tag):
        self.word = word
        self.__index = index
        self.__tag = tag
        self.prediction = None
        self.rank_score = None

    @property
    def index(self):
        return self.__index
    
    @property
    def tag(self):
        return self.__tag
    
    @property
    def is_ranked(self):
        return self.rank_score != None

    @property
    def is_predicted(self):
        return self.prediction != None

    def __str__(self):
        return  '{'+f'word: {self.word}, index: {self.index},  tag: {self.tag}, rank_score: {self.rank_score}'+'}'


class Instance:
    def __init__(self, text):
        self.__original_text = text
        self.nlp = spacy.load("en_core_web_trf")
        self._tokenized:list[str] = []
        self._tokens: list[Token] = []

        for token in self.nlp(text=text):
            self._tokenized.append(token.text)
            self._tokens.append(Token(word=token.text, index=token.i, tag=token.pos_))
        
        self.predictions = []
        # self.prediction = None

    @property
    def length(self):
        return len(self.tokens)

    @property
    def original_text(self):
        return self.__original_text

    @property
    def tokenized(self):
        return self._tokenized

    @property
    def tokens(self):
        return self._tokens

    @property
    def sorted_tokens(self):
        if self.is_word_ranked:
            return sorted(self.tokens, key=lambda token: abs(token.rank_score), reverse=True)

        return self.tokens

    @property
    def prediction(self):
        if len(self.predictions) == 0:
            return None
        
        # Mount a Prediction with most prediced label and average proba from predictions
        lbl_dict = collections.Counter([prediction.label for prediction in self.predictions])
        label = max(lbl_dict, key=lbl_dict.get)
        proba = sum([prediction.proba for prediction in self.predictions]) / len(self.predictions)
        
        # If a tie occurs, use the greater proba average to define the label
        if min(lbl_dict, key=lbl_dict.get) == label:
            label = argmax(proba)
        
        return Prediction(label, proba)
        
    @property
    def is_word_ranked(self):
        for token in self.tokens:
            if not token.is_ranked:
                return False
        return True

    @property
    def is_predicted(self):
        return self.prediction != None

    def split_to_sentences(self):

        ''' Generate sentences based on original text
        '''
        sentences = []
        
        start = 0; end = 0
        for sent in self.nlp(text=self.original_text).sents:
            sent = str(sent)
            chars = len(''.join([token.text for token in self.nlp(text=sent)]))
            while chars > 0:
                chars -= len(self.tokens[end].word)
                end += 1

            sentences.append(Sentence(text=sent, tokens=self.tokens[start: end], tokenized=self.tokenized[start: end], original_instance=self))
            start = end

        return sentences

    def __str__(self):
        return  '\n'. join([
            f'original_text: {self.original_text}'
            ]) 


class Sentence(Instance):
    def __init__(self, text, tokens=None, tokenized=None, original_instance=None):
        super().__init__(text)

        self.masked_text = None
        self.template_text = None
        
        if not tokens is None:
            self._tokens = tokens
        if not tokenized is None:
            self._tokenized = tokenized
        self.__original_instance = original_instance

    @property
    def original_instance(self):
        return self.__original_instance

    def to_array(self):
        return [self.prediction.label, self.original_text, self.masked_text, self.template_text]

    def __str__(self):
        return  '\n'. join([
            f'label: {self.prediction.label if self.prediction is not None else "Not predicted..."}',
            f'original_text: {self.original_text}',
            f'masked_text: {self.masked_text}',
            f'template_text: {self.template_text}',
            ]) 

