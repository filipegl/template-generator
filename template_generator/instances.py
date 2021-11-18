import nltk

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


class Instance:
    def __init__(self, text):
        self.__original_text = text     
        self._tokenized = nltk.tokenize.word_tokenize(text)
        self._tokens = self.__generate_tokens()
        
        self.prediction = None

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
    def is_word_ranked(self):
        for token in self.tokens:
            if not token.is_ranked:
                return False
        return True

    @property
    def is_predicted(self):
        return self.prediction != None

    def __generate_tokens(self):
        tokens_str = self.tokenized
        tags = nltk.pos_tag(tokens_str, tagset='universal')

        return [Token(tok, i, tag[1]) for tok, i, tag in zip(tokens_str, range(len(tokens_str)), tags)]

    def split_to_sentences(self):

        ''' Generate sentences based on original text
        '''
        sentences_str = nltk.tokenize.sent_tokenize(self.original_text)
        sentences = []
        tokens = self.tokens
        
        start = 0; end = 0
        for sent in sentences_str:
            chars = len(''.join(nltk.tokenize.word_tokenize(sent)))
            while chars > 0:
                chars -= len(tokens[end].word)
                end += 1

            sentences.append(Sentence(sent, self.tokens[start: end], self.tokenized[start: end], self))
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

