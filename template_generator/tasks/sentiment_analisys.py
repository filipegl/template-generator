from ..template_generation import (
    GenericTemplateGeneratorApp1, 
    GenericTemplateGeneratorApp2, 
    GenericTemplateGeneratorApp3, 
    GenericTemplateGeneratorApp4,
    GenericTemplateGeneratorApp5
    )
from ..word_ranking import WordRankR1S
from template_generator.utils.utils import make_prediction
import re


class PosNegTemplateGeneratorApp1(GenericTemplateGeneratorApp1):

    def __init__(self, model, oracle_models):
        super().__init__(WordRankR1S, model, oracle_models)
        self.__lexicons = {'pos_verb': set(), 'neg_verb': set(), 'pos_adj': set(), 'neg_adj': set()}


    @property
    def lexicons(self):
        return {
            'pos_verb': list(self.__lexicons['pos_verb']), 
            'neg_verb': list(self.__lexicons['neg_verb']), 
            'pos_adj': list(self.__lexicons['pos_adj']), 
            'neg_adj': list(self.__lexicons['neg_adj'])
            }


    @property
    def relevant_tags(self):
        return ['VERB', 'ADJ']


    def replace_with_masks(self, sentences, n_words=2):
        for sent in sentences:
            # get target model
            model = self.model
            # get a list of tokens from sentence
            tokens = sent.sorted_tokens

            for token in tokens[:n_words]:
                if token.tag in self.relevant_tags:
                    if not token.is_predicted:
                        token.prediction = make_prediction(token.word, model)
                
                # Mount lex name based on prediction label and tag from token
                lex_name = f'pos_{token.tag.lower()}' if token.prediction.label == 1 else f'neg_{token.tag.lower()}'
                self.__lexicons[lex_name].add(token.word)
                token.word = '{' + lex_name + '}'

            sent.template_text = ' '.join([token.word for token in sent.tokens])
            sent.masked_text = re.sub('{\S+}', '{mask}', sent.template_text)

        return sentences

    def generate_templates(self, texts_input, n_masks=2):
        return super().generate_templates(texts_input, self.relevant_tags, n_masks=n_masks)


class PosNegTemplateGeneratorApp2(GenericTemplateGeneratorApp2):

    def __init__(self, model, oracle_models):
        super().__init__(WordRankR1S, model, oracle_models)
        self.__lexicons = {'pos_verb': set(), 'neg_verb': set(), 'pos_adj': set(), 'neg_adj': set()}


    @property
    def lexicons(self):
        return {
            'pos_verb': list(self.__lexicons['pos_verb']), 
            'neg_verb': list(self.__lexicons['neg_verb']), 
            'pos_adj': list(self.__lexicons['pos_adj']), 
            'neg_adj': list(self.__lexicons['neg_adj'])
            }


    @property
    def relevant_tags(self):
        return ['VERB', 'ADJ']


    def replace_with_masks(self, sentences, n_words=2):
        for sent in sentences:
            # get target model
            model = self.model
            # get a list of tokens from sentence
            tokens = sent.sorted_tokens

            for token in tokens[:n_words]:
                if token.tag in self.relevant_tags:
                    if not token.is_predicted:
                        token.prediction = make_prediction(token.word, model)
                
                # Mount lex name based on prediction label and tag from token
                lex_name = f'pos_{token.tag.lower()}' if token.prediction.label == 1 else f'neg_{token.tag.lower()}'
                self.__lexicons[lex_name].add(token.word)
                token.word = '{' + lex_name + '}'

            sent.template_text = ' '.join([token.word for token in sent.tokens])
            sent.masked_text = re.sub('{\S+}', '{mask}', sent.template_text)

        return sentences


    def generate_templates(self, texts_input, n_masks=2):
        return super().generate_templates(texts_input, self.relevant_tags, n_masks=n_masks)


class PosNegTemplateGeneratorApp3(GenericTemplateGeneratorApp3):

    def __init__(self, model, oracle_models):
        super().__init__(WordRankR1S, model, oracle_models)
        self.__lexicons = {'pos_verb': set(), 'neg_verb': set(), 'pos_adj': set(), 'neg_adj': set()}


    @property
    def lexicons(self):
        return {
            'pos_verb': list(self.__lexicons['pos_verb']), 
            'neg_verb': list(self.__lexicons['neg_verb']), 
            'pos_adj': list(self.__lexicons['pos_adj']), 
            'neg_adj': list(self.__lexicons['neg_adj'])
            }


    @property
    def relevant_tags(self):
        return ['VERB', 'ADJ']


    def replace_with_masks(self, sentences, n_words=2):
        for sent in sentences:
            # get target model
            model = self.model
            # get a list of tokens from sentence
            tokens = sent.sorted_tokens
            tokens = [token for token in tokens if token.tag in self.relevant_tags]

            for token in tokens[:n_words]:
                if not token.is_predicted:
                    token.prediction = make_prediction(token.word, model)
                
                # Mount lex name based on prediction label and tag from token
                lex_name = f'pos_{token.tag.lower()}' if token.prediction.label == 1 else f'neg_{token.tag.lower()}'
                self.__lexicons[lex_name].add(token.word)
                token.word = '{' + lex_name + '}'

            sent.template_text = ' '.join([token.word for token in sent.tokens])
            sent.masked_text = re.sub('{\S+}', '{mask}', sent.template_text)

        return sentences


    def generate_templates(self, texts_input, n_masks=2, range_words=2, min_classification_score=0.9):
        return super().generate_templates(texts_input, self.relevant_tags, n_masks, range_words, min_classification_score)


class PosNegTemplateGeneratorApp4(GenericTemplateGeneratorApp4):

    def __init__(self, model, oracle_models):
        super().__init__(WordRankR1S, model, oracle_models)
        self.__lexicons = {'pos_verb': set(), 'neg_verb': set(), 'pos_adj': set(), 'neg_adj': set()}


    @property
    def lexicons(self):
        return {
            'pos_verb': list(self.__lexicons['pos_verb']), 
            'neg_verb': list(self.__lexicons['neg_verb']), 
            'pos_adj': list(self.__lexicons['pos_adj']), 
            'neg_adj': list(self.__lexicons['neg_adj'])
            }


    @property
    def relevant_tags(self):
        return ['VERB', 'ADJ']


    def replace_with_masks(self, sentences, n_words=2):
        for sent in sentences:
            # get target model
            model = self.model
            # get a list of tokens from sentence
            tokens = sent.sorted_tokens

            for token in tokens[:n_words]:
                if token.tag in self.relevant_tags:
                    if not token.is_predicted:
                        token.prediction = make_prediction(token.word, model)
                
                # Mount lex name based on prediction label and tag from token
                lex_name = f'pos_{token.tag.lower()}' if token.prediction.label == 1 else f'neg_{token.tag.lower()}'
                self.__lexicons[lex_name].add(token.word)
                token.word = '{' + lex_name + '}'

            sent.template_text = ' '.join([token.word for token in sent.tokens])
            sent.masked_text = re.sub('{\S+}', '{mask}', sent.template_text)

        return sentences


    def generate_templates(self, texts_input, n_masks=2):
        return super().generate_templates(texts_input, self.relevant_tags, n_masks=n_masks)


class PosNegTemplateGeneratorApp5(GenericTemplateGeneratorApp5):

    def __init__(self, model, oracle_models):
        super().__init__(WordRankR1S, model, oracle_models)
        self.__lexicons = {'pos_verb': set(), 'neg_verb': set(), 'pos_adj': set(), 'neg_adj': set()}


    @property
    def lexicons(self):
        return {
            'pos_verb': list(self.__lexicons['pos_verb']), 
            'neg_verb': list(self.__lexicons['neg_verb']), 
            'pos_adj': list(self.__lexicons['pos_adj']), 
            'neg_adj': list(self.__lexicons['neg_adj'])
            }


    @property
    def relevant_tags(self):
        return ['VERB', 'ADJ']


    def replace_with_masks(self, sentences, n_words=2):
        for sent in sentences:
            # get target model
            model = self.model
            # get a list of tokens from sentence
            tokens = sent.sorted_tokens
            tokens = [token for token in tokens if token.tag in self.relevant_tags]

            for token in tokens[:n_words]:
                if not token.is_predicted:
                    token.prediction = make_prediction(token.word, model)
                
                # Mount lex name based on prediction label and tag from token
                lex_name = f'pos_{token.tag.lower()}' if token.prediction.label == 1 else f'neg_{token.tag.lower()}'
                self.__lexicons[lex_name].add(token.word)
                token.word = '{' + lex_name + '}'

            sent.template_text = ' '.join([token.word for token in sent.tokens])
            sent.masked_text = re.sub('{\S+}', '{mask}', sent.template_text)

        return sentences


    def generate_templates(self, texts_input, n_masks=2, range_words=2, min_classification_score=0.9):
        return super().generate_templates(texts_input, self.relevant_tags, n_masks, range_words, min_classification_score)

