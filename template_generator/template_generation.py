from template_generator.instances import Instance
from template_generator.filters import *
from template_generator.oracle_model import OracleModel

from abc import ABC, abstractmethod
import pandas as pd
import random

class TemplateGenerator(ABC):

    def __init__(self, word_ranker, model, oracle_models):
        self.__word_ranker = word_ranker
        self.__model = model
        self.__oracle_model = OracleModel(oracle_models)

    @abstractmethod
    def replace_with_masks(self, sentences, n_words=2):
        pass

    @abstractmethod
    def generate_templates(self, texts_input, relevant_tags=[], n_masks = 2, ranked_words_count=2, min_classification_score=0.9):
        pass

    @property
    def model(self):
        return self.__model

    @property
    def oracle_model(self):
        return self.__oracle_model
    
    @property
    def word_ranker(self):
        return self.__word_ranker
    
    @property
    def sentences(self):
        return self.__sentences

    @sentences.setter
    def sentences(self, value):
        self.__sentences = value
        
    @property
    def original_texts(self):
        if self.sentences is not None:
            return [sent.original_text for sent in self.sentences]
        else:
            raise Exception('Sentences were not generated. Use generate_templates first.')
        
    @property
    def masked_texts(self):
        if self.sentences is not None:
            return [sent.masked_text for sent in self.sentences]
        else:
            raise Exception('Sentences were not generated. Use generate_templates first.')
        
    @property
    def template_texts(self):
        if self.sentences is not None:
            return [sent.template_text for sent in self.sentences]
        else:
            raise Exception('Sentences were not generated. Use generate_templates first.')

    def to_dataframe(self):
        if self.sentences is None:
            raise Exception('Sentences were not generated. Use generate_templates first.')

        data = [sent.to_array() for sent in self.sentences]

        return pd.DataFrame(data, columns=['label', 'original_text', 'masked_text', 'template_text'])


# Approach 1
class GenericTemplateGeneratorApp1(TemplateGenerator):

    def generate_templates(self, texts_input, relevant_tags, n_masks=2, ranked_words_count=2):
        if isinstance(texts_input, str):
            texts_input = [texts_input]
        
        instances = [Instance(text) for text in texts_input]

        # 1. Ranking words from entire instance by its importance when predicted by target model
        instances: list[Instance] = self.word_ranker.rank(instances, self.model)
        instances_to_write = list()
        for inst in instances:
            sorted_tokens_tuples = ' '.join([f"({token.word}, {token.rank_score:.7f})" for token in inst.sorted_tokens])
            tokens_tuples = ' '.join([f"({token.word}, {token.rank_score:.7f})" for token in inst.tokens])
                                            
            instances_to_write.append(inst.original_text + "\n" + tokens_tuples + "\n" + sorted_tokens_tuples + "\n")

        write_sentences("instances.txt", instances_to_write)

        # 2. Break instances into sentences
        print('Converting texts to sentences...')
        sentences1: list[Instance] = []
        for instance in instances:
            sentences1.extend(instance.split_to_sentences())
        print(f':: {len(sentences1)} sentences were generated.')
        
        # sentences1_to_write = list()
        # for sent in sentences1:
        #     sentences1_to_write.append(sent.original_text + "\n" + ' '.join([f"({token.word}, {token.rank_score})" for token in sent.tokens]))


        write_sentences("sentencesP2.txt", [x.original_text for x in sentences1])

        # 3. Filter sentences that contains some of K most ranked words (K = amount of sentences generated by instance)
        sentences2 = ContainingRankedWordsFilter.apply(sentences1)
        print(f':: {len(sentences2)} sentences remaining.')
        sentences2: list[Instance] = self.replace_with_masks(sentences2, n_masks)
        write_sentencesp3 = list()
        for sent in sentences2:
            tags = ' '.join([f"({w.word}, {w.tag}) " for w in sent.tokens])
            write_sentencesp3.append(sent.original_text + "\n" + tags + "\n")

        write_sentences("sentencesP3.txt", write_sentencesp3)
        
        # 4. Filtering sentences having only adjectives or verbs with higher ranked words
        sentences3 = RelevantWordsFilter.apply(sentences2, relevant_tags, n_masks, ranked_words_count)
        print(f':: {len(sentences3)} sentences remaining.')
        write_sentences("sentencesP4.txt", [x.template_text for x in sentences3])
        # 5. Predicting sentences with oracle models
        for sent, preds in zip(sentences3, self.oracle_model.predict_all(sentences3)):
            sent.predictions = preds
        print(f':: Sentence predictions done.')

        # 6. Replacing the n most relevant words with masks
        sentences3 = self.replace_with_masks(sentences3, n_masks)
        self.sentences = sentences3

        return sentences3

def write_sentences(filepath, sents):
    with open(filepath, "wt") as f:
        for s in sents:
            f.write(s + "\n")

# Approach 2
class GenericTemplateGeneratorApp2(TemplateGenerator):

    def generate_templates(self, texts_input, relevant_tags, n_masks=2, ranked_words_count=2):
        if isinstance(texts_input, str):
            texts_input = [texts_input]

        instances = [Instance(text) for text in texts_input]

        # 1. Ranking words from entire instance by its importance when predicted by target model
        instances = self.word_ranker.rank(instances, self.model)

        # 2. Break instances into sentences
        print('Converting texts to sentences...')
        sentences = []
        for instance in instances:
            sentences.extend(instance.split_to_sentences())
        print(f':: {len(sentences)} sentences were generated.')

        # 3. Filter sentences that contains some of K most ranked words (K = amount of sentences generated by instance)
        sentences = ContainingRankedWordsFilter.apply(sentences)
        print(f':: {len(sentences)} sentences remaining.')

        # 4. Ranking words by its importance when predicted by target model
        sentences = self.word_ranker.rank(sentences, self.model)
        print(f':: Word ranking done.')

        # 5. Filtering sentences having only adjectives or verbs with higher ranked words
        sentences = RelevantWordsFilter.apply(sentences, relevant_tags, n_masks, ranked_words_count)
        print(f':: {len(sentences)} sentences remaining.')

        # 6. Predicting sentences with oracle models
        for sent, preds in zip(sentences, self.oracle_model.predict_all(sentences)):
            sent.predictions = preds
        print(f':: Sentence predictions done.')

        # 7. Replacing the n most relevant words with masks
        sentences = self.replace_with_masks(sentences, n_masks)
        self.sentences = sentences

        return sentences


# Approach 3
class GenericTemplateGeneratorApp3(TemplateGenerator):

    def generate_templates(self, texts_input, relevant_tags, n_masks=2, ranked_words_count=2, min_classification_score=0.9):
        instances = [Instance(text) for text in texts_input]

        # 1. Break instances into sentences
        print('Converting texts to sentences...')
        sentences = []
        for instance in instances:
            sentences.extend(instance.split_to_sentences())
        print(f':: {len(sentences)} sentences were generated.')

        # 2. Predicting sentences with oracle models
        for sent, preds in zip(sentences, self.oracle_model.predict_all(sentences)):
            sent.predictions = preds
        print(f':: Sentence predictions done.')
        
        # 3. Filtering sentences classified unanimously
        sentences = UnanimousClassificationFilter.apply(sentences)
        print(f':: {len(sentences)} sentences remaining.')

        # 4. Filtering sentences with high average score in classification step
        sentences = HighClassificationScoreFilter.apply(sentences, min_classification_score)
        print(f':: {len(sentences)} sentences remaining.')

        # 5. Ranking words by its importance when predicted by target model
        sentences = self.word_ranker.rank(sentences, self.model)
        print(f':: Word ranking done.')

        # 6. Filtering sentences having only relevant words as higher ranked words
        sentences = RelevantWordsFilter.apply(sentences, relevant_tags, n_masks, ranked_words_count)
        print(f':: {len(sentences)} sentences remaining.')

        # 7. Filtering sentences having relevant words with high score
        sentences = HighClassificationScoreWordFilter.apply(sentences, self.model, relevant_tags, n_masks, ranked_words_count, 
                                                            min_classification_score)
        print(f':: {len(sentences)} sentences remaining.')

        # 8. Replacing the n most relevant words with masks
        sentences = self.replace_with_masks(sentences, n_masks)
        self.sentences = sentences

        return sentences


# Approach 4
class GenericTemplateGeneratorApp4(TemplateGenerator):

    def generate_templates(self, texts_input, relevant_tags, n_masks=2, ranked_words_count=2, min_classification_score=0.9):
        instances = [Instance(text) for text in texts_input]

        # 1. Predicting instances with oracle models
        for instance, preds in zip(instances, self.oracle_model.predict_all(instances)):
            instance.predictions = preds
        print(f':: Instance predictions done.')

        # 2. Filter instances unanimously
        instances = UnanimousClassificationFilter.apply(instances)
        print(f':: {len(instances)} instances remaining.')

        # 3. Break instances into sentences
        print('Converting texts to sentences...')
        sentences = []
        for instance in instances:
            sentences.extend(instance.split_to_sentences())
        print(f':: {len(sentences)} sentences were generated.')

        # 4. Predicting sentences with oracle models
        for sent, preds in zip(sentences, self.oracle_model.predict_all(sentences)):
            sent.predictions = preds
        print(f':: Sentence predictions done.')

        # 5. Filtering sentences classified unanimously
        sentences = UnanimousClassificationFilter.apply(sentences)
        print(f':: {len(sentences)} sentences remaining.')

        # 6. Filtering sentences with high average score in classification step
        sentences = HighClassificationScoreFilter.apply(sentences, min_classification_score)
        print(f':: {len(sentences)} sentences remaining.')

        # 7. Ranking words by its importance when predicted by target model
        sentences = self.word_ranker.rank(sentences, self.model)
        print(f':: Word ranking done.')

        # 8. Filtering sentences having only relevant words as higher ranked words
        sentences = RelevantWordsFilter.apply(sentences, relevant_tags, n_masks, ranked_words_count)
        print(f':: {len(sentences)} sentences remaining.')

        # 9. Filtering sentences having relevant words with high score
        sentences = HighClassificationScoreWordFilter.apply(sentences, self.model, relevant_tags, n_masks, ranked_words_count, 
                                                            min_classification_score)
        print(f':: {len(sentences)} sentences remaining.')

        # 10. Replacing the n most relevant words with masks
        sentences = self.replace_with_masks(sentences, n_masks)
        self.sentences = sentences

        return sentences


# Approach 5
class GenericTemplateGeneratorApp5(TemplateGenerator):

    def generate_templates(self, texts_input, relevant_tags, n_masks = 2, ranked_words_count=2, min_classification_score=0.9):
        instances = [Instance(text) for text in texts_input]

        # 1. Break instances into sentences
        print('Converting texts to sentences...')
        sentences = []
        for instance in instances:
            sentences.extend(instance.split_to_sentences())
        print(f':: {len(sentences)} sentences were generated.')

        # 2. Ranking words by its importance when predicted by target model
        sentences = self.word_ranker.rank(sentences, self.model)
        print(f':: Word ranking done.')

        # 3. Filter by sentence size
        sentences = MinimmumInputSizeFilter.apply(sentences, 5)
        print(f':: {len(sentences)} sentences remaining.')

        # 4. Filtering sentences having only relevant words as higher ranked words
        sentences = RelevantWordsFilter.apply(sentences, relevant_tags, n_masks, ranked_words_count)
        print(f':: {len(sentences)} sentences remaining.')

        # 5. Filtering sentences having relevant words with high score
        sentences = HighClassificationScoreWordFilter.apply(sentences, self.model, relevant_tags, n_masks, ranked_words_count, 
                                                            min_classification_score)
        print(f':: {len(sentences)} sentences remaining.')
        
        # 6. Predicting sentences with oracle models
        for sent, preds in zip(sentences, self.oracle_model.predict_all(sentences)):
            sent.predictions = preds
        print(f':: Sentence predictions done.')

        # 7. Replacing the n most relevant words with masks
        sentences = self.replace_with_masks(sentences, n_masks)
        self.sentences = sentences

        return sentences


# Random Approach
class GenericTemplateGeneratorRandom(TemplateGenerator):
    
    def generate_templates(self, texts_input, n_masks=2, k_templates=10):
        instances = [Instance(text) for text in texts_input]

        # 1. Break instances into sentences
        print('Converting texts to sentences...')
        sentences = []
        for instance in instances:
            sentences.extend(instance.split_to_sentences())
        print(f':: {len(sentences)} sentences were generated.')
        
        # 2. Sampling n sentences randomly
        sentences = random.sample(sentences, k_templates)

        # 3. Ranking words by its importance when predicted by target model
        sentences = self.word_ranker.rank(sentences, self.model)
        print(f':: Word ranking done.')

        # 4. Predicting sentences with oracle models
        for sent, preds in zip(sentences, self.oracle_model.predict_all(sentences)):
            sent.predictions = preds
        print(f':: Sentence predictions done.')

        # 5. Replacing the n most relevant words with masks
        sentences = self.replace_with_masks(sentences, n_masks)
        self.sentences = sentences

        return sentences
        