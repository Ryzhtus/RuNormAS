import os
import re
import collections
from nltk.stem.snowball import SnowballStemmer

class RuNormASReader():
    def __init__(self):
        self.endings = []
        self.endings_counter = collections.Counter()
        self.normalization_endings = []
        self.normalization_endings_counter = collections.Counter()
        self.normalized_entities_counter = collections.Counter()
        self.stemmer = SnowballStemmer('russian')

    def read(self, text_filename, annotation_filename, normalization_filename):

        document = open(text_filename, 'r').read()
        print(document)
        annotations = open(annotation_filename, 'r').read().split('\n')

        # парсим интервалы из .ann файла
        entities_intervals = []
        for interval in annotations:
            if interval != '':
                entity = interval.split(' ')
                entity[0], entity[1] = int(entity[0]), int(entity[1])
                entities_intervals.append(entity)

        # собираем нормализованные сущности из .norm файла
        normalized_entities = open(normalization_filename, 'r').read().split('\n')
        normalized_entities.pop()

        for normalized_entity in normalized_entities:
            self.normalized_entities_counter[normalized_entity] += 1

        # собираем сущности и .txt файла по интервалам из .ann файла
        entities = []
        for entity_interval in entities_intervals:
            entities.append(document[entity_interval[0] : entity_interval[1]])

        # сопоставляем сущности ее нормализацию
        entity2norm = {}
        for idx in range(len(entities)):
            entity2norm[entities[idx]] = normalized_entities[idx]

        # сортируем интервалы сущностей по порядку вхождения в текст
        entities_intervals = sorted(entities_intervals, key=lambda x: x[0])

        # пересобираем сущности по отсортированным интервалам
        entities = []
        for entity_interval in entities_intervals:
            entities.append(document[entity_interval[0]: entity_interval[1]])

        # создаем список нормализованных сущностей по порядку на основе прошлого шага и entity2norm
        normalized_entities = []
        for entity in entities:
            normalized_entities.append(entity2norm[entity])

        # копируем сущности и их нормализацию
        entities_copy = entities.copy()
        normalized_entities_copy = normalized_entities.copy()

        # заменяем сущности в тексте на нормализованные, получая текст с нормализацией
        while entities_copy:
            entity = entities_copy.pop(0)
            norm = normalized_entities_copy.pop(0)
            document = document.replace(entity, norm, 1)

        # собираем оригинальные предложения
        sentences = open(text_filename, 'r').read().split('\u2028')
        sentences = [re.split('[!?.]', sentence) for sentence in sentences]

        cleaned_sentences = []
        for sentence in sentences:
            for subsentence in sentence:
                cleaned_sentence = re.split('[\s,.]', subsentence)
                if cleaned_sentence != ['']:
                    cleaned_sentence = [word for word in cleaned_sentence if word != '']
                    cleaned_sentences.append(cleaned_sentence)

        # для каждой сущности находим окончание ее нормализации
        entities_copy = entities.copy()
        sentences_endings = []

        # Я чуть с ума не сошел, пока придумал как это сделать!!!
        for sentence in cleaned_sentences:
            sentence_endings = ['<NO>' for i in range(len(sentence))]
            entity_count = 1

            while entity_count != 0:
                entity_count = 0
                for window_start in range(0, len(sentence) - len(entities_copy[0].split(' '))):
                    entity_from_list = entities_copy[0].split(' ')
                    #print(entity_from_list)
                    for idx in range(len(entity_from_list)):
                        entity_from_list[idx] = re.sub(r'\W+', '', entity_from_list[idx])

                    words_from_sentence = sentence[window_start: window_start + len(entities_copy[0].split(' '))]
                    for idx in range(len(words_from_sentence)):
                        words_from_sentence[idx] = re.sub(r'\W+', '', words_from_sentence[idx])

                    if words_from_sentence == entity_from_list:
                        entity_count += 1
                        entity = entities_copy.pop(0)
                        norm_entity = entity2norm[entity].split(' ')
                        entity = entity.split(' ')

                        for entity_idx, word_idx in zip(range(len(entity)), range(window_start, window_start + len(entity))):
                            if not is_abbreviation(sentence[word_idx]):
                                stem = self.get_stem(sentence[word_idx])
                                ending = self.find_ending(norm_entity[entity_idx], normalization=True)

                                sentence[word_idx] = stem
                                sentence_endings[word_idx] = ending
                            else:
                                sentence_endings[word_idx] = ''

            sentences_endings.append(sentence_endings)
        
        for sentence, endings in zip(cleaned_sentences, sentences_endings):
            print(sentence)
            print(endings)
            print('-' * 75)

        return sentences, sentences_endings

    def get_stem(self, word):
        word_stem = self.stemmer.stem(word)
        if word[0].isupper():
            word_stem = word_stem.capitalize()

        return word_stem

    # функция для получения окончаний слова
    def find_ending(self, word, normalization=False):
        word_stem = self.stemmer.stem(word)
        if word[0].isupper():
            word_stem = word_stem.capitalize()
        ending = word.replace(word_stem, '')
        # print(word, word_stem, ending)

        if word != ending:
            if normalization:
                self.normalization_endings.append(ending)
                self.normalization_endings_counter[ending] += 1
            else:
                self.endings.append(ending)
                self.endings_counter[ending] += 1
            return ending
        else:
            return ''


def is_abbreviation(word):
    if (sum(1 for char in word if char.isupper()) / len(word)) >= 0.5:
        return True
    else:
        return False

def collect_sentences():
    reader = RuNormASReader()

    sentences = []
    sentences_endings = []

    filenames = []
    for _, _, files in os.walk("data/train/named/texts_and_ann"):
        for filename in sorted(files):
            filenames.append(filename.split('.')[0])

    filenames = sorted(set(filenames), reverse=True)

    for filename in filenames:
        text_filename = "data/train/named/texts_and_ann/" + filename + ".txt"
        annotation_filename = "data/train/named/texts_and_ann/" + filename + ".ann"
        normalization_filename = "data/train/named/norm/" + filename + ".norm"
        text, text_endings = reader.read(text_filename, annotation_filename, normalization_filename)

        sentences += text
        sentences_endings += text_endings

    reader.endings = set(reader.endings)
    reader.normalization_endings = set(reader.normalization_endings)
    print(reader.endings_counter)
    print(reader.normalization_endings_counter)
    print(reader.normalized_entities_counter)

    return sentences, sentences_endings


if __name__ == '__main__':
    sentences, sentences_endings = normalized_sentences_tags = collect_sentences()
    #text_filename = "data/train/named/texts_and_ann/1009448.txt"
    #annotation_filename = "data/train/named/texts_and_ann/1009448.ann"
    #normalization_filename = "data/train/named/norm/1009448.norm"
    #reader = RuNormASReader()
    #sentences, normalization = reader.read(text_filename, annotation_filename, normalization_filename)

    # sentences, normalized_sentences = collect_sentences()
    for i in range(2):
        print(sentences[i])
        print(sentences_endings[i])
        print('-' * 75)

    #print(len(sentences), len(normalized_sentences))

