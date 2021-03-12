import os
import re
import collections
from nltk.stem.snowball import SnowballStemmer
from natasha import Segmenter, Doc


def is_abbreviation(word):
    if (sum(1 for char in word if char.isupper()) / len(word)) >= 0.5:
        return True
    else:
        return False


class RuNormASReaderForSequenceTagging():
    def __init__(self):
        self.endings = []
        self.normalization_endings = []
        self.stemmer = SnowballStemmer('russian')

    def read(self, text_filename, annotation_filename, normalization_filename):
        text = open(text_filename, 'r', encoding='utf-8').read()
        annotation = open(annotation_filename, 'r', encoding='utf-8').read().strip().split('\n')
        normalization = open(normalization_filename, 'r', encoding='utf-8').read().strip().split('\n')

        segmenter = Segmenter()

        doc_text = Doc(text)
        doc_text.segment(segmenter)
        for sentence in doc_text.sents:
            print(sentence.tokens)

        entities = []
        entities_spans = []
        for line in annotation:
            spans = list(map(int, line.strip().split()))
            entry = ''
            while spans:
                start, stop = spans[0], spans[1]
                entry += text[start: stop] + " "
                spans = spans[2:]

            entry = entry.strip()

            doc = Doc(entry)
            doc.segment(segmenter)

            entities += doc.tokens

            spans = list(map(int, line.strip().split()))
            while spans:
                entities_spans += [[spans[0], spans[1]]]
                spans = spans[2:]

        normalized_entities = []
        for line in normalization:
            line.strip()
            entry = Doc(line)
            entry.segment(segmenter)
            normalized_entities += entry.tokens

        for span, entity, norm in zip(entities_spans, entities, normalized_entities):
            print(span, '->', entity, '->', norm)

        return None, None


    def get_stem(self, word):
        word_stem = self.stemmer.stem(word)
        if word[0].isupper():
            word_stem = word_stem.capitalize()

        return word_stem

    # функция для получения окончаний слова
    def find_ending(self, word, is_normalization=False):
        word_stem = self.stemmer.stem(word)
        if word[0].isupper():
            word_stem = word_stem.capitalize()
        ending = word.replace(word_stem, '')

        if word != ending:
            if is_normalization:
                self.normalization_endings.append(ending)
            else:
                self.endings.append(ending)
            return ending
        else:
            return ''


class RuNormASReaderForMachineTranslation():
    def __init__(self):
        pass


if __name__ == '__main__':
    text_filename = "data/train_new/named/texts_and_ann/196755.txt"
    annotation_filename = "data/train_new/named/texts_and_ann/196755.ann"
    normalization_filename = "data/train_new/named/norm/196755.norm"
    reader = RuNormASReaderForSequenceTagging()
    sentences, normalization = reader.read(text_filename, annotation_filename, normalization_filename)
