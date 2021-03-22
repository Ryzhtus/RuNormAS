import os
import tqdm
from nltk.stem.snowball import SnowballStemmer
from natasha import Segmenter, Doc
from collections import Counter

def is_abbreviation(word):
    if (sum(1 for char in word if char.isupper()) / len(word)) >= 0.5:
        return True
    else:
        return False

def clean_entity(entry):
    # в документе 994383 был странный токен с *, поэтому на такой случай удаляем такие штуки,
    # иначе ошибка с размерностью
    entry = entry.replace('*', '')
    # документ 990538 Sputnik'ом разибвается на три токена, поэтому 'ом убираем
    # документ 927141 Ростова-на Дону -> Ростова-на-Дону
    # документ 1041141 Кара- оол -> Кара-оол
    if '’ом' in entry:
        entry = entry.replace('’ом', '')

    if '«' in entry:
        entry = entry.replace('«', '')

    if '»' in entry:
        entry = entry.replace('»', '')

    if '(' in entry:
        entry = entry.replace('(', '')

    if ')' in entry:
        entry = entry.replace(')', '')

    if entry == 'Ростова-на Дону':
        entry = 'Ростова-на-Дону'

    if entry == 'Кара- оол':
        entry = 'Кара-оол'

    return entry

def clean_endings(endings_counter):
    endings = []
    res = False
    for key in endings_counter.keys():
        if key == '':
            endings.append(key)
        else:
            if len(key) <= 4 and key[0].isupper() != True:
                for item in ['0','1','2','3','4','5','6','7','8','9', '-', '!', '№', "'"]:
                    if item in key:
                        res = True
                if res == False:
                    endings.append(key)
    return endings

class RuNormASReaderForSequenceTagging():
    def __init__(self):
        self.endings = []
        self.normalization_endings = []
        self.stemmer = SnowballStemmer('russian')
        self.symbols = ['«', '»', '(', ')', "'", '’']
        self.norm_endings_counter = Counter()

    def read(self, text_filename, annotation_filename, normalization_filename):
        text = open(text_filename, 'r', encoding='utf-8').read()
        annotation = open(annotation_filename, 'r', encoding='utf-8').read().strip().split('\n')
        normalization = open(normalization_filename, 'r', encoding='utf-8').read().strip().split('\n')

        segmenter = Segmenter()

        doc_text = Doc(text)
        doc_text.segment(segmenter)

        entities = []

        for line in annotation:
            spans = list(map(int, line.strip().split()))
            entry = ''

            while spans:
                start, stop = spans[0], spans[1]
                entry = text[start: stop] + ' '
                spans = spans[2:]

            entry = entry.strip()
            entry = clean_entity(entry)

            doc = Doc(entry)
            doc.segment(segmenter)

            entities += [doc.tokens]

        normalized_entities = []
        for line in normalization:
            line.strip()
            entry = clean_entity(line)
            entry = Doc(entry)
            entry.segment(segmenter)
            normalized_entities += [entry.tokens]

        entities_stem = []
        norm_endings = []

        for entity, norm in zip(entities, normalized_entities):
            norms_and_endings = []
            stems_and_endings = []
            for token, token_norm in zip(entity, norm):
                if is_abbreviation(token.text):
                    stem = token.text
                    ending = ''
                    norm_ending = ''
                else:
                    stem = self.get_stem(token.text)
                    ending = token.text.replace(stem, '')
                    norm_ending = token_norm.text.replace(stem, '')

                self.endings.append(ending)
                self.normalization_endings.append(norm_ending)
                self.norm_endings_counter[norm_ending] += 1
                stems_and_endings.append(stem)
                stems_and_endings.append(ending)
                norms_and_endings.append('<NO>')
                self.norm_endings_counter['<NO>'] += 1
                norms_and_endings.append(norm_ending)
            entities_stem.append(stems_and_endings)
            norm_endings.append(norms_and_endings)

        return entities_stem, norm_endings

    def get_stem(self, word):
        if is_abbreviation(word):
            return word
        else:
            word_stem = self.stemmer.stem(word)
            if word[0].isupper():
                word_stem = word_stem.capitalize()

            return word_stem

def collect_sentences_for_sequence_tagging():
    reader = RuNormASReaderForSequenceTagging()

    entities_train = []
    endings_train = []

    entities_eval = []
    endings_eval = []

    filenames = []
    for _, _, files in os.walk("data/train_new/named/texts_and_ann"):
        for filename in sorted(files):
            filenames.append(filename.split('.')[0])

    filenames = sorted(set(filenames), reverse=True)

    train = filenames[:2000]
    eval = filenames[2000:]

    for filename in tqdm.tqdm(train, total=len(train)):
        text_filename = "../data/train_new/named/texts_and_ann/" + filename + ".txt"
        annotation_filename = "../data/train_new/named/texts_and_ann/" + filename + ".ann"
        normalization_filename = "../data/train_new/named/norm/" + filename + ".norm"
        document_entities, document_normalization = reader.read(text_filename, annotation_filename, normalization_filename)

        entities_train += document_entities
        endings_train += document_normalization

    for filename in tqdm.tqdm(eval, total=len(eval)):
        text_filename = "../data/train_new/named/texts_and_ann/" + filename + ".txt"
        annotation_filename = "../data/train_new/named/texts_and_ann/" + filename + ".ann"
        normalization_filename = "../data/train_new/named/norm/" + filename + ".norm"
        document_entities, document_normalization = reader.read(text_filename, annotation_filename, normalization_filename)

        entities_eval += document_entities
        endings_eval += document_normalization

    reader.normalization_endings.append('')
    reader.normalization_endings.append('<NO>')
    reader.normalization_endings = set(reader.normalization_endings)

    return entities_train, endings_train, entities_eval, endings_eval, reader.normalization_endings, reader.norm_endings_counter