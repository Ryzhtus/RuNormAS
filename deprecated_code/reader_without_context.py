import os
import tqdm
from nltk.stem.snowball import SnowballStemmer
from natasha import Segmenter, Doc


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



class RuNormASReaderForSequenceTagging():
    def __init__(self):
        self.endings = []
        self.normalization_endings = []
        self.stemmer = SnowballStemmer('russian')
        self.symbols = ['«', '»', '(', ')', "'", '’']

    def read(self, text_filename, annotation_filename, normalization_filename):
        text = open(text_filename, 'r', encoding='utf-8').read()
        annotation = open(annotation_filename, 'r', encoding='utf-8').read().strip().split('\n')
        normalization = open(normalization_filename, 'r', encoding='utf-8').read().strip().split('\n')

        segmenter = Segmenter()

        doc_text = Doc(text)
        doc_text.segment(segmenter)

        entities = []
        number_of_entities_in_annotation = 0

        for line in annotation:
            if line != '\n' or line != '':
                number_of_entities_in_annotation += 1

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

        # print(len(entities), entities)

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
            endings = []
            stems = []
            for token, token_norm in zip(entity, norm):
                stem = self.get_stem(token.text)
                ending = self.find_ending(token_norm.text, stem, is_normalization=True)
                stems.append(stem)
                endings.append(ending)
            entities_stem.append(stems)
            norm_endings.append(endings)


        #print(len(normalized_entities), normalized_entities)
        """
        for idx in range(len(entities)):
            print(entities[idx], normalized_entities[idx])"""

        return entities_stem, norm_endings

    def get_stem(self, word):
        if is_abbreviation(word):
            return word
        else:
            word_stem = self.stemmer.stem(word)
            if word[0].isupper():
                word_stem = word_stem.capitalize()

            return word_stem

    # функция для получения окончаний слова
    def find_ending(self, norm, word, is_normalization=False):
        if is_abbreviation(norm):
            return ''
        else:
            word_stem = self.stemmer.stem(norm)
            if norm[0].isupper():
                word_stem = word_stem.capitalize()
            if norm == word:
                return ''
            else:
                ending = norm.replace(word_stem, '')

                if norm != ending:
                    if is_normalization:
                        self.normalization_endings.append(ending)
                    else:
                        self.endings.append(ending)
                    return ending
                else:
                    return ''

def collect_sentences_for_sequence_tagging():
    reader = RuNormASReaderForSequenceTagging()

    entities_train = []
    endings_train = []

    entities_eval = []
    endings_eval = []

    filenames = []
    for _, _, files in os.walk("../data/train_new/named/texts_and_ann"):
        for filename in sorted(files):
            filenames.append(filename.split('.')[0])

    filenames = sorted(set(filenames), reverse=True)

    train = filenames[:2000]
    eval = filenames[2000:]


    for filename in tqdm.tqdm(train, total=len(train)):
        text_filename = "data/train_new/named/texts_and_ann/" + filename + ".txt"
        annotation_filename = "data/train_new/named/texts_and_ann/" + filename + ".ann"
        normalization_filename = "data/train_new/named/norm/" + filename + ".norm"
        document_entities, document_normalization = reader.read(text_filename, annotation_filename, normalization_filename)

        if len(document_entities) != len(document_normalization):
            print(filename)
            print('---------')

        entities_train += document_entities
        endings_train += document_normalization

    for filename in tqdm.tqdm(eval, total=len(eval)):
        text_filename = "data/train_new/named/texts_and_ann/" + filename + ".txt"
        annotation_filename = "data/train_new/named/texts_and_ann/" + filename + ".ann"
        normalization_filename = "data/train_new/named/norm/" + filename + ".norm"
        document_entities, document_normalization = reader.read(text_filename, annotation_filename,
                                                                normalization_filename)

        if len(document_entities) != len(document_normalization):
            print(filename)
            print('---------')

        entities_eval += document_entities
        endings_eval += document_normalization

    reader.normalization_endings.append('')
    reader.normalization_endings.append('<NO>')
    reader.normalization_endings = set(reader.normalization_endings)

    return entities_train, endings_train, entities_eval, endings_eval,  reader.normalization_endings

if __name__ == '__main__':
    entities_train, endings_train, entities_eval, endings_eval, normalization_endings = collect_sentences_for_sequence_tagging()

    counter = 0
    for entity, norm in zip(entities_train, endings_train):
        print(entity, norm)
        counter += 1
        if counter == 100:
            break



    """
    text_filename = "data/train_new/named/texts_and_ann/970431.txt"
    annotation_filename = "data/train_new/named/texts_and_ann/970431.ann"
    normalization_filename = "data/train_new/named/norm/970431.norm"
    reader = RuNormASReaderForSequenceTagging()
    document_entities, document_normalization, = reader.read(text_filename, annotation_filename, normalization_filename)

    for entity, norm in zip(document_entities, document_normalization):
        print(entity, norm)"""