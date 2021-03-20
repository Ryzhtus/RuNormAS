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
    """
    if '«' in entry:
        entry = entry.replace('«', '')

    #if '»' in entry:
        entry = entry.replace('»', '')

    if '(' in entry:
        entry = entry.replace('(', '')

    if ')' in entry:
        entry = entry.replace(')', '')"""

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
        entity_id = 0
        number_of_entities_in_annotation = 0

        for line in annotation:
            if line != '\n' or line != '':
                number_of_entities_in_annotation += 1

            spans = list(map(int, line.strip().split()))
            entry = ''

            while spans:
                start, stop = spans[0], spans[1]
                entry = text[start: stop]

                entry = entry.strip()
                entry = clean_entity(entry)

                doc = Doc(entry)
                doc.segment(segmenter)

                for idx in range(len(doc.tokens)):
                    token = doc.tokens[idx]
                    token.start = start
                    token.stop = stop
                    doc.tokens[idx] = token

                for token in doc.tokens:
                    entities += [[token, entity_id]]

                spans = spans[2:]

            entity_id += 1

        print(len(entities), entities)

        normalized_entities = []
        entity_id = 0
        for line in normalization:
            line.strip()
            entry = Doc(line)
            entry.segment(segmenter)
            for token in entry.tokens:
                normalized_entities += [[token, entity_id]]

            entity_id += 1

        print(len(normalized_entities), normalized_entities)

        for idx in range(len(entities)):
            print(entities[idx], normalized_entities[idx])

        return doc_text, entities, normalized_entities, number_of_entities_in_annotation

    def parse_entities(self, text, normalized_entities, entities):
        sentences = []
        sentences_endings = []
        sentences_entities_ids = []
        for sentence in text.sents:
            sentence_tokens = []
            sentence_endings = []
            sentence_entities_id = []
            for token in sentence.tokens:
                if self.find_entity_by_spans(token, entities):
                    entry, entry_id = self.find_entity_by_spans_with_entry_id(token, entities, normalized_entities)
                    if token.text in self.symbols:
                        sentence_tokens.append(token.text)
                        sentence_endings.append('')
                        sentence_entities_id.append(entry_id)
                    elif token.text == entry:
                        sentence_tokens.append(token.text)
                        sentence_endings.append('')
                        sentence_entities_id.append(entry_id)
                    else:
                        stem = self.get_stem(token.text)
                        ending = self.find_ending(entry, stem, is_normalization=True)
                        sentence_tokens.append(stem)
                        sentence_endings.append(ending)
                        sentence_entities_id.append(entry_id)
                else:
                    sentence_endings.append('<NO>')
                    sentence_tokens.append(token.text)
                    sentence_entities_id.append(-1)
            sentences.append(sentence_tokens)
            sentences_endings.append(sentence_endings)
            sentences_entities_ids.append(sentence_entities_id)

        return sentences, sentences_endings, sentences_entities_ids

    def find_entity_by_spans(self, token, entities):
        for idx in range(len(entities)):
            if (token.start >= entities[idx][0].start and token.stop <= entities[idx][0].stop) and entities[idx][0].text == token.text:
                return True

    def find_entity_by_spans_with_entry_id(self, token, entities, normalized_entities):
        for idx in range(len(entities)):
            if (token.start >= entities[idx][0].start and token.stop <= entities[idx][0].stop) and entities[idx][0].text == token.text:
                if token.text in self.symbols:
                    return token.text, entities[idx][1]
                else:
                    for idx_norm in range(len(normalized_entities)):
                        if self.match_entity_and_norm(token.text, normalized_entities[idx_norm][0].text):
                            return normalized_entities[idx_norm][0].text, normalized_entities[idx_norm][1]

    def match_entity_and_norm(self, entity, norm):
        entity = entity.lower()
        norm = norm.lower()
        min_length = min(entity, norm, key=lambda x: len(x))

        count = 0
        for idx in range(len(min_length)):
            if entity[idx] == norm[idx]:
                count += 1

        if (count / len(min_length)) >= 0.3:
            return True
        else:
            return False

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

    all_sentences_train = []
    all_sentences_endings_train = []

    all_sentences_eval = []
    all_sentences_endings_eval = []

    filenames = []
    for _, _, files in os.walk("../data/train_new/named/texts_and_ann"):
        for filename in sorted(files):
            filenames.append(filename.split('.')[0])

    filenames = sorted(set(filenames), reverse=True)

    train = filenames[:2000]
    eval = filenames[2000:]


    for filename in tqdm.tqdm(train, total=len(train)):
        print(filename)
        text_filename = "data/train_new/named/texts_and_ann/" + filename + ".txt"
        annotation_filename = "data/train_new/named/texts_and_ann/" + filename + ".ann"
        normalization_filename = "data/train_new/named/norm/" + filename + ".norm"
        document, document_entities, document_normalization, counter = reader.read(text_filename, annotation_filename, normalization_filename)
        sentences, sentences_endings, sentences_entities_ids = reader.parse_entities(document, document_normalization, document_entities)
        """
        ids = []
        for sentences_ids in sentences_entities_ids:
            for idx in sentences_ids:
                if idx != -1:
                    ids.append(idx)
        ids = set(ids)

        if len(ids) != counter:
            print(filename)
            print(ids)
            print(counter)
            print('---------')"""

        all_sentences_train += sentences
        all_sentences_endings_train += sentences_endings

    for filename in tqdm.tqdm(eval, total=len(eval)):
        text_filename = "data/train_new/named/texts_and_ann/" + filename + ".txt"
        annotation_filename = "data/train_new/named/texts_and_ann/" + filename + ".ann"
        normalization_filename = "data/train_new/named/norm/" + filename + ".norm"
        document, document_entities, document_normalization, counter = reader.read(text_filename, annotation_filename, normalization_filename)
        sentences, sentences_endings, sentences_entities_ids = reader.parse_entities(document, document_normalization, document_entities)
        """
        ids = []
        for sentences_ids in sentences_entities_ids:
            for idx in sentences_ids:
                if idx != -1:
                    ids.append(idx)
        ids = set(ids)

        if len(ids) != counter:
            print(filename)
            print(ids)
            print(counter)
            print('---------')"""

        all_sentences_eval += sentences
        all_sentences_endings_eval += sentences_endings

    reader.normalization_endings.append('')
    reader.normalization_endings.append('<NO>')
    reader.normalization_endings = set(reader.normalization_endings)

    return all_sentences_train, all_sentences_endings_train, all_sentences_eval, all_sentences_endings_eval, reader.normalization_endings


if __name__ == '__main__':
    #all_sentences_train, all_sentences_endings_train, all_sentences_eval, all_sentences_endings_eval, all_endings = collect_sentences_for_sequence_tagging()


    #text_filename = "data/train_new/named/texts_and_ann/992640.txt"
    #annotation_filename = "data/train_new/named/texts_and_ann/992640.ann"
    #normalization_filename = "data/train_new/named/norm/992640.norm"
    text_filename = "../data/train_new/named/texts_and_ann/986080.txt"
    annotation_filename = "../data/train_new/named/texts_and_ann/986080.ann"
    normalization_filename = "../data/train_new/named/norm/986080.norm"
    reader = RuNormASReaderForSequenceTagging()
    document, document_entities, document_normalization, counter = reader.read(text_filename, annotation_filename, normalization_filename)
    sentences, sentences_endings, sentences_entities_ids = reader.parse_entities(document, document_normalization, document_entities)

    for sentence, sentence_endings, entities_id in zip(sentences, sentences_endings, sentences_entities_ids):
        for token, ending, entity_id in zip(sentence, sentence_endings, entities_id):
            if entity_id != -1:
                print(token, '->', ending, '->', entity_id)



