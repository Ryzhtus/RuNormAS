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

    def read(self, text_filename, annotation_filename, normalization_filename):
        text = open(text_filename, 'r', encoding='utf-8').read()
        annotation = open(annotation_filename, 'r', encoding='utf-8').read().strip().split('\n')
        normalization = open(normalization_filename, 'r', encoding='utf-8').read().strip().split('\n')

        segmenter = Segmenter()

        doc_text = Doc(text)
        doc_text.segment(segmenter)

        entities = []
        entities_spans = []
        entity_id = 0
        entities_ids = []
        entity2id = []
        for line in annotation:
            spans = list(map(int, line.strip().split()))
            entry = ''

            while spans:
                start, stop = spans[0], spans[1]
                entry += text[start: stop] + " "
                spans = spans[2:]

            entry = entry.strip()
            entry = clean_entity(entry)

            doc = Doc(entry)
            doc.segment(segmenter)

            entities += doc.tokens
            entry_ids = [entity_id for i in range(len(doc.tokens))]
            entities_ids += entry_ids

            for token in range(len(doc.tokens)):
                entity2id.append([doc.tokens[token], entry_ids[token]])
                #entity2id[doc.tokens[token]] = entry_ids[token]

            entity_id += 1

            spans = list(map(int, line.strip().split()))
            if len(spans) == 2 and len(doc.tokens) > 1:
                start = spans[0]
                stop = start
                for idx in range(len(doc.tokens)):
                    start += doc.tokens[idx].start
                    stop = start + (doc.tokens[idx].stop - doc.tokens[idx].start)  # вот это я математику тут придумал
                    entities_spans += [[start, stop]]
            else:
                while spans:
                    entities_spans += [[spans[0], spans[1]]]
                    spans = spans[2:]

        normalized_entities = []
        for line in normalization:
            line.strip()
            entry = Doc(line)
            entry.segment(segmenter)
            normalized_entities += entry.tokens

        print(len(entities_spans), len(entities), len(normalized_entities), len(entities_ids), len(entity2id))
        print(entities)
        print(normalized_entities)
        print(entities_spans)
        print(entities_ids)
        print(entity2id)

        return doc_text, entities, normalized_entities, entities_spans, entity2id

    def parse_entities(self, doc_text, normalized_entities, entities_spans, entity2id):
        sentences = []
        sentences_endings = []
        sentences_entities_ids = []
        for sentence in doc_text.sents:
            sentence_tokens = []
            sentence_endings = []
            sentence_entities_id = []
            for token in sentence.tokens:
                if self.find_entity(token, entity2id):
                    #print(token.text)
                    id = self.find_entity(token, entity2id)
                    if token.text == normalized_entities[id].text:
                        sentence_tokens.append(token.text)
                        sentence_endings.append('')
                        sentence_entities_id.append(id)
                    else:
                        stem = self.get_stem(token.text)
                        ending = self.find_ending(normalized_entities[id].text, stem, is_normalization=True)
                        sentence_tokens.append(stem)
                        sentence_endings.append(ending)
                        sentence_entities_id.append(id)
                else:
                    sentence_endings.append('<NO>')
                    sentence_tokens.append(token.text)
                    sentence_entities_id.append(-1)
            sentences.append(sentence_tokens)
            sentences_endings.append(sentence_endings)
            sentences_entities_ids.append(sentence_entities_id)

        return sentences, sentences_endings, sentences_entities_ids

    def find_entity_by_spans(self, token, entities_spans):
            for idx in range(len(entities_spans)):
                if entities_spans[idx][0] == token.start and entities_spans[idx][1] == token.stop:
                    return idx

    def find_entity(self, token, entity2id):
        for idx in range(len(entity2id)):
            #print(token.text)
            #print(entity2id[idx])
            if token.text == entity2id[idx][0].text:
                return idx

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
        text_filename = "data/train_new/named/texts_and_ann/" + filename + ".txt"
        annotation_filename = "data/train_new/named/texts_and_ann/" + filename + ".ann"
        normalization_filename = "data/train_new/named/norm/" + filename + ".norm"
        document, document_entities, document_normalization, document_entities_spans, entities_ids = reader.read(text_filename, annotation_filename, normalization_filename)
        sentences, sentences_endings = reader.parse_entities(document, document_entities, document_normalization,
                                                             document_entities_spans)
        all_sentences_train += sentences
        all_sentences_endings_train += sentences_endings

    for filename in tqdm.tqdm(eval, total=len(eval)):
        text_filename = "data/train_new/named/texts_and_ann/" + filename + ".txt"
        annotation_filename = "data/train_new/named/texts_and_ann/" + filename + ".ann"
        normalization_filename = "data/train_new/named/norm/" + filename + ".norm"
        document, document_entities, document_normalization, document_entities_spans, entities_ids = reader.read(text_filename, annotation_filename, normalization_filename)
        sentences, sentences_endings = reader.parse_entities(document, document_entities, document_normalization,
                                                             document_entities_spans)
        all_sentences_eval += sentences
        all_sentences_endings_eval += sentences_endings

    reader.normalization_endings.append('')
    reader.normalization_endings.append('<NO>')
    reader.normalization_endings = set(reader.normalization_endings)

    return all_sentences_train, all_sentences_endings_train, all_sentences_eval, all_sentences_endings_eval, reader.normalization_endings

if __name__ == '__main__':
    text_filename = "../data/train_new/named/texts_and_ann/1041141.txt"
    annotation_filename = "../data/train_new/named/texts_and_ann/1041141.ann"
    normalization_filename = "../data/train_new/named/norm/1041141.norm"
    reader = RuNormASReaderForSequenceTagging()
    document, document_entities, document_normalization, document_entities_spans, entity2id = reader.read(text_filename, annotation_filename, normalization_filename)
    sentences, sentences_endings, sentences_entities_ids = reader.parse_entities(document, document_normalization, document_entities_spans, entity2id)

    for sentence, sentence_endings, entities_id in zip(sentences, sentences_endings, sentences_entities_ids):
        for token, ending, entity_id in zip(sentence, sentence_endings, entities_id ):
            print(token, '->', ending, '->', entity_id)