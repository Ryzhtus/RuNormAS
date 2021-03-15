import os
import tqdm
from nltk.stem.snowball import SnowballStemmer
from natasha import Segmenter, Doc

class RuNormASReaderForMachineTranslation():
    def __init__(self):
        self.endings = []
        self.normalization_endings = []

    def read(self, text_filename, annotation_filename, normalization_filename):
        text = open(text_filename, 'r', encoding='utf-8').read()
        annotation = open(annotation_filename, 'r', encoding='utf-8').read().strip().split('\n')
        normalization = open(normalization_filename, 'r', encoding='utf-8').read().strip().split('\n')

        segmenter = Segmenter()

        doc_text = Doc(text)
        doc_text.segment(segmenter)

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
            entry = clean_entity(entry)

            doc = Doc(entry)
            doc.segment(segmenter)

            entities += doc.tokens

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

        return doc_text, entities, normalized_entities, entities_spans

    def parse_entities(self, doc_text, entities, normalized_entities, entities_spans):
        entity2norm = {token.text: norm.text for token, norm in zip(entities, normalized_entities)}
        print(entity2norm)
        sentences = []
        sentences_normalized = []
        for sentence in doc_text.sents:
            sentence_tokens = []
            sentence_norm_tokens = []
            for token in sentence.tokens:
                #if self.find_entity(token, entities_spans):
                print(token)
                if token.text not in entity2norm.keys():
                    sentence_tokens.append(token.text)
                    sentence_norm_tokens.append(token.text)
                else:
                    #id = self.find_entity(token, entities_spans)
                    sentence_tokens.append(token.text)
                    sentence_norm_tokens.append(entity2norm[token])

            sentences.append(sentence_tokens)
            sentences_normalized.append(sentence_norm_tokens)

        return sentences, sentences_normalized

    def find_entity(self, token, entities):
        for idx in range(len(entities)):
            if entities[idx][0] == token.start and entities[idx][1] == token.stop:
                return idx

def collect_sentences_for_machine_translation():
    reader = RuNormASReaderForMachineTranslation()

    all_sentences_train = []
    all_sentences_norm_train = []

    all_sentences_eval = []
    all_sentences_norm_eval = []

    filenames = []
    for _, _, files in os.walk("data/train_new/named/texts_and_ann"):
        for filename in sorted(files):
            filenames.append(filename.split('.')[0])

    filenames = sorted(set(filenames), reverse=True)

    train = filenames[:2000]
    eval = filenames[2000:]

    for filename in tqdm.tqdm(train, total=len(train)):
        text_filename = "data/train_new/named/texts_and_ann/" + filename + ".txt"
        annotation_filename = "data/train_new/named/texts_and_ann/" + filename + ".ann"
        normalization_filename = "data/train_new/named/norm/" + filename + ".norm"
        document, document_entities, document_normalization, document_entities_spans = reader.read(text_filename, annotation_filename, normalization_filename)
        sentences, sentences_norm = reader.parse_entities(document, document_entities, document_normalization,
                                                             document_entities_spans)
        all_sentences_train += sentences
        all_sentences_norm_train += sentences_norm

    for filename in tqdm.tqdm(eval, total=len(eval)):
        text_filename = "data/train_new/named/texts_and_ann/" + filename + ".txt"
        annotation_filename = "data/train_new/named/texts_and_ann/" + filename + ".ann"
        normalization_filename = "data/train_new/named/norm/" + filename + ".norm"
        document, document_entities, document_normalization, document_entities_spans = reader.read(text_filename, annotation_filename, normalization_filename)
        sentences, sentences_norm = reader.parse_entities(document, document_entities, document_normalization,
                                                             document_entities_spans)
        all_sentences_eval += sentences
        all_sentences_norm_eval += sentences_norm

    return all_sentences_train, all_sentences_norm_train, all_sentences_eval, all_sentences_norm_eval

if __name__ == '__main__':
    all_sentences, all_sentences_endings, all_sentences_eval, all_sentences_endings_eval = collect_sentences_for_machine_translation()
    for i in range(10):
        print(all_sentences[i])
        print(all_sentences_endings[i])
        print('-' * 75)

    for i in range(10):
        print(all_sentences_eval[i])
        print(all_sentences_endings_eval[i])
        print('-' * 75)