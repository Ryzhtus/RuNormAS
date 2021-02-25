import os


class RuNormASReader():
    def read(self, text_filename, annotation_filename, normalization_filename):

        document = open(text_filename, 'r').read()
        annotations = open(annotation_filename, 'r').read().split('\n')

        entities_intervals = []
        for interval in annotations:
            if interval != '':
                entity = interval.split(' ')
                entity[0], entity[1] = int(entity[0]), int(entity[1])
                entities_intervals.append(entity)

        normalized_entities = open(normalization_filename, 'r').read().split('\n')
        normalized_entities.pop()

        entities = []
        for entity_interval, normalized_entity in zip(entities_intervals, normalized_entities):
            entities.append(document[entity_interval[0] : entity_interval[1]])
            document_length = len(document)

            # меняется длина, поэтому нужно придумать что-то другое
            # придумал добавлять символ * к каждой сущности, если ее длина < длины интервала
            # потом будем по этому символу чистить
            entity_length = entity_interval[1] - entity_interval[0]
            if len(normalized_entity) < entity_length:
                normalized_entity += ('*' * (entity_length - len(normalized_entity)))
            document = document[:entity_interval[0]] + normalized_entity + document[entity_interval[1]: document_length]

        sentences = open(text_filename, 'r').read().split('\u2028')
        normalized_sentences = document.split('\u2028')

        return sentences, normalized_sentences, entities, normalized_entities


def markup_entities(text, entities):
    sentences, sentences_tags = [], []
    entities = get_unique_entities(entities)

    for sentence in text:
        sentence = sentence.replace('*', '')
        sentence = sentence.split()
        tags = []

        for word in sentence:
            # убираем '*' в случае нормализованных сущностей
            if (word in entities) or (word[:-1] in entities): # убираем последний символ, который может быть пунктуацией
                tags.append('ENTITY')
            else:
                tags.append('O')

        # for word, tag in zip(sentence, tags):
        #    print(word, tag)

        sentences.append(sentence)
        sentences_tags.append(tags)

    return sentences, sentences_tags

def get_unique_entities(document_entities):
    return set([idx for entity in document_entities for idx in entity.split()])

def collect_sentences():
    reader = RuNormASReader()

    sentences = []
    sentences_tags = []

    normalized_sentences = []
    normalized_sentences_tags = []

    filenames = []
    for _, _, files in os.walk("data/train/named/texts_and_ann"):
        for filename in files:
            filenames.append(filename.split('.')[0])

    filenames = set(filenames)

    for filename in filenames:
        text_filename = "data/train/named/texts_and_ann/" + filename + ".txt"
        annotation_filename = "data/train/named/texts_and_ann/" + filename + ".ann"
        normalization_filename = "data/train/named/norm/" + filename + ".norm"
        text, normalized_text, entities, normalized_entities = reader.read(text_filename, annotation_filename, normalization_filename)

        document_sentences, document_tags = markup_entities(text, entities)
        document_normalized_sentences, document_normalized_tags = markup_entities(normalized_text, normalized_entities)

        for sentence, tag in zip(document_sentences, document_tags):
            if sentence != []:
                sentences.append(sentence)
                sentences_tags.append(tag)

        for sentence, tag in zip(document_normalized_sentences, document_normalized_tags):
            if sentence != []:
                normalized_sentences.append(sentence)
                normalized_sentences_tags.append(tag)

    return sentences, sentences_tags, normalized_sentences, normalized_sentences_tags


if __name__ == '__main__':
    sentences, sentences_tags, normalized_sentences, normalized_sentences_tags = collect_sentences()

